"""Guitar neck geometry from a single frame.

Given the guitar's bounding box (from the detector) we:
  1. find the neck direction as the dominant orientation of long straight lines
     (strings and neck edges are the longest parallel lines on a guitar);
  2. measure how wide the instrument is along that axis; the narrow end is the
     neck, the wide end the body (the player's wrists break ties);
  3. locate the neck's centre line and width from the lines along the neck;
  4. fit the fret rule, fret n at scale * (1 - 2^(-n/12)) from the nut, to the
     fret-edge profile along the neck, falling back to typical guitar
     proportions when the frets are not clearly visible.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Typical acoustic proportions as a fraction of overall length (headstock tip to tail).
NUT_AT = 0.17
SCALE_OF_LENGTH = 0.63


def fret_u(n: float) -> float:
    """Position of fret n in units where the nut is 0 and the 12th fret is 1."""
    return 2.0 * (1.0 - 2.0 ** (-n / 12.0))


def string_v(s: int) -> float:
    """Across-neck position of string s (0 = low E) between the neck edges (0..1)."""
    return 0.1 + s * 0.16


@dataclass
class NeckModel:
    nut: np.ndarray  # centre of the nut, image pixels
    fret12: np.ndarray  # centre of the neck at the 12th fret
    width_nut: float
    width_12: float
    low_side: np.ndarray  # unit vector from the neck centre towards the low E edge
    confidence: float  # 0..1, how sure the fret fit is
    frets_fitted: bool

    def point(self, u: float, v: float) -> np.ndarray:
        """u: along the neck in fret units (0 nut, 1 12th fret); v: 0 low E edge .. 1 high E edge."""
        c = self.nut + (self.fret12 - self.nut) * u
        w = self.width_nut + (self.width_12 - self.width_nut) * u
        return c + self.low_side * (0.5 - v) * w

    def fret_point(self, fret: int, string: int) -> np.ndarray:
        """Where a finger presses: just behind the fret wire."""
        if fret <= 0:
            return self.point(-0.05, string_v(string))
        u = fret_u(fret - 1) + 0.62 * (fret_u(fret) - fret_u(fret - 1))
        return self.point(u, string_v(string))

    def as_array(self) -> np.ndarray:
        return np.array([*self.nut, *self.fret12, self.width_nut, self.width_12, *self.low_side, self.confidence])

    @classmethod
    def from_array(cls, a: np.ndarray, fitted: bool = True) -> "NeckModel":
        side = a[6:8] / (np.linalg.norm(a[6:8]) or 1)
        return cls(a[0:2].copy(), a[2:4].copy(), float(a[4]), float(a[5]), side, float(a[8]), fitted)


def _dominant_angle(lines: np.ndarray) -> float | None:
    if lines is None or not len(lines):
        return None
    x0, y0, x1, y1 = lines.T.astype(float)
    ang = np.mod(np.arctan2(y1 - y0, x1 - x0), np.pi)
    length = np.hypot(x1 - x0, y1 - y0)
    hist, edges = np.histogram(ang, bins=180, range=(0, np.pi), weights=length ** 1.5)
    k = np.array([1, 2, 4, 2, 1], float)
    smooth = np.convolve(np.concatenate([hist[-2:], hist, hist[:2]]), k / k.sum(), mode="valid")
    i = int(np.argmax(smooth))
    # Refine with the length-weighted mean of lines near the peak.
    centre = (edges[i] + edges[i + 1]) / 2
    diff = np.angle(np.exp(2j * (ang - centre))) / 2
    near = np.abs(diff) < np.radians(3)
    if near.any():
        centre += np.average(diff[near], weights=length[near])
    return float(centre)


def _profile_peaks_score(profile: np.ndarray, positions: np.ndarray, min_count: int = 8) -> float:
    idx = positions.astype(int)
    ok = (idx >= 0) & (idx < len(profile) - 1)
    if ok.sum() < min_count:
        return -np.inf
    frac = positions[ok] - idx[ok]
    vals = profile[idx[ok]] * (1 - frac) + profile[idx[ok] + 1] * frac
    return float(vals.mean())


def _sample(profile: np.ndarray, positions) -> np.ndarray:
    pos = np.clip(np.atleast_1d(np.asarray(positions, float)), 0, len(profile) - 1.001)
    i = pos.astype(int)
    f = pos - i
    return profile[i] * (1 - f) + profile[i + 1] * f


def estimate_neck(img: np.ndarray, box: np.ndarray, wrists: list[np.ndarray] | None = None) -> NeckModel | None:
    H, W = img.shape[:2]
    x0, y0, x1, y1 = box
    pad = 0.08 * max(x1 - x0, y1 - y0)
    cx0, cy0 = int(max(0, x0 - pad)), int(max(0, y0 - pad))
    cx1, cy1 = int(min(W, x1 + pad)), int(min(H, y1 + pad))
    crop = img[cy0:cy1, cx0:cx1]
    if crop.size == 0 or min(crop.shape[:2]) < 40:
        return None
    scale = min(1.0, 720 / max(crop.shape[:2]))
    if scale < 1:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 40, 120)
    h, w = gray.shape
    diag = float(np.hypot(h, w))

    lines = None
    for frac in (0.25, 0.15, 0.08):
        lines = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=30, minLineLength=int(frac * diag), maxLineGap=int(0.01 * diag) + 3)
        if lines is not None:
            lines = lines.reshape(-1, 4)  # OpenCV versions differ in the extra axis
            if len(lines) >= 6:
                break
    theta = _dominant_angle(lines)
    if theta is None:
        return None
    d = np.array([np.cos(theta), np.sin(theta)])
    n = np.array([-d[1], d[0]])

    # Instrument extent along the axis, from edge pixels inside the detection box.
    bx0, by0 = (x0 - cx0) * scale, (y0 - cy0) * scale
    bx1, by1 = (x1 - cx0) * scale, (y1 - cy0) * scale
    ys, xs = np.nonzero(edges)
    inside = (xs >= bx0) & (xs <= bx1) & (ys >= by0) & (ys <= by1)
    xs, ys = xs[inside], ys[inside]
    if len(xs) < 200:
        return None
    pts = np.column_stack([xs, ys]).astype(float)
    s, t = pts @ d, pts @ n
    smin, smax = np.percentile(s, [1, 99])
    length = smax - smin
    if length < 0.3 * diag:
        return None

    # Width profile along the axis: the neck end is the narrow end.
    bins = np.linspace(smin, smax, 11)
    widths = []
    for a, b in zip(bins[:-1], bins[1:]):
        sel = (s >= a) & (s < b)
        widths.append(np.subtract(*np.percentile(t[sel], [90, 10])) if sel.sum() > 20 else np.nan)
    widths = np.array(widths)
    first, last = np.nanmean(widths[:3]), np.nanmean(widths[-3:])
    ratio = min(first, last) / max(first, last) if np.isfinite(first) and np.isfinite(last) and max(first, last) > 0 else 1.0
    neck_at_min = first < last
    if wrists is not None and len(wrists) and ratio > 0.8:
        # Ambiguous shape: the fretting hand sits on the neck, far from the body centre.
        ws = np.array([(np.asarray(p) - [cx0, cy0]) * scale for p in wrists]) @ d
        mid = (smin + smax) / 2
        far = ws[np.argmax(np.abs(ws - mid))]
        neck_at_min = far < mid
    if not neck_at_min:
        d, n = -d, -n
        s, t = -s, -t
        smin, smax = -smax, -smin

    geo = dict(gray=gray, lines=lines, theta=theta, d=d, n=n, smin=smin, smax=smax, length=length,
               box_t=float(np.array([(bx0 + bx1) / 2, (by0 + by1) / 2]) @ n))

    def to_model(fit, low_hint_n):
        d_, n_, smin_, centre_t, width, nut_s, scale_len, z, fitted, closeup = fit[:10]

        def to_img(s_val, t_val):
            return (d_ * (smin_ + s_val) + n_ * t_val) / scale + [cx0, cy0]

        width_img = width / scale
        # Low E is the string nearer the player's head, which is usually higher in the frame.
        low = -n_ if n_[1] > 0 else n_
        if abs(n_[1]) < 0.3:  # near-vertical neck: assume low E on the left
            low = -n_ if n_[0] > 0 else n_
        confidence = float(np.clip((z - 2.0) / 3.0, 0, 1))
        w_nut, w_12 = (width_img, width_img) if closeup else (width_img * 0.92, width_img * 1.1)
        return NeckModel(to_img(nut_s, centre_t), to_img(nut_s + scale_len / 2, centre_t), w_nut, w_12, low, confidence, fitted)

    # Two hypotheses: a whole guitar, or a close-up showing only part of the neck
    # (say, the fretting hand). A close-up shows a few widely spaced frets that all
    # line up exactly, so it must explain the edges almost perfectly (F1 >= 0.9)
    # and clearly better than the whole-guitar fit.
    full = _fit_view(geo, closeup=False)
    close = _fit_view(geo, closeup=True)
    if close[8] and close[10] >= 0.9 and close[10] - full[10] >= 0.25 and close[6] > 1.2 * length:
        return to_model(close, n)
    if full[8]:
        return to_model(full, n)
    if ratio > 0.7 and np.nanmax(widths) < 0.4 * length:
        return None  # looks like a close-up, but no frets could be placed
    d_, n_, smin_, centre_t, width, _, _, z, _, _, _ = full
    return to_model((d_, n_, smin_, centre_t, width, NUT_AT * length, SCALE_OF_LENGTH * length, z, False, False), n)


def _fit_view(geo: dict, closeup: bool):
    """Neck centre line, width and fret fit for one hypothesis (whole guitar or
    close-up of the neck). Returns (d, n, smin, centre_t, width, nut_s, scale, z, fitted, closeup)."""
    gray, lines, theta = geo["gray"], geo["lines"], geo["theta"]
    d, n, smin, smax, length, box_t = geo["d"], geo["n"], geo["smin"], geo["smax"], geo["length"], geo["box_t"]
    seg = lines.astype(float)
    a_pts, b_pts = seg[:, :2], seg[:, 2:]
    ang = np.mod(np.arctan2(b_pts[:, 1] - a_pts[:, 1], b_pts[:, 0] - a_pts[:, 0]) - theta, np.pi)
    parallel = (ang < np.radians(4)) | (ang > np.pi - np.radians(4))
    mid_pts = (a_pts + b_pts) / 2
    ms, mt = mid_pts @ d, mid_pts @ n
    # The guitar's axis passes close to the centre of its box, so ignore parallel
    # lines far from it (wall panels, stripes, mic stands).
    if closeup:
        on_neck = parallel & (np.abs(mt - box_t) < 0.35 * length)
        w_lo, w_hi, w_default = 0.06 * length, 0.5 * length, 0.2 * length
    else:
        on_neck = parallel & (ms > smin + 0.1 * length) & (ms < smin + 0.55 * length) & (np.abs(mt - box_t) < 0.16 * length)
        w_lo, w_hi, w_default = 0.035 * length, 0.11 * length, 0.055 * length
    neck_t = mt[on_neck]
    if len(neck_t) >= 2:
        lo, hi = np.percentile(neck_t, [10, 90])
        centre_t = float(np.median(neck_t))
        width = float(np.clip(hi - lo, w_lo, w_hi))
    else:
        centre_t, width = box_t, w_default

    # Sample a strip along the neck and look for edges across it.
    strip_len = int(length if closeup else 0.75 * length)
    strip_w = max(8, int(width * 0.8))
    origin = d * smin + n * (centre_t - strip_w / 2)
    M = np.array([[d[0], n[0], origin[0]], [d[1], n[1], origin[1]]])  # strip -> crop
    strip = cv2.warpAffine(gray, M, (strip_len, strip_w), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    gx = np.abs(cv2.Sobel(strip.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3))
    profile = np.median(gx, axis=0)  # a fret crosses the whole strip; noise does not
    # Where the fretboard meets the headstock the wood usually changes: a step in brightness.
    level = cv2.GaussianBlur(np.median(strip.astype(np.float32), axis=0).reshape(1, -1), (0, 0), 2).ravel()
    step = np.abs(np.roll(level, -5) - np.roll(level, 5))
    step[:5] = step[-5:] = 0
    step = step / (step.std() + 1e-6)
    profile = profile - cv2.blur(profile.reshape(1, -1), (31, 1)).ravel()
    profile = np.maximum(profile, 0)
    # A fret shows up as two sharp edges a few pixels apart; blur them into one
    # bump so the grid search cannot fall between them.
    profile = cv2.GaussianBlur(profile.reshape(1, -1).astype(np.float32), (0, 0), 1.8).ravel()
    if profile.std() > 0:
        profile /= profile.std()

    if closeup:
        # Either end could be the headstock. Frets get closer together towards the
        # body, so fitting the fret law both ways round tells us which is which.
        fits = [(_fit_frets(profile, length, step, closeup=True), False),
                (_fit_frets(profile[::-1].copy(), length, step[::-1].copy(), closeup=True), True)]
        (best, z), flipped = max(fits, key=lambda f: f[0][1])
        if flipped:
            d, n = -d, -n
            smin = -smax
            centre_t = -centre_t
            profile = profile[::-1].copy()
    else:
        best, z = _fit_frets(profile, length, step, closeup=False)
        if best is None or z <= 3.2:
            # The nut may be out of frame (a shot of the body with part of the neck).
            alt, z_alt = _fit_frets(profile, length, step, closeup=False, nut_offscreen=True)
            if alt is not None and z_alt > 4.0:
                best, z = alt, z_alt
    fitted = best is not None and z > 3.2
    nut_s, scale_len = (best[1], best[2]) if fitted else (NUT_AT * length, SCALE_OF_LENGTH * length)
    f1 = _fret_f1(profile, nut_s, scale_len) if fitted else 0.0
    return d, n, smin, centre_t, width, nut_s, scale_len, z, fitted, closeup, f1


def _fret_f1(profile: np.ndarray, nut: float, scale_len: float) -> float:
    """How well a fret hypothesis explains the strip: F1 of predicted frets landing
    on edge peaks, and of edge peaks (between the outer predicted frets) being
    explained. Unlike the z-score it means the same for any hypothesis."""
    n = len(profile)
    peaks = np.array([i for i in range(2, n - 2) if profile[i] > 1.0 and profile[i] == profile[i - 2:i + 3].max()])
    pos = nut + scale_len * (1 - 2.0 ** (-np.arange(0, 20) / 12))
    gaps = np.diff(np.append(pos, pos[-1] + (pos[-1] - pos[-2])))
    inside = (pos >= 0) & (pos < n)
    if inside.sum() < 3 or not len(peaks):
        return 0.0
    pos, tol = pos[inside], np.maximum(2.0, 0.25 * gaps[inside])
    hit = np.array([np.min(np.abs(peaks - p)) <= t for p, t in zip(pos, tol)])
    region = peaks[(peaks >= pos.min() - tol.max()) & (peaks <= pos.max() + tol.max())]
    if not len(region):
        return 0.0
    explained = np.array([np.min(np.abs(pos - q) - tol) <= 0 for q in region])
    precision, recall = hit.mean(), explained.mean()
    return float(2 * precision * recall / (precision + recall + 1e-9))


def _fit_frets(profile: np.ndarray, length: float, step: np.ndarray, closeup: bool, nut_offscreen: bool = False):
    """Grid search for the nut position and scale length that best explain the
    fret edges in `profile`. Returns ((total, nut, scale, base), z)."""
    ks = np.arange(0, 20 if nut_offscreen else 13)
    if nut_offscreen:
        nuts = np.arange(-1.2, 0.0, 0.006) * length
        scales = np.geomspace(0.6, 2.5, 90) * length
        min_count = 5
    elif closeup:
        nuts = np.arange(-0.05, 0.5, 0.003) * length
        scales = np.geomspace(0.8, 5.0, 160) * length
        min_count = 4
    else:
        nuts = np.arange(0.06, 0.30, 0.003) * length
        scales = np.arange(0.48, 0.80, 0.003) * length
        min_count = 8
    best, bases = None, []
    n = len(profile)
    for nut in nuts:
        for scale_len in scales:
            if not closeup and not nut_offscreen and nut + scale_len / 2 > 0.62 * length:
                continue
            pos = nut + scale_len * (1 - 2.0 ** (-ks / 12))
            base = _profile_peaks_score(profile, pos, min_count)
            if not np.isfinite(base):
                continue
            # The nut is a strong edge, and there are no frets beyond it on the headstock
            # side; without this the fit often lands one fret too far up the neck.
            total = base
            if 0 <= nut < n:
                ghost = _sample(profile, nut + scale_len * (1 - 2.0 ** (np.array([1, 2]) / 12)))
                total += 0.5 * float(_sample(profile, nut)[0]) - 1.2 * float(ghost[0]) - 0.4 * float(ghost[1]) \
                    + 0.4 * float(_sample(step, nut)[0])
            bases.append(base)
            if best is None or total > best[0]:
                best = (total, nut, scale_len, base)
    if best is None or len(bases) < 10:
        return None, 0.0
    bases = np.array(bases)
    # Confidence: how far the chosen frets stand out from all other alignments.
    return best, float((best[3] - bases.mean()) / (bases.std() + 1e-9))
