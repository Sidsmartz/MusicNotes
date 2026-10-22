"""Piano keyboard geometry from a single frame.

Black keys are the most reliable landmark: dark, elongated blobs with white
keys on both sides, lined up in a row. Their spacing alternates between one
and two white-key widths in the fixed pattern C# D# _ F# G# A# _, which tells
us which black key is which. A 1-D projective fit from "white key number" to
position along the row handles perspective, so every key - black and white -
can be outlined, not only the ones that were detected.

Coordinates: w counts white keys from the left edge of a C (C# sits at w=1,
D# at 2, F# at 4, G# at 5, A# at 6, then +7 per octave). v runs from the back
of the keys (0) to their front edge (1); black keys end at v = 0.63.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Pitch class -> centre of the black key in white-key widths from the left edge of C.
# Black keys are not centred on the white-key boundaries; these are real piano proportions.
BLACK_W = {1: 0.64, 3: 1.79, 6: 3.59, 8: 4.67, 10: 5.75}
WHITE_PCS = [0, 2, 4, 5, 7, 9, 11]
GAP_PATTERN = [1, 2, 1, 1, 2]  # short/long gaps: C#->D#, D#->F#, F#->G#, G#->A#, A#->C#
PATTERN_PCS = [1, 3, 6, 8, 10]
BLACK_DEPTH = 0.63
BLACK_WIDTH = 0.58  # in white-key widths


@dataclass
class KeyboardModel:
    origin: np.ndarray  # point on the black-key row line (image pixels)
    row_dir: np.ndarray  # unit vector along the row, towards higher notes
    axis: np.ndarray  # unit vector along the keys, from back to front
    h: np.ndarray  # projective map w -> distance along the row: (h0*w + h1) / (h2*w + 1)
    blen: np.ndarray  # black key length as a linear function of distance: blen0 + blen1 * dist
    w_min: float
    w_max: float
    c_midi: int  # midi number of the C whose left edge is w = 0
    confidence: float
    seen_min: float = 0.0  # white-key range actually covered by detected black keys
    seen_max: float = 0.0

    def seen(self, midi: int) -> bool:
        w, _ = self.key_w(midi)
        return self.seen_min - 0.5 <= w <= self.seen_max + 0.5

    def _dist(self, w):
        w = np.asarray(w, float)
        return (self.h[0] * w + self.h[1]) / (self.h[2] * w + 1)

    def point(self, w: float, v: float) -> np.ndarray:
        d = float(self._dist(w))
        lb = max(2.0, self.blen[0] + self.blen[1] * d)
        lw = lb / BLACK_DEPTH
        return self.origin + self.row_dir * d + self.axis * (v * lw - lb / 2)

    def key_w(self, midi: int) -> tuple[float, bool]:
        """Centre of a key in white-key units, and whether it is black."""
        octave, pc = divmod(midi - self.c_midi, 12)
        if pc in BLACK_W:
            return octave * 7 + BLACK_W[pc], True
        return octave * 7 + WHITE_PCS.index(pc) + 0.5, False

    def has_key(self, midi: int) -> bool:
        w, _ = self.key_w(midi)
        return self.w_min - 0.5 <= w <= self.w_max + 0.5

    def key_polygon(self, midi: int) -> np.ndarray:
        w, black = self.key_w(midi)
        hb = BLACK_WIDTH / 2
        if black:
            uv = [(w - hb, 0), (w + hb, 0), (w + hb, BLACK_DEPTH), (w - hb, BLACK_DEPTH)]
        else:
            l, r = w - 0.5, w + 0.5
            pc = (midi - self.c_midi) % 12
            octave = np.floor((w - 0.5) / 7) * 7
            # Notches where the neighbouring black keys overlap this white key.
            ln = max(0.0, octave + BLACK_W[(pc - 1) % 12] + hb - l) if (pc - 1) % 12 in BLACK_W else 0.0
            rn = max(0.0, r - (octave + BLACK_W[(pc + 1) % 12] - hb)) if (pc + 1) % 12 in BLACK_W else 0.0
            d = BLACK_DEPTH
            uv = [(l + ln, 0), (r - rn, 0), (r - rn, d), (r, d), (r, 1), (l, 1), (l, d), (l + ln, d)]
        return np.array([self.point(a, b) for a, b in uv])

    def key_size(self) -> float:
        """Typical white-key width in pixels."""
        mid = (self.w_min + self.w_max) / 2
        return float(np.linalg.norm(self.point(mid + 0.5, 1) - self.point(mid - 0.5, 1)))

    def midi_range(self) -> tuple[int, int]:
        lo = self.c_midi + int(np.floor(self.w_min / 7)) * 12 - 12
        hi = self.c_midi + int(np.ceil(self.w_max / 7)) * 12 + 12
        keys = [m for m in range(max(21, lo), min(108, hi) + 1) if self.has_key(m)]
        return (keys[0], keys[-1]) if keys else (60, 72)

    def x_of(self, midi: int) -> np.ndarray:
        w, black = self.key_w(midi)
        return self.point(w, 0.3 if black else 0.8)


def _black_key_candidates(gray: np.ndarray, k: int, strip: str = ""):
    """Black keys at one scale. A morphological closing of the bright (white-key)
    area fills narrow dark gaps - the black keys - but not wide dark areas like
    the fallboard behind the keys, so dark pixels inside the closed area are
    black keys and nothing else."""
    h, w = gray.shape
    thr, _ = cv2.threshold(cv2.GaussianBlur(gray, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright = (gray > thr).astype(np.uint8)
    closed = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    dark = ((closed > 0) & (bright == 0)).astype(np.uint8)
    # Drop seams between white keys: anything much thinner than a black key at this scale.
    o = max(2, k // 5)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (o, o)))
    if strip:
        # Black keys often touch a dark band behind the keys, forming one comb-shaped blob.
        # Remove long bands running across the keys so the teeth come apart.
        size = (3 * k, 1) if strip == "h" else (1, 3 * k)
        band = cv2.morphologyEx(dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, size))
        band = cv2.dilate(band, np.ones((3, 3), np.uint8))
        dark = dark & (1 - band)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=4)
    cands = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 20 or area > 0.04 * h * w:
            continue
        x, y, bw, bh = stats[i, :4]
        ys, xs = np.nonzero(labels[y:y + bh, x:x + bw] == i)
        pts = np.column_stack([xs + x, ys + y]).astype(np.float32)
        (cx, cy), (rw, rh), ang = cv2.minAreaRect(pts)
        long_, short = max(rw, rh), min(rw, rh)
        if short < 2 or long_ < 2.0 * short or long_ > 10 * short or area < 0.6 * rw * rh:
            continue
        theta = np.radians(ang if rw >= rh else ang + 90)
        axis = np.array([np.cos(theta), np.sin(theta)])
        cands.append(dict(c=np.array([cx, cy]), axis=axis, length=long_, width=short))
    return cands


def estimate_keyboard(img: np.ndarray, box: np.ndarray) -> KeyboardModel | None:
    H, W = img.shape[:2]
    x0, y0, x1, y1 = box
    pad = 0.03 * max(x1 - x0, y1 - y0)
    cx0, cy0, cx1, cy1 = int(max(0, x0 - pad)), int(max(0, y0 - pad)), int(min(W, x1 + pad)), int(min(H, y1 + pad))
    crop = img[cy0:cy1, cx0:cx1]
    if crop.size == 0:
        return None
    scale = min(1.0, 1000 / max(crop.shape[:2]))
    if scale < 1:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    best = None
    for k in (7, 11, 17, 25, 35, 51, 71):
        for strip in ("", "h", "v"):
            model = _fit(gray, _black_key_candidates(gray, k, strip))
            if model is not None and (best is None or model[1] > best[1]):
                best = model
    if best is None:
        return None
    (mean, row, axis, hpar, blen, wpos, match), _ = best
    return _to_image(mean, row, axis, hpar, blen, wpos, match, scale, cx0, cy0)


def _fit(gray, cands):
    """Fits the key pattern to black-key candidates; returns (params, quality) or None."""
    if len(cands) < 5:
        return None
    # Black keys are the largest consistent blobs; seams and specks are much smaller.
    lens_all = np.array(sorted((c["length"] for c in cands), reverse=True))
    ref_len = np.median(lens_all[:8])
    ref_w = np.median(sorted((c["width"] for c in cands), reverse=True)[:8])
    cands = [c for c in cands if 0.45 * ref_len <= c["length"] <= 1.8 * ref_len and 0.35 * ref_w <= c["width"] <= 2.0 * ref_w]
    if len(cands) < 5:
        return None

    # Keys share an orientation (up to perspective): keep the dominant one.
    angs = np.array([np.mod(np.arctan2(c["axis"][1], c["axis"][0]), np.pi) for c in cands])
    ref = angs[np.argmax([np.sum(np.abs(np.angle(np.exp(2j * (angs - a)))) < np.radians(25)) for a in angs])]
    cands = [c for c, a in zip(cands, angs) if abs(np.angle(np.exp(2j * (a - ref)))) / 2 < np.radians(12)]
    if len(cands) < 5:
        return None
    centres = np.array([c["c"] for c in cands])
    lengths = np.array([c["length"] for c in cands])

    # RANSAC: the black keys' centres lie on one line.
    rng = np.random.default_rng(0)
    key_dir = np.median(np.array([c["axis"] for c in cands]) * np.sign(np.array([c["axis"] for c in cands]) @ cands[0]["axis"])[:, None], 0)
    key_dir /= np.linalg.norm(key_dir)
    best_in, best_err = None, np.inf
    for _ in range(400):
        i, j = rng.choice(len(centres), 2, replace=False)
        d = centres[j] - centres[i]
        if np.linalg.norm(d) < 1:
            continue
        d /= np.linalg.norm(d)
        if abs(d @ key_dir) > 0.6:  # the row runs across the keys, not along them
            continue
        nrm = np.array([-d[1], d[0]])
        dist = np.abs((centres - centres[i]) @ nrm)
        inl = dist < 0.2 * np.median(lengths)
        err = dist[inl].mean() if inl.any() else np.inf
        if best_in is None or inl.sum() > best_in.sum() or (inl.sum() == best_in.sum() and err < best_err):
            best_in, best_err = inl, err
    if best_in is None or best_in.sum() < 5:
        return None
    pts, lens = centres[best_in], lengths[best_in]
    axes = np.array([c["axis"] for c, k in zip(cands, best_in) if k])
    mean = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - mean)
    row = vt[0]
    if row[0] < 0:
        row = -row
    lam = (pts - mean) @ row
    order = np.argsort(lam)
    lam, lens, pts = lam[order], lens[order], pts[order]

    # Gap pattern -> which black key is which.
    gaps = np.diff(lam)
    if len(gaps) < 4:
        return None
    units = np.empty(len(gaps))
    for k in range(len(gaps)):
        window = gaps[max(0, k - 3):k + 4]
        units[k] = np.percentile(window, 30)
    # Long gaps (across E-F or B-C) are about 1.65x the short ones; much larger means keys were missed.
    ratio = gaps / units
    steps = np.where(ratio > 1.3, 2, 1)
    steps[(ratio < 0.6) | (ratio > 2.6)] = 0
    if np.any(steps == 0):
        # Missing or merged keys break the pattern: keep the longest clean run.
        good = (steps >= 1) & (steps <= 2)
        runs, start = [], 0
        for k in range(len(good) + 1):
            if k == len(good) or not good[k]:
                runs.append((start, k))
                start = k + 1
        a, b = max(runs, key=lambda r: r[1] - r[0])
        if b - a < 4:
            return None
        lam, lens, pts, steps = lam[a:b + 1], lens[a:b + 1], pts[a:b + 1], steps[a:b]
    best = None
    for off in range(5):
        expect = [GAP_PATTERN[(off + k) % 5] for k in range(len(steps))]
        score = np.mean(np.array(expect) == steps)
        if best is None or score > best[0]:
            best = (score, off)
    match, off = best
    if match < 0.8:
        return None
    # White-key coordinate of each black key.
    wpos, octave, idx = [], 0, off
    for k in range(len(lam)):
        pc = PATTERN_PCS[idx % 5]
        wpos.append(octave * 7 + BLACK_W[pc])
        if idx % 5 == 4:
            octave += 1
        idx += 1
    wpos = np.array(wpos, float)

    # w -> lam: a straight scale, or a 1-D homography when the keyboard recedes
    # in perspective. The homography is only used when it clearly fits better.
    a, b = np.polyfit(wpos, lam, 1)
    hpar = np.array([a, b, 0.0])
    lin_err = np.sqrt(np.mean((a * wpos + b - lam) ** 2))
    if len(wpos) >= 6:
        A = np.column_stack([wpos, np.ones_like(wpos), -wpos * lam])
        hp, *_ = np.linalg.lstsq(A, lam, rcond=None)
        proj_err = np.sqrt(np.mean(((hp[0] * wpos + hp[1]) / (hp[2] * wpos + 1) - lam) ** 2))
        if proj_err < 0.6 * lin_err and lin_err > 0.08 * abs(a):
            hpar = hp
    # Black-key length along the row: only keep a trend that is clearly there
    # (perspective); otherwise noise would tilt the keys once extrapolated.
    blen = np.array([0.0, np.median(lens)])
    if len(lam) >= 5:
        i, j = np.triu_indices(len(lam), 1)
        slope = float(np.median((lens[j] - lens[i]) / (lam[j] - lam[i] + 1e-9)))  # Theil-Sen
        if abs(slope) * (lam.max() - lam.min()) > 0.4 * np.median(lens):
            blen = np.array([slope, np.median(lens) - slope * np.median(lam)])

    # Which end of a black key is the front? The white key fronts are bright.
    axis = np.median(axes * np.sign(axes @ axes[0])[:, None], axis=0)
    axis /= np.linalg.norm(axis)
    votes = 0
    for p, L in zip(pts, lens):
        vals = []
        for sgn in (1, -1):
            q = p + axis * sgn * L * 0.85
            x, y = int(round(q[0])), int(round(q[1]))
            vals.append(gray[y, x] if 0 <= x < gray.shape[1] and 0 <= y < gray.shape[0] else 0)
        votes += 1 if vals[0] > vals[1] else -1
    if votes < 0:
        axis = -axis
    # Sanity: keys run across the row, and keep a positive length along it.
    if abs(float(axis @ row)) > 0.75:
        return None
    span = np.array([lam.min() - 3 * np.median(np.diff(lam)), lam.max() + 3 * np.median(np.diff(lam))])
    if np.any(np.polyval(blen, span) < 0.3 * np.median(lens)):
        return None
    # Prefer a keyboard with many, large keys (not a tiny picture of a keyboard nearby).
    return (mean, row, axis, hpar, blen, wpos, match), match * len(wpos) * np.sqrt(np.median(lens))


def _to_image(mean, row, axis, hpar, blen, wpos, match, scale, cx0, cy0) -> KeyboardModel:
    inv = 1 / scale
    off_px = np.array([cx0, cy0], float)
    model = KeyboardModel(
        origin=mean * inv + off_px, row_dir=row, axis=axis,
        h=np.array([hpar[0] * inv, hpar[1] * inv, hpar[2]]), blen=np.array([blen[1] * inv, blen[0]]),
        w_min=float(wpos.min() - 7), w_max=float(wpos.max() + 7), c_midi=60, confidence=float(match),
        seen_min=float(wpos.min() - 1), seen_max=float(wpos.max() + 1),
    )
    # Default octave: the C nearest the middle of what is visible is middle C.
    mid_c = int(np.round(((model.w_min + model.w_max) / 2 - 3.5) / 7))
    model.c_midi = 60 - 12 * mid_c
    return model
