"""Video-level vision: which instrument is in the video and where it is over time.

Frames are sampled a few times per second. On each sample we run the
instrument detector and the pose model, then the instrument-specific geometry
(guitar neck, piano keys). Results are smoothed over time with outlier
rejection, so a bad frame does not make the overlay jump.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import median_filter

from .. import media
from .detector import Detection, InstrumentDetector, PoseDetector
from .drums import DrumPiece, kit_layout
from .guitar import NeckModel, estimate_neck
from .piano import KeyboardModel, estimate_keyboard

KIND_NAMES = {"guitar": "Guitar", "piano": "Piano", "drums": "Drums", "melody": "Melody instrument"}


@dataclass
class VisionResult:
    size: tuple[int, int]
    kind: str | None  # what the camera saw, or None
    label: str  # detector's class name for the main instrument
    confidence: float
    coverage: float  # share of sampled frames where it was seen
    times: np.ndarray = field(default_factory=lambda: np.zeros(0))
    boxes: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))  # smoothed, per sample
    neck: np.ndarray | None = None  # smoothed neck parameters per sample (NeckModel.as_array)
    neck_fitted: float = 0.0  # share of samples where frets were found
    keyboard: KeyboardModel | None = None
    keyboard_time: float = 0.0
    drums: list[DrumPiece] = field(default_factory=list)
    wrists: list[np.ndarray] = field(default_factory=list)  # per sample, (k, 2)
    # Other views of the same instrument, e.g. the panels of a split-screen tutorial.
    views: list["VisionResult"] = field(default_factory=list)

    def _interp(self, arr: np.ndarray, t: float) -> np.ndarray:
        if len(self.times) == 1:
            return arr[0]
        return np.array([np.interp(t, self.times, arr[:, j]) for j in range(arr.shape[1])])

    def box_at(self, t: float) -> np.ndarray | None:
        return self._interp(self.boxes, t) if len(self.times) else None

    def neck_at(self, t: float) -> NeckModel | None:
        if self.neck is None:
            return None
        return NeckModel.from_array(self._interp(self.neck, t), self.neck_fitted > 0.3)

    def shift_at(self, t: float, since: float) -> np.ndarray:
        """How far the instrument's box centre moved between two times (camera or player motion)."""
        if not len(self.times):
            return np.zeros(2)
        b1, b0 = self.box_at(t), self.box_at(since)
        return (b1[:2] + b1[2:]) / 2 - (b0[:2] + b0[2:]) / 2

    def keyboard_at(self, t: float) -> KeyboardModel | None:
        if self.keyboard is None:
            return None
        k = self.keyboard
        shift = self.shift_at(t, self.keyboard_time)
        return KeyboardModel(k.origin + shift, k.row_dir, k.axis, k.h, k.blen, k.w_min, k.w_max, k.c_midi, k.confidence, k.seen_min, k.seen_max)

    def wrists_at(self, t: float) -> np.ndarray:
        if not self.wrists:
            return np.zeros((0, 2))
        return self.wrists[int(np.argmin(np.abs(self.times - t)))]


def _fill_and_smooth(values: np.ndarray, valid: np.ndarray, window: int = 5) -> np.ndarray:
    """Linear interpolation over gaps, then a median filter per column."""
    out = values.copy()
    idx = np.arange(len(values))
    if not valid.any():
        return out
    for j in range(values.shape[1]):
        out[:, j] = np.interp(idx, idx[valid], values[valid, j])
    if len(out) >= 3:
        out = median_filter(out, size=(min(window, len(out) | 1), 1), mode="nearest")
    return out


def _pick_box(dets: list[Detection], kind: str, prev: np.ndarray | None) -> Detection | None:
    cands = [d for d in dets if d.kind == kind]
    if not cands:
        return None

    def iou(a, b):
        ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        i = ix * iy
        return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)

    if kind == "piano":
        # Prefer the tighter "Musical keyboard" box over the whole piano.
        keys = [d for d in cands if d.label == "Musical keyboard"]
        cands = keys or cands
    return max(cands, key=lambda d: d.score * np.sqrt(d.area) * (1 + (2 * iou(d.box, prev) if prev is not None else 0)))


def analyze_video(path: str, info: media.VideoInfo, forced_kind: str | None = None, sample_fps: float = 3.0,
                  max_side: int = 1280, detector: InstrumentDetector | None = None, pose: PoseDetector | None = None,
                  progress=None) -> VisionResult:
    report = progress or (lambda p, msg="": None)
    size = media.output_size(info, max_side)
    if not info.has_video:
        return VisionResult(size, None, "", 0.0, 0.0)
    detector = detector or InstrumentDetector()
    pose = pose or PoseDetector()
    fps = min(sample_fps, max(0.5, 240 / max(info.duration, 1)))  # cap the number of samples on long videos
    expected = max(1, int(info.duration * fps))

    # One pass over sampled frames. Geometry is estimated on the spot for any
    # guitar or keyboard in view, so frames need not be kept in memory.
    dets, people, guitars, keyboards = [], [], [], []
    for i, frame in enumerate(media.FrameReader(path, size, fps)):
        ds, ps = detector(frame), pose(frame)
        dets.append(ds)
        people.append(ps)
        # Every guitar in view gets a neck estimate: split-screen videos show the
        # same guitar more than once (say, a close-up of the fretting hand).
        found = []
        if forced_kind in (None, "guitar"):
            for g in sorted((d for d in ds if d.kind == "guitar"), key=lambda d: -d.score)[:3]:
                found.append((g, estimate_neck(frame, g.box, list(_wrists(ps)))))
        guitars.append(found)
        k = _pick_box(ds, "piano", None) if forced_kind in (None, "piano") else None
        if k is not None and (len(keyboards) < 4 or i % 3 == 0):
            km = estimate_keyboard(frame, k.box)
            if km is not None:
                keyboards.append((km, (i + 0.5) / fps))
        report(min(0.95, 0.95 * (i + 1) / expected), "Looking for the instrument")
    frames = dets  # one entry per sample
    times = (np.arange(len(frames)) + 0.5) / fps
    W, H = size

    # Main instrument: consistently seen, confident and large.
    scores: dict[str, float] = {}
    labels: dict[str, dict[str, float]] = {}
    for ds in dets:
        best: dict[str, Detection] = {}
        for d in ds:
            if d.kind and (d.kind not in best or d.score > best[d.kind].score):
                best[d.kind] = d
        for k, d in best.items():
            scores[k] = scores.get(k, 0.0) + d.score * np.sqrt(d.area / (W * H))
            labels.setdefault(k, {})[d.label] = labels.get(k, {}).get(d.label, 0) + d.score
    seen_kind = max(scores, key=scores.get) if scores else None
    kind = forced_kind or seen_kind
    if kind is None:
        return VisionResult(size, None, "", 0.0, 0.0, times=times, wrists=[_wrists(p) for p in people])
    label = max(labels.get(kind, {"": 1}).items(), key=lambda kv: kv[1])[0]

    wrists = [_wrists(p) for p in people]
    if kind == "guitar":
        views = _guitar_views(guitars, times, size, label, wrists)
        if not views:
            return VisionResult(size, kind if forced_kind else None, label, 0.0, 0.0, times=times, wrists=wrists)
        report(1.0, "Instrument found")
        main, *rest = views
        main.views = rest
        return main

    boxes = np.zeros((len(frames), 4))
    valid = np.zeros(len(frames), bool)
    conf = []
    prev = None
    for i, ds in enumerate(dets):
        d = _pick_box(ds, kind, prev)
        if d is not None:
            boxes[i], valid[i], prev = d.box, True, d.box
            conf.append(d.score)
    coverage = float(valid.mean())
    if not valid.any():
        return VisionResult(size, kind if forced_kind else None, label, 0.0, 0.0, times=times, wrists=wrists)
    boxes = _fill_and_smooth(boxes, valid)
    result = VisionResult(size, kind, label, float(np.mean(conf)), coverage, times, boxes, wrists=wrists)

    if kind == "piano":
        if keyboards:
            km, t = max(keyboards, key=lambda kt: kt[0].confidence * (kt[0].w_max - kt[0].w_min) * np.sqrt(kt[0].key_size()))
            result.keyboard, result.keyboard_time = km, t
    elif kind == "drums":
        result.drums = kit_layout([[d.box for d in ds if d.kind == "drums"] for ds in dets], size)
    report(1.0, "Instrument found")
    return result


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def _guitar_views(guitars, times, size, label, wrists) -> list[VisionResult]:
    """Groups guitar detections into tracks (one per view of the guitar), and
    smooths each track's box and neck. The main view comes first."""
    n = len(guitars)
    tracks: list[dict] = []
    for i, found in enumerate(guitars):
        used = set()
        for det, nm in found:
            best = max((t for t in tracks if id(t) not in used), key=lambda t: _iou(t["last"], det.box), default=None)
            if best is None or _iou(best["last"], det.box) < 0.3:
                best = {"items": {}, "last": det.box, "weight": 0.0}
                tracks.append(best)
            used.add(id(best))
            best["items"][i] = (det, nm)
            best["last"] = det.box
            best["weight"] += det.score * np.sqrt(det.area)
    views = []
    for tr in sorted(tracks, key=lambda t: -t["weight"]):
        if len(tr["items"]) < max(2, 0.4 * n):
            continue
        boxes, valid = np.zeros((n, 4)), np.zeros(n, bool)
        params, ok = np.zeros((n, 9)), np.zeros(n, bool)
        fitted = 0
        for i, (det, nm) in tr["items"].items():
            boxes[i], valid[i] = det.box, True
            if nm is not None:
                params[i], ok[i] = nm.as_array(), True
                fitted += bool(nm.frets_fitted)
        vr = VisionResult(size, "guitar", label, float(np.mean([d.score for d, _ in tr["items"].values()])), float(valid.mean()),
                          times, _fill_and_smooth(boxes, valid), wrists=wrists)
        if ok.any():
            # Reject samples whose neck points the other way or has an odd length.
            vec = params[ok, 2:4] - params[ok, 0:2]
            ref = np.median(vec, 0)
            length = np.linalg.norm(vec, axis=1)
            good = (vec @ ref > 0) & (np.abs(length / np.median(length) - 1) < 0.35)
            ok[np.where(ok)[0][~good]] = False
        if ok.sum() >= max(2, 0.3 * len(tr["items"])):
            # Keep the low-E side consistent.
            side = np.median(params[ok, 6:8], 0)
            params[ok & (params[:, 6:8] @ side < 0), 6:8] *= -1
            vr.neck = _fill_and_smooth(params, ok, window=7)
            vr.neck_fitted = fitted / max(1, len(tr["items"]))
        views.append(vr)
    return views


def _wrists(people) -> np.ndarray:
    pts = [w for p in people for w in p.wrists()]
    return np.array(pts) if pts else np.zeros((0, 2))


def align_keyboard_octave(vr: VisionResult, notes) -> None:
    """Picks the octave numbering for the mapped keys. Notes being played should
    sit under the player's hands, and as many notes as possible should fall on
    keys that are in view. Without any evidence the C nearest the middle stays
    middle C."""
    kb = vr.keyboard
    if kb is None or not notes:
        return
    W, H = vr.size
    best = None
    for shift in range(-3, 4):
        c_midi = kb.c_midi + 12 * shift
        trial = KeyboardModel(kb.origin, kb.row_dir, kb.axis, kb.h, kb.blen, kb.w_min, kb.w_max, c_midi, kb.confidence,
                              kb.seen_min, kb.seen_max)
        on_screen = 0.0
        for nt in notes:
            p = trial.x_of(nt.midi)
            inside = 0 <= p[0] < W and 0 <= p[1] < H
            on_screen += 1.0 if trial.seen(nt.midi) and inside else 0.3 if trial.has_key(nt.midi) and inside else 0.0
        hand, n = 0.0, 0
        for t, wr in zip(vr.times, vr.wrists):
            if not len(wr):
                continue
            shift_px = vr.shift_at(t, vr.keyboard_time)
            for nt in notes:
                if nt.t0 <= t <= nt.t1:
                    hand += min(np.linalg.norm(wr - (trial.x_of(nt.midi) + shift_px), axis=1)) / W
                    n += 1
        cost = -on_screen / len(notes) + (hand / n if n else 0) + 0.01 * abs(shift)
        if best is None or cost < best[0]:
            best = (cost, c_midi)
    kb.c_midi = best[1]
