"""Drum kit layout from drum detections collected across a video."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DrumPiece:
    kind: str  # kick | snare | hat
    box: np.ndarray

    @property
    def centre(self) -> np.ndarray:
        return np.array([(self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2])

    @property
    def radii(self) -> tuple[float, float]:
        w, h = self.box[2] - self.box[0], self.box[3] - self.box[1]
        return w / 2, h / 2


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def kit_layout(boxes_per_sample: list[list[np.ndarray]], frame_size: tuple[int, int]) -> list[DrumPiece]:
    """Clusters drum boxes over time into kit pieces and names them.

    The detector only knows "Drum", so we name pieces by layout: the bass drum
    is the biggest piece low in the frame, the snare the closest piece to the
    middle of the kit, the hi-hat the highest piece on the other side.
    """
    clusters: list[list[np.ndarray]] = []
    for boxes in boxes_per_sample:
        for b in boxes:
            for c in clusters:
                if _iou(np.median(c, 0), b) > 0.35:
                    c.append(b)
                    break
            else:
                clusters.append([b])
    n = max(1, len(boxes_per_sample))
    pieces = [np.median(c, 0) for c in clusters if len(c) >= max(2, 0.2 * n)]
    if not pieces:
        return []
    W, H = frame_size
    pieces.sort(key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))
    if len(pieces) == 1:
        # One box around the whole kit: place the parts inside it.
        x0, y0, x1, y1 = pieces[0]
        w, h = x1 - x0, y1 - y0
        sub = lambda cx, cy, rw, rh: np.array([x0 + (cx - rw) * w, y0 + (cy - rh) * h, x0 + (cx + rw) * w, y0 + (cy + rh) * h])
        return [DrumPiece("kick", sub(0.5, 0.7, 0.2, 0.22)), DrumPiece("snare", sub(0.3, 0.5, 0.14, 0.08)), DrumPiece("hat", sub(0.78, 0.3, 0.14, 0.06))]
    low = [b for b in pieces if (b[1] + b[3]) / 2 > np.median([(p[1] + p[3]) / 2 for p in pieces])] or pieces
    kick = max(low, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    rest = [b for b in pieces if b is not kick]
    kit_x = (kick[0] + kick[2]) / 2
    snare = min(rest, key=lambda b: abs((b[0] + b[2]) / 2 - kit_x) + 0.5 * abs((b[1] + b[3]) / 2 - (kick[1])))
    out = [DrumPiece("kick", kick), DrumPiece("snare", snare)]
    others = [b for b in rest if b is not snare]
    if others:
        side = np.sign((snare[0] + snare[2]) / 2 - kit_x) or 1
        opposite = [b for b in others if np.sign((b[0] + b[2]) / 2 - kit_x) != side] or others
        out.append(DrumPiece("hat", min(opposite, key=lambda b: b[1])))
    else:
        cx = 2 * kit_x - (snare[0] + snare[2]) / 2
        w, h = snare[2] - snare[0], snare[3] - snare[1]
        out.append(DrumPiece("hat", np.array([cx - w / 2, snare[1] - h * 1.2, cx + w / 2, snare[1] - h * 0.2])))
    return out
