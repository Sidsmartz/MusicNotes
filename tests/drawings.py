"""Synthetic instrument images with exactly known geometry."""
from __future__ import annotations

import cv2
import numpy as np

from musicnotes.vision.piano import BLACK_W


def _background(h, w, seed=0):
    rng = np.random.default_rng(seed)
    noise = cv2.GaussianBlur(rng.integers(60, 120, (h, w), dtype=np.uint8), (0, 0), 6)
    return cv2.cvtColor(noise, cv2.COLOR_GRAY2BGR)


def guitar(angle_deg=12.0, size=(1280, 720), seed=0):
    """A guitar seen from the front. Returns the image, its box, and the true
    nut and 12th-fret centres."""
    W, H = size
    img = _background(H, W, seed)
    L = 900.0  # overall length, headstock tip to tail
    nut_s, scale = 0.17 * L, 0.63 * L
    d = np.array([np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg))])
    n = np.array([-d[1], d[0]])
    start = np.array([W / 2, H / 2]) - d * L / 2
    P = lambda s, t: (start + d * s + n * t)
    neck_w = 42.0
    body = P(0.78 * L, 0)
    cv2.ellipse(img, tuple(np.int32(body)), (int(0.22 * L), int(0.17 * L)), angle_deg, 0, 360, (60, 120, 190), -1, cv2.LINE_AA)
    cv2.circle(img, tuple(np.int32(P(0.62 * L, 0))), 50, (20, 20, 25), -1, cv2.LINE_AA)
    head = np.array([P(0, -30), P(nut_s, -24), P(nut_s, 24), P(0, 30)], np.int32)
    cv2.fillPoly(img, [head], (40, 50, 60), cv2.LINE_AA)
    board = np.array([P(nut_s, -neck_w / 2), P(0.7 * L, -neck_w / 2 - 4), P(0.7 * L, neck_w / 2 + 4), P(nut_s, neck_w / 2)], np.int32)
    cv2.fillPoly(img, [board], (30, 40, 55), cv2.LINE_AA)
    cv2.line(img, tuple(np.int32(P(nut_s, -neck_w / 2))), tuple(np.int32(P(nut_s, neck_w / 2))), (235, 235, 225), 5, cv2.LINE_AA)
    for k in range(1, 20):
        s = nut_s + scale * (1 - 2 ** (-k / 12))
        if s > 0.7 * L:
            break
        cv2.line(img, tuple(np.int32(P(s, -neck_w / 2))), tuple(np.int32(P(s, neck_w / 2))), (200, 205, 210), 2, cv2.LINE_AA)
    for st in range(6):
        t = -neck_w / 2 + neck_w * (0.1 + st * 0.16)
        cv2.line(img, tuple(np.int32(P(0.05 * L, t))), tuple(np.int32(P(0.86 * L, t))), (190, 200, 210), 1, cv2.LINE_AA)
    pts = np.array([P(0, -40), P(0, 40), P(L, -0.17 * L), P(L, 0.17 * L), P(0.56 * L, -0.17 * L), P(0.56 * L, 0.17 * L)])
    box = np.array([pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()])
    return img, box, P(nut_s, 0), P(nut_s + scale / 2, 0), L


def keyboard(first_c=0, octaves=3, size=(1280, 720), tilt_deg=3.0, seed=0):
    """A keyboard seen from above-front. Returns image, box, and a function giving
    the true image x,y of a black key's centre for (octave, pitch class)."""
    W, H = size
    img = _background(H, W, seed)
    ww, depth, x0, y0 = 1100 / (octaves * 7), 230.0, 90.0, 260.0
    M = cv2.getRotationMatrix2D((W / 2, H / 2), tilt_deg, 1.0)
    cv2.rectangle(img, (int(x0) - 20, int(y0) - 40), (int(x0 + octaves * 7 * ww) + 20, int(y0)), (25, 25, 30), -1)
    for k in range(octaves * 7):
        cv2.rectangle(img, (int(x0 + k * ww) + 1, int(y0)), (int(x0 + (k + 1) * ww) - 1, int(y0 + depth)), (245, 245, 240), -1)
    for o in range(octaves):
        for pc, w in BLACK_W.items():
            cx = x0 + (o * 7 + w) * ww
            cv2.rectangle(img, (int(cx - 0.29 * ww), int(y0)), (int(cx + 0.29 * ww), int(y0 + 0.63 * depth)), (20, 20, 22), -1)
    img = cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REFLECT)

    def black_centre(octave, pc):
        p = np.array([x0 + (octave * 7 + BLACK_W[pc]) * ww, y0 + 0.315 * depth, 1.0])
        return M @ p

    box = np.array([x0 - 30, y0 - 60, x0 + octaves * 7 * ww + 30, y0 + depth + 30])
    return img, box, black_centre, ww
