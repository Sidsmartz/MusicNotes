"""Benchmark the vision side on real photos from the Open Images V7 validation set.

Downloads up to 40 photos each of guitars, pianos/keyboards and drums (real
objects, not drawings, not cut off, filling at least 8% of the frame), then
reports:
  * detection: found with IoU >= 0.5 against the human-drawn box;
  * guitar: neck found, and frets fitted (versus typical proportions);
  * piano: keys mapped from the black-key pattern.
It also writes contact sheets so you can check the geometry by eye.

    python scripts/benchmark.py --out bench/
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from musicnotes import theory  # noqa: E402
from musicnotes.vision.detector import InstrumentDetector, PoseDetector  # noqa: E402
from musicnotes.vision.guitar import estimate_neck, fret_u, string_v  # noqa: E402
from musicnotes.vision.piano import estimate_keyboard  # noqa: E402

BOXES = "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv"
IMAGE = "https://open-images-dataset.s3.amazonaws.com/validation/{}.jpg"
CLASSES = {"/m/0342h": "guitar", "/m/05r5c": "piano", "/m/057cc": "piano", "/m/026t6": "drums"}


def fetch(url: str, path: Path) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(urllib.request.urlopen(url, timeout=60).read())
    return path


def pick(cache: Path, per_kind: int):
    rows = csv.DictReader(io.StringIO(fetch(BOXES, cache / "validation-boxes.csv").read_text()))
    by_image = defaultdict(list)
    for r in rows:
        if r["LabelName"] in CLASSES:
            by_image[r["ImageID"]].append(r)
    picked = {"guitar": [], "piano": [], "drums": []}
    for image_id, rs in sorted(by_image.items()):
        for r in rs:
            kind = CLASSES[r["LabelName"]]
            area = (float(r["XMax"]) - float(r["XMin"])) * (float(r["YMax"]) - float(r["YMin"]))
            if r["IsGroupOf"] == "0" and r["IsDepiction"] == "0" and r["IsTruncated"] == "0" and area > 0.08 \
                    and len(picked[kind]) < per_kind and all(p["id"] != image_id for p in picked[kind]):
                boxes = [[float(x[c]) for c in ("XMin", "YMin", "XMax", "YMax")] for x in rs if CLASSES[x["LabelName"]] == kind]
                picked[kind].append({"id": image_id, "boxes": boxes})
    return picked


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)


def tile(img, size=360):
    s = size / max(img.shape[:2])
    img = cv2.resize(img, None, fx=s, fy=s)
    out = np.zeros((size, size, 3), np.uint8)
    out[:img.shape[0], :img.shape[1]] = img
    return out


def sheet(tiles, path, cols=4):
    if not tiles:
        return
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(str(path), np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bench")
    ap.add_argument("--per-kind", type=int, default=40)
    ap.add_argument("--model", default="yolov8m-oiv7.fp16.onnx")
    a = ap.parse_args()
    out = Path(a.out)
    cache = out / "cache"
    det, pose = InstrumentDetector(a.model), PoseDetector()
    picked = pick(cache, a.per_kind)
    report = {}
    for kind, items in picked.items():
        found = neck = fitted = keys = 0
        tiles = []
        for it in items:
            img = cv2.imread(str(fetch(IMAGE.format(it["id"]), cache / "img" / f"{it['id']}.jpg")))
            H, W = img.shape[:2]
            ds = det(img)
            gts = [[b[0] * W, b[1] * H, b[2] * W, b[3] * H] for b in it["boxes"]]
            found += any(d.kind == kind and max(iou(d.box, g) for g in gts) >= 0.5 for d in ds)
            mine = [d for d in ds if d.kind == kind]
            if not mine:
                continue
            vis = img.copy()
            if kind == "guitar":
                box = max(mine, key=lambda d: d.score).box
                nm = estimate_neck(img, box, [w for p in pose(img) for w in p.wrists()])
                if nm is not None:
                    neck += 1
                    fitted += bool(nm.frets_fitted)
                    for f in range(16):
                        p0, p1 = nm.point(fret_u(f), 0), nm.point(fret_u(f), 1)
                        cv2.line(vis, tuple(np.int32(p0)), tuple(np.int32(p1)), (0, 0, 255) if f in (0, 12) else (0, 255, 255), 2, cv2.LINE_AA)
                    for s in range(6):
                        p0, p1 = nm.point(-0.05, string_v(s)), nm.point(fret_u(15), string_v(s))
                        cv2.line(vis, tuple(np.int32(p0)), tuple(np.int32(p1)), (230, 230, 230), 1, cv2.LINE_AA)
            elif kind == "piano":
                kb_boxes = [d for d in mine if d.label == "Musical keyboard"] or mine
                km = estimate_keyboard(img, max(kb_boxes, key=lambda d: d.score * d.area ** 0.5).box)
                if km is not None:
                    keys += 1
                    lo, hi = km.midi_range()
                    for m in range(lo, hi + 1):
                        if not km.seen(m):
                            continue
                        r, g, b = theory.note_rgb(m)
                        ov = vis.copy()
                        cv2.fillPoly(ov, [km.key_polygon(m).astype(np.int32)], (int(b * 255), int(g * 255), int(r * 255)))
                        vis = cv2.addWeighted(ov, 0.45, vis, 0.55, 0)
            else:
                for d in mine:
                    cv2.rectangle(vis, tuple(d.box[:2].astype(int)), tuple(d.box[2:].astype(int)), (0, 200, 255), 3)
            tiles.append(tile(vis))
        report[kind] = {"photos": len(items), "detected": found}
        if kind == "guitar":
            report[kind].update(neck_found=neck, frets_fitted=fitted)
        if kind == "piano":
            report[kind].update(keys_mapped=keys)
        sheet(tiles, out / f"{kind}.jpg")
    print(json.dumps(report, indent=2))
    (out / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
