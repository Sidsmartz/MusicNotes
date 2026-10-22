"""YOLOv8 inference through onnxruntime (no PyTorch needed at runtime).

Two models are used:
  * an Open Images V7 detector, which knows 601 classes including guitars,
    pianos, keyboards, drums, violins, saxophones and other instruments;
  * a COCO pose model, used for the players' wrists (where the hands are).
"""
from __future__ import annotations

import ast
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

MODELS_DIR = Path(os.environ.get("MUSICNOTES_MODELS", Path(__file__).resolve().parent.parent.parent / "models"))

# Open Images class name -> how MusicNotes treats it.
INSTRUMENT_KINDS = {
    "Guitar": "guitar", "Banjo": "guitar",
    "Piano": "piano", "Musical keyboard": "piano", "Harpsichord": "piano", "Organ (Musical Instrument)": "piano",
    "Drum": "drums",
    "Violin": "melody", "Cello": "melody", "Saxophone": "melody", "Trumpet": "melody", "Trombone": "melody",
    "Flute": "melody", "Harp": "melody", "Accordion": "melody", "French horn": "melody", "Oboe": "melody",
    "Harmonica": "melody",
}

# COCO keypoint indices.
LEFT_WRIST, RIGHT_WRIST, LEFT_ELBOW, RIGHT_ELBOW = 9, 10, 7, 8


@dataclass
class Detection:
    box: np.ndarray  # x0, y0, x1, y1 in image pixels
    score: float
    label: str

    @property
    def kind(self) -> str | None:
        return INSTRUMENT_KINDS.get(self.label)

    @property
    def area(self) -> float:
        return float(max(0, self.box[2] - self.box[0]) * max(0, self.box[3] - self.box[1]))


@dataclass
class Person:
    box: np.ndarray
    score: float
    keypoints: np.ndarray  # (17, 3): x, y, confidence

    def wrists(self, min_conf: float = 0.3) -> list[np.ndarray]:
        return [self.keypoints[i, :2] for i in (LEFT_WRIST, RIGHT_WRIST) if self.keypoints[i, 2] >= min_conf]


def _session(path: Path) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    source: str | bytes = str(path)
    if path.name.endswith(".fp16.onnx"):
        # Weights are stored as float16 to halve the file size; inference runs in float32.
        import onnx
        from onnx import numpy_helper

        model = onnx.load(str(path))
        for init in model.graph.initializer:
            if init.data_type == onnx.TensorProto.FLOAT16:
                init.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(init).astype(np.float32), init.name))
        source = model.SerializeToString()
    return ort.InferenceSession(source, sess_options=opts, providers=["CPUExecutionProvider"])


def _letterbox(img: np.ndarray, size: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[top:top + nh, left:left + nw] = resized
    blob = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return np.ascontiguousarray(blob), r, (left, top)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou: float) -> list[int]:
    if not len(boxes):
        return []
    xywh = np.column_stack([boxes[:, 0], boxes[:, 1], boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]])
    keep = cv2.dnn.NMSBoxes(xywh.tolist(), scores.tolist(), 0.0, iou)
    return [int(i) for i in np.array(keep).reshape(-1)]


class InstrumentDetector:
    def __init__(self, model: str | Path = "yolov8m-oiv7.fp16.onnx", size: int = 640):
        path = Path(model) if Path(model).is_absolute() else MODELS_DIR / model
        self.session = _session(path)
        self.size = size
        names = ast.literal_eval(self.session.get_modelmeta().custom_metadata_map["names"])
        self.names = [names[i] for i in range(len(names))]
        self.instrument_ids = np.array([i for i, n in enumerate(self.names) if n in INSTRUMENT_KINDS])

    def __call__(self, img: np.ndarray, conf: float = 0.12, iou: float = 0.5) -> list[Detection]:
        blob, r, (dx, dy) = _letterbox(img, self.size)
        out = self.session.run(None, {self.session.get_inputs()[0].name: blob})[0][0]  # (4 + nc, N)
        scores_all = out[4:][self.instrument_ids]  # only instrument classes
        cls_local = scores_all.argmax(0)
        scores = scores_all[cls_local, np.arange(scores_all.shape[1])]
        mask = scores >= conf
        if not mask.any():
            return []
        cx, cy, w, h = out[:4, mask]
        boxes = np.column_stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
        boxes = (boxes - [dx, dy, dx, dy]) / r
        H, W = img.shape[:2]
        boxes = np.clip(boxes, 0, [W, H, W, H])
        scores, labels = scores[mask], self.instrument_ids[cls_local[mask]]
        # Class-agnostic NMS: one object should not be both a "Piano" and a "Musical keyboard".
        keep = _nms(boxes, scores, iou)
        return [Detection(boxes[i], float(scores[i]), self.names[labels[i]]) for i in keep]


class PoseDetector:
    def __init__(self, model: str | Path = "yolov8n-pose.onnx", size: int = 640):
        path = Path(model) if Path(model).is_absolute() else MODELS_DIR / model
        self.session = _session(path)
        self.size = size

    def __call__(self, img: np.ndarray, conf: float = 0.3, iou: float = 0.5) -> list[Person]:
        blob, r, (dx, dy) = _letterbox(img, self.size)
        out = self.session.run(None, {self.session.get_inputs()[0].name: blob})[0][0]  # (56, N)
        scores = out[4]
        mask = scores >= conf
        if not mask.any():
            return []
        cx, cy, w, h = out[:4, mask]
        boxes = (np.column_stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]) - [dx, dy, dx, dy]) / r
        kpts = out[5:, mask].T.reshape(-1, 17, 3).copy()
        kpts[:, :, 0] = (kpts[:, :, 0] - dx) / r
        kpts[:, :, 1] = (kpts[:, :, 1] - dy) / r
        keep = _nms(boxes, scores[mask], iou)
        return [Person(boxes[i], float(scores[mask][i]), kpts[i]) for i in keep]
