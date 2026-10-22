"""Re-create the ONNX files in models/ from the official Ultralytics weights.

Needs the export tools, which the app itself does not:
    pip install ultralytics onnx onnxslim
    python scripts/export_models.py
"""
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper
from ultralytics import YOLO

OUT = Path(__file__).resolve().parent.parent / "models"

for name in ("yolov8m-oiv7", "yolov8s-oiv7", "yolov8n-pose"):
    path = Path(YOLO(f"{name}.pt").export(format="onnx", imgsz=640, simplify=True, opset=17, dynamic=False))
    if name == "yolov8m-oiv7":
        # Halve the file size: float16 weights, converted back to float32 at load time.
        model = onnx.load(str(path))
        for init in model.graph.initializer:
            if init.data_type == onnx.TensorProto.FLOAT:
                init.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(init).astype(np.float16), init.name))
        onnx.save(model, str(OUT / f"{name}.fp16.onnx"))
    else:
        path.replace(OUT / path.name)
    print("wrote", name)
