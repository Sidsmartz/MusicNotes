# Models

| File | What it is | Used for |
| --- | --- | --- |
| `yolov8m-oiv7.fp16.onnx` | Ultralytics YOLOv8m trained on Open Images V7 (601 classes) | finding the instrument (default) |
| `yolov8s-oiv7.onnx` | Ultralytics YOLOv8s, Open Images V7 | a faster, less accurate detector |
| `yolov8n-pose.onnx` | Ultralytics YOLOv8n-pose (COCO keypoints) | the player's wrists |

These are detection and pose-estimation models. They locate objects and body joints; they do not generate images.

The `.fp16.onnx` file stores its weights as 16-bit floats to stay under GitHub's 100 MB file limit. They are converted back to 32-bit when loaded, and detection results are identical (see the benchmark in the main README).

All three were exported from the official Ultralytics release weights with `scripts/export_models.py`.

**License:** Ultralytics models are released under AGPL-3.0. If you distribute this project, or run it as a network service for other people, the AGPL terms apply. Ultralytics also offers an enterprise license.
