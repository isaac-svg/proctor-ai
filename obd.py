"""
Object detection (YOLO) for the proctoring pipeline.

One inference per frame returns everything the rules care about -- people
(catches someone whose face is turned away), phones, laptops, books --
instead of running the model once per question. The model is loaded lazily
and shared across sessions; `Ultralytics.predict()` isn't guaranteed
thread-safe on a shared model, so calls are serialised here.

`objectDetection.predict()` keeps its original bool contract (phone in frame)
for main.py's local demo.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import List, Optional

import numpy as np

from observations import BoundingBox, ObjectObservation

# Resolved next to this file, not against the process's working directory, so
# `uvicorn service:app` works from anywhere.
_MODEL_PATH = Path(__file__).resolve().parent / "yolo26n.pt"

# COCO class ids: person, laptop, cell phone, book. Everything else the model
# knows about is irrelevant to proctoring and dropped at the source.
_CLASSES = [0, 63, 67, 73]
# Deliberately low: the *rules* apply the real per-label confidence floors
# (pipeline_config.object_rules) and require confirmation across frames.
_DETECT_CONF = 0.25

_model = None
_lock = threading.Lock()


def _get_model():
    global _model
    if _model is None:
        from ultralytics import YOLO  # heavy import, deferred until first use

        _model = YOLO(str(_MODEL_PATH))
    return _model


def detect_objects(frame_bgr: np.ndarray) -> List[ObjectObservation]:
    """Detect relevant objects in one BGR frame. Boxes are normalized 0..1."""
    with _lock:
        model = _get_model()
        result = model.predict(source=frame_bgr, conf=_DETECT_CONF, classes=_CLASSES, verbose=False)[0]
        names = model.names
        found: List[ObjectObservation] = []
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return found
        xyxyn = boxes.xyxyn.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
    for (x1, y1, x2, y2), conf, cls in zip(xyxyn, confs, classes):
        found.append(
            ObjectObservation(
                label=str(names[int(cls)]),
                confidence=float(conf),
                box=BoundingBox(float(x1), float(y1), float(x2 - x1), float(y2 - y1)),
            )
        )
    return found


class objectDetection:
    """Original interface, kept for main.py's local demo."""

    def __init__(self) -> None:
        pass

    def predict(self, frame) -> bool:
        return any(o.label == "cell phone" for o in detect_objects(frame))
