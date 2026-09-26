"""
Object detection (YOLO) for the proctoring pipeline.

One inference per frame returns everything the rules care about -- people
(catches someone whose face is turned away), phones, laptops, books --
instead of running the model once per question. The model is loaded lazily
and shared across sessions; `Ultralytics.predict()` isn't guaranteed
thread-safe on a shared model, so calls are serialised here.

What makes this more robust than one plain pass:

  * **Higher input resolution** (`PROCTOR_YOLO_IMGSZ`, default 960). A phone in
    a hand is a few percent of a 640x480 webcam frame; the stock 640 input
    shrinks it to a handful of pixels.
  * **A second pass over the lower part of the frame** (`PROCTOR_OBJECT_SECOND_PASS`,
    default on), where hands and desks are, upscaled so small objects are larger to
    the model. Frames arrive about once a second, so the extra inference is affordable.
    The two passes' boxes are merged (per label, highest confidence wins).
  * **Plausibility filters.** A "phone" covering half the frame is a monitor, and one a few
    pixels across is noise; neither is passed on.
  * **A bigger or different model** without code changes (`PROCTOR_YOLO_MODEL`), and an
    optional **extra model** (`PROCTOR_EXTRA_MODEL` + `PROCTOR_EXTRA_LABELS`) for things the
    stock COCO classes do not include -- earbuds, headphones, smartwatches, tablets, notes.
    None is shipped; the rules simply report those labels when a model provides them.

`objectDetection.predict()` keeps its original bool contract (phone in frame)
for main.py's local demo.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from observations import BoundingBox, ObjectObservation

log = logging.getLogger("proctor-ai.objects")

# Resolved next to this file, not against the process's working directory, so
# `uvicorn service:app` works from anywhere.
_DEFAULT_MODEL = Path(__file__).resolve().parent / "yolo26n.pt"

# COCO class ids: person, laptop, cell phone, book. Everything else the stock model
# knows about is irrelevant to proctoring and dropped at the source.
_CLASSES = [0, 63, 67, 73]
# Deliberately low: the *rules* apply the real per-label confidence floors
# (pipeline_config.object_rules) and confirm across frames.
_DETECT_CONF = 0.20

# Where hands and phones are: the lower part of a webcam frame.
_SECOND_PASS_TOP = 0.35

# Plausible size of each label as a fraction of the frame. Anything outside is a mis-detection.
_AREA_LIMITS: Dict[str, Tuple[float, float]] = {
    "cell phone": (0.0004, 0.20),
    "book": (0.002, 0.60),
    "laptop": (0.01, 0.90),
    "person": (0.005, 1.0),
}
_DEFAULT_AREA_LIMITS = (0.0002, 0.90)

_MERGE_IOU = 0.5

_models: Dict[str, Any] = {}
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Configuration (read each time so a test or an operator can change it without a reload)
# ---------------------------------------------------------------------------
def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    return default if raw is None else raw.strip().lower() not in ("0", "false", "no", "off", "")


def settings() -> Dict[str, Any]:
    imgsz = int(os.environ.get("PROCTOR_YOLO_IMGSZ", "960") or 960)
    extra_labels: Dict[str, str] = {}
    raw = os.environ.get("PROCTOR_EXTRA_LABELS", "")
    if raw:
        try:
            parsed = json.loads(raw)
            extra_labels = {str(k).lower(): str(v) for k, v in parsed.items()}
        except (ValueError, AttributeError):
            log.warning("PROCTOR_EXTRA_LABELS is not a JSON object of {model class: label}; ignored")
    return {
        "model": os.environ.get("PROCTOR_YOLO_MODEL") or str(_DEFAULT_MODEL),
        "imgsz": max(320, min(1920, imgsz)),
        "second_pass": _env_bool("PROCTOR_OBJECT_SECOND_PASS", True),
        "extra_model": os.environ.get("PROCTOR_EXTRA_MODEL") or None,
        "extra_labels": extra_labels,
    }


def describe() -> Dict[str, Any]:
    """What is configured, for /health: an operator can see which model is really running."""
    s = settings()
    return {
        "model": Path(s["model"]).name,
        "imgsz": s["imgsz"],
        "second_pass": s["second_pass"],
        "extra_model": Path(s["extra_model"]).name if s["extra_model"] else None,
        "extra_labels": sorted(set(s["extra_labels"].values())),
    }


def _get_model(path: Optional[str] = None):
    path = path or settings()["model"]
    if path not in _models:
        from ultralytics import YOLO  # heavy import, deferred until first use

        _models[path] = YOLO(path)
    return _models[path]


# ---------------------------------------------------------------------------
# Pure helpers (no model): merging, filtering, coordinate mapping
# ---------------------------------------------------------------------------
def _iou(a: BoundingBox, b: BoundingBox) -> float:
    ix = max(0.0, min(a.x + a.width, b.x + b.width) - max(a.x, b.x))
    iy = max(0.0, min(a.y + a.height, b.y + b.height) - max(a.y, b.y))
    inter = ix * iy
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def merge_detections(detections: Iterable[ObjectObservation], iou_threshold: float = _MERGE_IOU) -> List[ObjectObservation]:
    """Non-maximum suppression across passes: per label, overlapping boxes collapse to the most confident one."""
    kept: List[ObjectObservation] = []
    for det in sorted(detections, key=lambda d: -d.confidence):
        if any(k.label == det.label and _iou(k.box, det.box) >= iou_threshold for k in kept):
            continue
        kept.append(det)
    return kept


def plausible(det: ObjectObservation) -> bool:
    low, high = _AREA_LIMITS.get(det.label, _DEFAULT_AREA_LIMITS)
    return low <= det.box.area <= high and det.box.width > 0 and det.box.height > 0


def remap_crop_box(x1: float, y1: float, x2: float, y2: float, top: float) -> BoundingBox:
    """A normalised box measured inside the crop that starts at fraction `top` of the frame height, back to
    normalised full-frame coordinates."""
    scale = 1.0 - top
    return BoundingBox(x1, top + y1 * scale, x2 - x1, (y2 - y1) * scale)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def _predict(model, image: np.ndarray, imgsz: int, classes: Optional[List[int]]) -> List[Tuple[str, float, Tuple[float, float, float, float]]]:
    kwargs: Dict[str, Any] = {"source": image, "conf": _DETECT_CONF, "imgsz": imgsz, "verbose": False}
    if classes is not None:
        kwargs["classes"] = classes
    result = model.predict(**kwargs)[0]
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    names = model.names
    xyxyn = boxes.xyxyn.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    return [(str(names[int(c)]), float(cf), (float(a), float(b), float(d), float(e))) for (a, b, d, e), cf, c in zip(xyxyn, confs, cls)]


def detect_objects(frame_bgr: np.ndarray) -> List[ObjectObservation]:
    """Detect relevant objects in one BGR frame. Boxes are normalized 0..1."""
    s = settings()
    found: List[ObjectObservation] = []
    with _lock:
        model = _get_model(s["model"])
        for label, conf, (x1, y1, x2, y2) in _predict(model, frame_bgr, s["imgsz"], _CLASSES):
            found.append(ObjectObservation(label, conf, BoundingBox(x1, y1, x2 - x1, y2 - y1), "main"))

        if s["second_pass"]:
            h = frame_bgr.shape[0]
            crop = frame_bgr[int(h * _SECOND_PASS_TOP):, :]
            if crop.size:
                for label, conf, (x1, y1, x2, y2) in _predict(model, crop, s["imgsz"], _CLASSES):
                    found.append(ObjectObservation(label, conf, remap_crop_box(x1, y1, x2, y2, _SECOND_PASS_TOP), "hands"))

        if s["extra_model"] and s["extra_labels"]:
            try:
                extra = _get_model(s["extra_model"])
                for name, conf, (x1, y1, x2, y2) in _predict(extra, frame_bgr, s["imgsz"], None):
                    label = s["extra_labels"].get(name.lower())
                    if label:
                        found.append(ObjectObservation(label, conf, BoundingBox(x1, y1, x2 - x1, y2 - y1), "extra"))
            except Exception as exc:  # the extra model is optional: its failure must not silence phones and people
                log.warning("extra object model failed: %s", exc)
    return merge_detections(d for d in found if plausible(d))


class objectDetection:
    """Original interface, kept for main.py's local demo."""

    def __init__(self) -> None:
        pass

    def predict(self, frame) -> bool:
        return any(o.label == "cell phone" for o in detect_objects(frame))
