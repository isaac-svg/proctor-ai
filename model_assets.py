"""
Downloadable model files, fetched on first use and verified.

face_landmarker.task and yolo26n.pt are checked into this repo (~1 MB and
small). The identity and speaker models are 25-40 MB each, so they're
downloaded to ./models/ (git-ignored) the first time they're needed instead,
and every download is checked against a pinned SHA-256 -- a model file is
code the service will execute, so it must not be silently replaced by
whatever a URL happens to return one day.

Nothing here runs at import time; a deployment can also pre-populate
./models/ (air-gapped installs) and no network access is attempted when the
file is already present and matches its hash.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

MODELS_DIR = Path(__file__).resolve().parent / "models"


@dataclass(frozen=True)
class ModelAsset:
    filename: str
    url: str
    sha256: Optional[str]  # None = not pinned yet (logged, not enforced)


# OpenCV Zoo, Apache-2.0. (InsightFace's weights are non-commercial and
# deliberately not used.)
YUNET = ModelAsset(
    "face_detection_yunet_2023mar.onnx",
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
)
SFACE = ModelAsset(
    "face_recognition_sface_2021dec.onnx",
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
    "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
)
# WeSpeaker ResNet34 speaker-embedding model (Apache-2.0), ONNX export.
WESPEAKER = ModelAsset(
    "voxceleb_resnet34_LM.onnx",
    "https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34-LM/resolve/main/voxceleb_resnet34_LM.onnx",
    "7bb2f06e9df17cdf1ef14ee8a15ab08ed28e8d0ef5054ee135741560df2ec068",
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class ModelIntegrityError(RuntimeError):
    pass


def ensure_asset(asset: ModelAsset, models_dir: Path = MODELS_DIR) -> Path:
    """Return the local path of `asset`, downloading it if absent. Raises
    ModelIntegrityError if the file doesn't match its pinned hash (a corrupt
    or tampered download is deleted rather than left to be loaded later)."""
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / asset.filename

    if path.exists():
        if asset.sha256 is None or _sha256(path) == asset.sha256:
            return path
        path.unlink()  # stale/corrupt: fall through and re-download

    fd, tmp_name = tempfile.mkstemp(dir=models_dir, suffix=".part")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        with urllib.request.urlopen(asset.url, timeout=120) as resp, tmp.open("wb") as out:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
        if asset.sha256 is not None and _sha256(tmp) != asset.sha256:
            raise ModelIntegrityError(
                f"{asset.filename}: downloaded file does not match its pinned SHA-256; refusing to use it"
            )
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return path
