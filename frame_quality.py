"""
Model-free image statistics: brightness, contrast, sharpness and
frame-to-frame change. Feeds the camera-integrity rules (covered / frozen).
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from observations import FrameQuality

# 160x120: coarse enough to ignore per-pixel noise, fine enough that a real
# sensor's residual noise still registers as change frame to frame. (64x48
# averaged the noise away and made a live static scene look frozen.)
_THUMB = (160, 120)


def compute_quality(
    frame_bgr: np.ndarray, previous_thumb: Optional[np.ndarray] = None
) -> Tuple[FrameQuality, np.ndarray]:
    """Returns the frame's quality stats and a small grayscale thumbnail to
    pass back in as `previous_thumb` for the next frame's motion estimate."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    thumb = cv2.resize(gray, _THUMB, interpolation=cv2.INTER_AREA)

    motion: Optional[float] = None
    if previous_thumb is not None and previous_thumb.shape == thumb.shape:
        motion = float(np.mean(np.abs(thumb.astype(np.int16) - previous_thumb.astype(np.int16))))

    return (
        FrameQuality(
            mean_luma=float(gray.mean()),
            luma_std=float(gray.std()),
            sharpness=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
            motion=motion,
        ),
        thumb,
    )
