"""Model-free gaze helper, split out of facetracking.py so the rules layer
(pipeline.py) can use it without importing mediapipe."""

from __future__ import annotations


def gaze_bucket(gaze_x: float, gaze_y: float, threshold: float = 0.35) -> str:
    """
    Buckets a continuous (gaze_x, gaze_y) offset into one of the discrete
    directions API_CONTRACTS.md §3's AI_ANALYSIS.gaze_direction expects
    ("CENTER"/"LEFT"/"RIGHT"/"UP"/"DOWN"). Whichever axis is furthest past
    threshold wins; ties/near-center collapse to CENTER.
    """
    if abs(gaze_x) < threshold and abs(gaze_y) < threshold:
        return "CENTER"
    if abs(gaze_x) >= abs(gaze_y):
        return "RIGHT" if gaze_x > 0 else "LEFT"
    return "DOWN" if gaze_y > 0 else "UP"
