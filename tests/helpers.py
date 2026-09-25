"""Builders for synthetic observation timelines (1 sample/second by default,
matching shepherd-ai's default snapshot cadence)."""

from __future__ import annotations

from typing import Callable, Iterable, List, Optional

from observations import (
    AlertEvent,
    AudioObservation,
    BoundingBox,
    FaceObservation,
    FrameObservation,
    FrameQuality,
    ObjectObservation,
)


def face(area: float = 0.12, cx: float = 0.5, cy: float = 0.5, yaw: float = 0.0, pitch: float = 0.0,
         gaze_x: float = 0.0, gaze_y: float = 0.0) -> FaceObservation:
    side = area ** 0.5
    return FaceObservation(
        box=BoundingBox(cx - side / 2, cy - side / 2, side, side),
        yaw=yaw, pitch=pitch, gaze_x=gaze_x, gaze_y=gaze_y,
    )


def frame(ts: float, faces: Optional[List[FaceObservation]] = None,
          objects: Optional[List[ObjectObservation]] = None, quality: Optional[FrameQuality] = None,
          identity: Optional[float] = None) -> FrameObservation:
    return FrameObservation(
        ts=ts,
        faces=[face()] if faces is None else faces,
        objects=objects or [],
        quality=quality or FrameQuality(motion=5.0),
        identity_similarity=identity,
    )


def obj(label: str, conf: float = 0.9, area: float = 0.05) -> ObjectObservation:
    side = area ** 0.5
    return ObjectObservation(label, conf, BoundingBox(0.1, 0.1, side, side))


def audio(ts: float, speech: float = 0.0, dbfs: float = -50.0, flat: float = 0.1, emb=None) -> AudioObservation:
    return AudioObservation(ts=ts, duration_s=1.0, rms_dbfs=dbfs, peak=0.2, speech_ratio=speech,
                            spectral_flatness=flat, speaker_embedding=emb)


def run_frames(rule, start: float, seconds: int, make: Callable[[float], object]) -> List[AlertEvent]:
    """Feed `make(ts)` (a FrameObservation) to rule.on_frame once per second."""
    out: List[AlertEvent] = []
    for i in range(seconds):
        out += rule.on_frame(make(start + i))
    return out


def run_audio(rule, start: float, seconds: int, make: Callable[[float], object]) -> List[AlertEvent]:
    out: List[AlertEvent] = []
    for i in range(seconds):
        out += rule.on_audio(make(start + i))
    return out


def types(events: Iterable[AlertEvent]) -> List[str]:
    return [e.alert_type for e in events]
