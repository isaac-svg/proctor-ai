"""
Plain-data types passed between proctor-ai's two layers.

The service is split deliberately in two:

  perception (facetracking.py, obd.py, face_identity.py, audio_features.py, ...)
      Heavy, model-backed. Turns a JPEG or a PCM chunk into *observations*:
      "2 faces, the primary one is yaw 31 degrees, mean luma 14".

  rules (rules/*.py, pipeline.py)
      Pure logic over observations and timestamps. Decides what those
      observations *mean* over time: "the face has been gone for 6 seconds,
      that's an episode, raise FACE_ABSENT once".

Nothing in the rules layer imports mediapipe, torch or ultralytics, and every
rule takes its notion of "now" from the observation's own `ts`, never from the
wall clock. That is what makes the rules unit-testable with synthetic
timelines (tests/test_rules_*.py) and lets a rule be tuned without loading a
single model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

SEVERITIES = ("LOW", "MEDIUM", "HIGH", "CRITICAL")


@dataclass
class BoundingBox:
    """Axis-aligned box in *normalized* image coordinates (0..1), so rules
    don't depend on the capture resolution."""

    x: float
    y: float
    width: float
    height: float

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.width / 2.0, self.y + self.height / 2.0

    def clipped_area_ratio(self) -> float:
        """Fraction of this box that lies inside the frame. 1.0 = fully
        visible; a face half out of the frame is ~0.5."""
        total = self.area
        if total <= 0:
            return 0.0
        ix = max(0.0, min(self.x + self.width, 1.0) - max(self.x, 0.0))
        iy = max(0.0, min(self.y + self.height, 1.0) - max(self.y, 0.0))
        return (ix * iy) / total

    def to_dict(self) -> Dict[str, float]:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}


@dataclass
class FaceObservation:
    """One detected face. `pose` fields are only meaningful for the primary
    face (the largest one) -- head pose is not estimated for the others."""

    box: BoundingBox
    confidence: float = 1.0
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    gaze_x: float = 0.0
    gaze_y: float = 0.0


@dataclass
class ObjectObservation:
    label: str  # COCO class name, e.g. "cell phone", "person", "book"
    confidence: float
    box: BoundingBox


@dataclass
class FrameQuality:
    """Cheap per-frame image statistics, all computed without any model."""

    mean_luma: float = 128.0  # 0..255
    luma_std: float = 40.0  # contrast; ~0 for a covered/uniform frame
    sharpness: float = 100.0  # variance of the Laplacian; low = blurred/covered
    # Mean absolute difference from the previous frame (0..255, on a small
    # grayscale thumbnail). None on the first frame. A real sensor is never
    # exactly static -- a value near 0 for many frames means a frozen or
    # looped feed.
    motion: Optional[float] = None


@dataclass
class FrameObservation:
    ts: float
    faces: List[FaceObservation] = field(default_factory=list)  # primary (largest) first
    objects: List[ObjectObservation] = field(default_factory=list)
    quality: FrameQuality = field(default_factory=FrameQuality)
    # Cosine similarity of the primary face to the enrolled reference, or None
    # when no reference was enrolled / the frame wasn't good enough to score.
    identity_similarity: Optional[float] = None
    # Set by the perception layer when it *tried* to score identity but the
    # face was unusable (turned away, too small, blurry) -- distinguishes
    # "no data" from "different person".
    identity_skipped_reason: Optional[str] = None

    @property
    def primary(self) -> Optional[FaceObservation]:
        return self.faces[0] if self.faces else None

    @property
    def persons(self) -> List[ObjectObservation]:
        return [o for o in self.objects if o.label == "person"]


@dataclass
class AudioObservation:
    """Features for one analysis window of microphone audio (~1 second)."""

    ts: float
    duration_s: float
    rms_dbfs: float  # -inf-ish for digital silence, 0 = full scale
    peak: float  # 0..1
    clipping_ratio: float = 0.0
    zero_crossing_rate: float = 0.0
    spectral_flatness: float = 0.0  # 0 = tonal, 1 = noise-like
    speech_ratio: float = 0.0  # fraction of the window the VAD called speech
    # L2-normalised speaker embedding for the speech in this window, when a
    # speaker embedder is available and there was enough speech to embed.
    speaker_embedding: Optional[List[float]] = None

    @property
    def is_speech(self) -> bool:
        return self.speech_ratio >= 0.3


@dataclass
class AlertEvent:
    """What the rules layer emits. alert_mapping.alert_to_message() turns it
    into the AI_ALERT wire shape."""

    alert_type: str
    severity: str
    ts: float
    description: str
    details: Dict[str, Any] = field(default_factory=dict)
    confidence: Optional[float] = None
    # Ask the service to attach a short evidence clip (keyframes / audio).
    evidence: bool = False
    # "audio" attaches audio, "video" attaches keyframes, "both" attaches both.
    evidence_kind: str = "video"

    def __post_init__(self) -> None:
        if self.severity not in SEVERITIES:
            raise ValueError(f"unknown severity {self.severity!r}")
