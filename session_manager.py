"""
Per-session state for the video-stream analysis service (service.py).

A ProctorSession is the glue between the two layers described in
observations.py: it runs *perception* (MediaPipe pose, YOLO objects, YuNet/
SFace identity, Silero VAD, speaker embeddings, image/audio statistics) to
turn each JPEG / PCM chunk into observations, hands those to a
`ProctorPipeline` (pure rules), and snapshots an evidence clip for any alert
that asks for one.

main.py's local demo is unaffected: it still uses events.py's original
EventDetector directly.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Dict, List, NamedTuple, Optional

import cv2
import numpy as np

from audio_features import analyze_window
from evidence import EvidenceBuffer
from face_identity import EnrollmentResult, FaceIdentityVerifier
from facetracking import HeadPoseEstimator, PoseResult
from frame_quality import compute_quality
from obd import detect_objects
from observations import (
    AlertEvent,
    BoundingBox,
    FaceObservation,
    FrameObservation,
)
from pipeline import ProctorPipeline
from pipeline_config import PipelineConfig
from speaker_embedding import MIN_SAMPLES as SPEAKER_MIN_SAMPLES
from speaker_embedding import SpeakerEmbedder, load_speaker_embedder
from vad import VoiceActivityDectector

log = logging.getLogger("proctor-ai.session")

# One VAD/feature window per second of buffered audio: 16 kHz * 2 bytes.
_AUDIO_WINDOW_BYTES = 16000 * 2
# Speaker embeddings need ~2 s of context to be reliable; embed the last two
# windows whenever the current one is mostly speech.
_SPEAKER_CONTEXT_BYTES = _AUDIO_WINDOW_BYTES * 2

# Faces to look for: enough to notice a second and third person.
_MAX_FACES = 3
# Identity is scored on every Nth frame -- at ~1 frame/second that's every
# couple of seconds, plenty for "is it still the same person".
_IDENTITY_EVERY_N_FRAMES = 2


class Emission(NamedTuple):
    event: AlertEvent
    clip: Optional[Dict[str, Any]]


class SharedModels:
    """Process-wide, lazily loaded model handles. Loading can fail (missing
    optional dependency, no network for a first download); each failure is
    recorded once and the affected capability is reported as off rather than
    taking the whole service down."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._identity: Optional[FaceIdentityVerifier] = None
        self._identity_tried = False
        self._speaker: Optional[SpeakerEmbedder] = None
        self._speaker_tried = False
        self.errors: Dict[str, str] = {}

    @property
    def identity(self) -> Optional[FaceIdentityVerifier]:
        with self._lock:
            if not self._identity_tried:
                self._identity_tried = True
                try:
                    self._identity = FaceIdentityVerifier()
                except Exception as exc:
                    self.errors["identity"] = str(exc)
                    log.warning("identity verification unavailable: %s", exc)
            return self._identity

    @property
    def speaker(self) -> Optional[SpeakerEmbedder]:
        with self._lock:
            if not self._speaker_tried:
                self._speaker_tried = True
                self._speaker = load_speaker_embedder()
                if self._speaker is None:
                    self.errors["speaker_diarization"] = "onnxruntime or model unavailable"
            return self._speaker

    def warm_up(self) -> None:
        """Load everything now, so the first real frame doesn't pay for it."""
        _ = self.identity
        _ = self.speaker
        try:
            import obd

            obd._get_model()
        except Exception as exc:
            self.errors["objects"] = str(exc)
            log.warning("object detection unavailable: %s", exc)

    def capabilities(self) -> Dict[str, bool]:
        return {
            "identity": self._identity is not None,
            "speaker_diarization": self._speaker is not None,
            "objects": "objects" not in self.errors,
        }


class ProctorSession:
    """All per-session analysis state. Thread-safe: service.py runs video and
    audio handling for one session on a thread pool, so both funnel through
    `self._lock`."""

    def __init__(
        self,
        session_id: str,
        shared: SharedModels,
        cfg: Optional[PipelineConfig] = None,
        frame_width: int = 640,
        frame_height: int = 480,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.session_id = session_id
        self.cfg = cfg or PipelineConfig()
        self.shared = shared
        self._clock = clock
        self._lock = threading.Lock()

        self.estimator = HeadPoseEstimator(frame_width, frame_height, num_faces=_MAX_FACES)
        self.vad = VoiceActivityDectector(open_stream=False)
        self.pipeline = ProctorPipeline(self.cfg)
        self.evidence = EvidenceBuffer(self.cfg)

        self._reference: Optional[List[float]] = None
        self._prev_thumb: Optional[np.ndarray] = None
        self._frame_count = 0
        self._last_ms = 0
        self._audio_buffer = bytearray()
        self._speech_tail = b""
        self.created_at = time.monotonic()

    # ------------------------------------------------------------------
    # Identity enrollment
    # ------------------------------------------------------------------
    def enroll(self, jpegs: List[bytes]) -> EnrollmentResult:
        """Validate check-in captures and return the candidate embedding. The
        images are not retained."""
        verifier = self.shared.identity
        if verifier is None:
            return EnrollmentResult(False, "identity_unavailable")
        frames = []
        for raw in jpegs[:5]:
            img = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                frames.append(img)
        if not frames:
            return EnrollmentResult(False, "no_valid_image")
        result = verifier.enroll(frames)
        if result.ok and result.embedding is not None:
            self._reference = result.embedding
        return result

    def set_reference(self, embedding: List[float]) -> bool:
        if not embedding or len(embedding) < 16 or not all(isinstance(x, (int, float)) for x in embedding):
            return False
        self._reference = [float(x) for x in embedding]
        return True

    @property
    def has_reference(self) -> bool:
        return self._reference is not None

    # ------------------------------------------------------------------
    # Video
    # ------------------------------------------------------------------
    def on_video_frame(self, jpeg_bytes: bytes) -> List[Emission]:
        frame = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return []  # corrupt/partial JPEG -- drop this frame, not fatal

        ts = self._clock()
        with self._lock:
            self._frame_count += 1
            try:
                obs = self._observe_frame(frame, ts)
            except Exception:
                # One bad frame (or a model hiccup) must not end the exam's
                # analysis, let alone the WebSocket: log it and carry on.
                log.exception("frame analysis failed for session %s", self.session_id)
                return []
            self.evidence.add_frame(ts, frame)
            return self._emit(self.pipeline.on_frame(obs))

    def _observe_frame(self, frame: np.ndarray, ts: float) -> FrameObservation:
        quality, self._prev_thumb = compute_quality(frame, self._prev_thumb)

        # MediaPipe VIDEO mode needs strictly increasing millisecond stamps.
        ms = max(int(ts * 1000), self._last_ms + 1)
        self._last_ms = ms
        pose = self.estimator.process_frame(frame, ms)
        faces = self._faces_from_pose(pose)

        try:
            objects = detect_objects(frame)
        except Exception as exc:  # a broken model must not silence every other rule
            log.warning("object detection failed: %s", exc)
            self.shared.errors["objects"] = str(exc)
            objects = []

        similarity, skipped = self._score_identity(frame, pose)
        return FrameObservation(
            ts=ts,
            faces=faces,
            objects=objects,
            quality=quality,
            identity_similarity=similarity,
            identity_skipped_reason=skipped,
        )

    @staticmethod
    def _faces_from_pose(pose: PoseResult) -> List[FaceObservation]:
        if not pose.face_found or not pose.face_boxes:
            return []
        faces: List[FaceObservation] = []
        for i, b in enumerate(pose.face_boxes):
            box = BoundingBox(b["x"], b["y"], b["width"], b["height"])
            if i == 0:  # pose is only estimated for the primary (largest) face
                faces.append(
                    FaceObservation(
                        box=box,
                        yaw=pose.yaw,
                        pitch=pose.pitch,
                        roll=pose.roll,
                        gaze_x=pose.gaze_x,
                        gaze_y=pose.gaze_y,
                    )
                )
            else:
                faces.append(FaceObservation(box=box))
        return faces

    def _score_identity(self, frame: np.ndarray, pose: PoseResult) -> tuple[Optional[float], Optional[str]]:
        if self._reference is None:
            return None, None
        if self._frame_count % _IDENTITY_EVERY_N_FRAMES:
            return None, None
        if not pose.face_found:
            return None, "no_face"
        cfg = self.cfg
        if abs(pose.yaw) > cfg.identity_max_yaw_deg or abs(pose.pitch) > cfg.identity_max_pitch_deg:
            return None, "face_not_frontal"
        verifier = self.shared.identity
        if verifier is None:
            return None, "identity_unavailable"
        try:
            return verifier.score(
                frame,
                self._reference,
                min_sharpness=cfg.identity_min_sharpness,
                min_area=cfg.identity_min_face_area,
                max_yaw_ratio=0.35,
            )
        except Exception as exc:
            log.warning("identity scoring failed: %s", exc)
            return None, "error"

    # ------------------------------------------------------------------
    # Audio
    # ------------------------------------------------------------------
    def on_audio_chunk(self, pcm_bytes: bytes) -> List[Emission]:
        out: List[Emission] = []
        with self._lock:
            self._audio_buffer.extend(pcm_bytes)
            while len(self._audio_buffer) >= _AUDIO_WINDOW_BYTES:
                window = bytes(self._audio_buffer[:_AUDIO_WINDOW_BYTES])
                del self._audio_buffer[:_AUDIO_WINDOW_BYTES]
                out += self._process_audio_window(window)
        return out

    def _process_audio_window(self, window: bytes) -> List[Emission]:
        ts = self._clock()
        try:
            speech_ratio = self.vad.speech_ratio(window)
        except Exception:
            log.exception("VAD failed for session %s", self.session_id)
            speech_ratio = 0.0
        self._speech_tail = (self._speech_tail + window)[-_SPEAKER_CONTEXT_BYTES:]

        embedding = None
        if speech_ratio >= self.cfg.voice_min_speech_ratio and len(self._speech_tail) >= SPEAKER_MIN_SAMPLES * 2:
            speaker = self.shared.speaker
            if speaker is not None:
                try:
                    embedding = speaker.embed(self._speech_tail)
                except Exception as exc:
                    log.warning("speaker embedding failed: %s", exc)

        self.evidence.add_audio(ts, window)
        obs = analyze_window(window, ts, speech_ratio=speech_ratio, speaker_embedding=embedding)
        return self._emit(self.pipeline.on_audio(obs))

    # ------------------------------------------------------------------
    def on_tick(self) -> List[Emission]:
        with self._lock:
            return self._emit(self.pipeline.on_tick(self._clock()))

    def analysis_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            snap = self.pipeline.snapshot()
        snap["identity_enrolled"] = self.has_reference
        return snap

    def _emit(self, events: List[AlertEvent]) -> List[Emission]:
        return [Emission(e, self.evidence.build(e)) for e in events]

    def close(self) -> None:
        self.estimator.close()
        self.vad.close()
        # The reference embedding is biometric data: drop it eagerly rather
        # than waiting for garbage collection.
        self._reference = None


class ProctorSessionManager:
    def __init__(self, shared: Optional[SharedModels] = None) -> None:
        self.shared = shared or SharedModels()
        self._sessions: Dict[str, ProctorSession] = {}
        self._lock = threading.Lock()

    def get_or_create(self, session_id: str) -> ProctorSession:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                session = ProctorSession(session_id, self.shared)
                self._sessions[session_id] = session
            return session

    def remove(self, session_id: str) -> None:
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is not None:
            session.close()

    def count(self) -> int:
        return len(self._sessions)
