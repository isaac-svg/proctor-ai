"""
Per-session state for the video-stream analysis service (service.py).

main.py's local demo uses bare module-level globals (`state = {...}`,
`active_flags = {}`) for exactly one implicit session -- the camera in front
of it. A network service handling many exam-takers concurrently needs one
HeadPoseEstimator/EventDetector/VAD-state per session_id instead, torn down
when that session's connection closes. That's what this module adds; it
doesn't change main.py's own demo loop at all.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from events import EventDetector
from facetracking import HeadPoseEstimator, PoseResult, gaze_bucket
from obd import objectDetection
from vad import VoiceActivityDectector

# obd.py's `model = YOLO(...)` is a single module-level instance shared by
# every ProctorSession (loading YOLO weights per-session would be wasteful
# and slow). Ultralytics' .predict() isn't guaranteed safe to call from
# multiple threads concurrently against one shared model, and service.py
# offloads each session's frame handling to a thread-pool executor (see its
# own comments) -- so calls into it are serialized here rather than trusting
# concurrent access to be safe.
_obd_lock = threading.Lock()

# One VAD analysis per ~1s of buffered audio, matching vad.py's own
# RATE/CHUNK constants (16kHz, s16le mono => 32000 bytes/second).
_AUDIO_BYTES_PER_ANALYSIS = 16000 * 2

# Detected face count above 1 is itself worth surfacing (API_CONTRACTS.md
# §3's MULTIPLE_PERSONS alert), so the estimator is asked for a few faces
# even though only the primary one drives head-pose/gaze events.
_MAX_FACES = 3


class ProctorSession:
    """All per-session analysis state: one estimator, one event pipeline,
    one rolling audio buffer. Not thread-safe on its own -- service.py is
    responsible for not calling into the same session's methods concurrently
    from two frames at once (a per-session asyncio.Lock, or simply not
    awaiting two handlers for the same session_id in parallel)."""

    def __init__(self, session_id: str, frame_width: int = 640, frame_height: int = 480):
        self.session_id = session_id
        self.estimator = HeadPoseEstimator(frame_width, frame_height, num_faces=_MAX_FACES)
        self.detector = EventDetector()
        self.obd = objectDetection()
        self.vad = VoiceActivityDectector(open_stream=False)

        self.state: Dict[str, Any] = {"is_speaking": False, "isCellphone_detected": False}
        self._audio_buffer = bytearray()
        self.last_pose: Optional[PoseResult] = None
        self.created_at = time.monotonic()

    def on_video_frame(self, jpeg_bytes: bytes) -> List[Dict[str, Any]]:
        """Decode one JPEG snapshot and run the full detection pipeline.
        Returns raw events.py-shaped event dicts (map via alert_mapping.py
        before sending as AI_ALERT)."""
        arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return []  # corrupt/partial JPEG -- drop this frame, not fatal

        ts = time.time()
        pose = self.estimator.process_frame(frame, int(ts * 1000))
        self.last_pose = pose

        with _obd_lock:
            self.state["isCellphone_detected"] = self.obd.predict(frame)

        # Deliberately the *only* call site for detector.update(): running
        # it again from on_audio_chunk() with pose=None would make
        # HeadPoseSubDetector reset its _away_start timer (it treats
        # pose=None as "no face"), which would starve sustained_look_away
        # of the continuous window it needs whenever an audio chunk lands
        # between video frames. Audio-derived state (is_speaking) still
        # feeds in below -- it's just folded into the next video-driven call
        # rather than triggering its own.
        return self.detector.update(pose=pose, state=self.state, ts=ts)

    def on_audio_chunk(self, pcm_bytes: bytes) -> List[Dict[str, Any]]:
        """Buffer raw PCM and update self.state['is_speaking'] once enough
        has accumulated. Never calls detector.update() itself -- see
        on_video_frame()'s comment for why. Always returns []; kept as a
        list-returning method for symmetry with on_video_frame()."""
        self._audio_buffer.extend(pcm_bytes)
        if len(self._audio_buffer) >= _AUDIO_BYTES_PER_ANALYSIS:
            chunk = bytes(self._audio_buffer[:_AUDIO_BYTES_PER_ANALYSIS])
            del self._audio_buffer[:_AUDIO_BYTES_PER_ANALYSIS]
            self.state["is_speaking"] = self.vad.analyze_pcm(chunk)
        return []

    def analysis_snapshot(self) -> Dict[str, Any]:
        """A cheap instantaneous heuristic for the periodic AI_ANALYSIS
        message -- NOT the persisted cheat score (that's shepherd-backend's
        risk-score.ts, computed from accumulated evidence_events instead)."""
        pose = self.last_pose
        if pose is None or not pose.face_found:
            return {
                "faces_detected": 0,
                "persons_in_frame": 0,
                "gaze_direction": "CENTER",
                "head_pose": {"yaw": 0, "pitch": 0, "roll": 0},
                "risk_score": 0.0,
            }

        turned = abs(pose.yaw) > self.detector.cfg.yaw_threshold_deg or abs(
            pose.pitch
        ) > self.detector.cfg.pitch_threshold_deg
        gaze_off = gaze_bucket(pose.gaze_x, pose.gaze_y) != "CENTER"
        phone = bool(self.state.get("isCellphone_detected", False))
        risk = min(1.0, 0.3 * turned + 0.3 * gaze_off + 0.4 * phone)

        return {
            "faces_detected": pose.num_faces,
            "persons_in_frame": pose.num_faces,
            "face_bounding_box": pose.face_bounding_box,
            "gaze_direction": gaze_bucket(pose.gaze_x, pose.gaze_y),
            "head_pose": {"yaw": pose.yaw, "pitch": pose.pitch, "roll": pose.roll},
            "risk_score": risk,
        }

    def close(self) -> None:
        self.estimator.close()
        self.vad.close()


class ProctorSessionManager:
    def __init__(self):
        self._sessions: Dict[str, ProctorSession] = {}
        self._lock = threading.Lock()

    def get_or_create(self, session_id: str) -> ProctorSession:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                session = ProctorSession(session_id)
                self._sessions[session_id] = session
            return session

    def remove(self, session_id: str) -> None:
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is not None:
            session.close()

    def count(self) -> int:
        return len(self._sessions)
