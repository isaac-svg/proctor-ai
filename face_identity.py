"""
Identity consistency: face embeddings via OpenCV's YuNet (detection) and
SFace (recognition), both Apache-2.0 models from the OpenCV Zoo, run through
cv2 itself -- no extra runtime dependency.

Privacy shape: the reference *image* is used once, to compute an embedding,
and is never stored by this service; the embedding lives in the session's
memory and is dropped when the session closes. Nothing here writes a face to
disk.

Two operations:
  enroll(frames)  - validate a check-in capture strictly and return its
                    embedding (mean of the valid frames).
  score(frame)    - during the exam: similarity of the largest face in the
                    frame to the reference, or None (with a reason) when the
                    frame isn't good enough to judge -- see identity_gate().
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import List, Optional, Sequence

import cv2
import numpy as np

from model_assets import SFACE, YUNET, ensure_asset

# YuNet output row: x, y, w, h, then 5 landmarks (right eye, left eye, nose
# tip, right mouth corner, left mouth corner) as x,y pairs, then score.
_R_EYE, _L_EYE, _NOSE = (4, 5), (6, 7), (8, 9)


@dataclass
class EnrollmentResult:
    ok: bool
    reason: Optional[str] = None
    embedding: Optional[List[float]] = None
    faces_seen: int = 0


def _l2(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.float64).ravel()
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    return float(np.dot(_l2(np.asarray(a)), _l2(np.asarray(b))))


def _yaw_ratio(face_row: np.ndarray) -> float:
    """Rough frontalness from YuNet's landmarks: horizontal offset of the nose
    from the midpoint of the eyes, relative to the eye distance. ~0 = facing
    the camera; grows as the head turns."""
    r_eye = face_row[list(_R_EYE)]
    l_eye = face_row[list(_L_EYE)]
    nose = face_row[list(_NOSE)]
    eye_dist = float(np.linalg.norm(r_eye - l_eye)) or 1.0
    return abs(float(nose[0] - (r_eye[0] + l_eye[0]) / 2.0)) / eye_dist


class FaceIdentityVerifier:
    """Not thread-safe (cv2.dnn nets aren't); callers hold `lock`."""

    def __init__(self) -> None:
        self._detector = cv2.FaceDetectorYN.create(
            str(ensure_asset(YUNET)), "", (320, 320), score_threshold=0.8, nms_threshold=0.3, top_k=8
        )
        self._recognizer = cv2.FaceRecognizerSF.create(str(ensure_asset(SFACE)), "")
        self.lock = threading.Lock()

    # ------------------------------------------------------------------
    def _detect(self, frame_bgr: np.ndarray) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        self._detector.setInputSize((w, h))
        _, faces = self._detector.detect(frame_bgr)
        if faces is None:
            return np.empty((0, 15), dtype=np.float32)
        order = np.argsort(-(faces[:, 2] * faces[:, 3]))  # largest first
        return faces[order]

    def _embed_row(self, frame_bgr: np.ndarray, row: np.ndarray) -> np.ndarray:
        aligned = self._recognizer.alignCrop(frame_bgr, row)
        return _l2(self._recognizer.feature(aligned))

    # ------------------------------------------------------------------
    def enroll(self, frames: Sequence[np.ndarray]) -> EnrollmentResult:
        """Strict on purpose: a wrong reference (two people in shot, a
        profile view, a dark room) would make every later comparison
        meaningless -- better to make the candidate retake it at check-in."""
        embeddings: List[np.ndarray] = []
        worst_reason: Optional[str] = None
        max_faces = 0
        with self.lock:
            for frame in frames:
                faces = self._detect(frame)
                max_faces = max(max_faces, len(faces))
                reason = self._enrollment_problem(frame, faces)
                if reason:
                    worst_reason = worst_reason or reason
                    continue
                embeddings.append(self._embed_row(frame, faces[0]))

        # Require most captures to be usable, not just one lucky frame.
        if len(embeddings) < max(1, (len(frames) + 1) // 2):
            return EnrollmentResult(False, worst_reason or "no_face", None, max_faces)
        mean = _l2(np.mean(np.stack(embeddings), axis=0))
        return EnrollmentResult(True, None, [float(x) for x in mean], max_faces)

    def _enrollment_problem(self, frame: np.ndarray, faces: np.ndarray) -> Optional[str]:
        h, w = frame.shape[:2]
        if len(faces) == 0:
            return "no_face"
        first = faces[0]
        if len(faces) > 1 and (faces[1][2] * faces[1][3]) >= 0.25 * (first[2] * first[3]):
            return "multiple_faces"
        if (first[2] * first[3]) / float(w * h) < 0.03:
            return "face_too_small"
        if _yaw_ratio(first) > 0.25:
            return "face_not_frontal"
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if gray.mean() < 45:
            return "too_dark"
        if cv2.Laplacian(gray, cv2.CV_64F).var() < 25:
            return "too_blurry"
        return None

    # ------------------------------------------------------------------
    def embed_primary(
        self, frame_bgr: np.ndarray, min_sharpness: float, min_area: float, max_yaw_ratio: float
    ) -> tuple[Optional[List[float]], Optional[str]]:
        """(embedding, None) of the largest face, or (None, reason_skipped) when the frame is not good enough
        to judge. Used both to compare against the check-in reference and, independently, against the
        face seen earlier in the same exam."""
        with self.lock:
            faces = self._detect(frame_bgr)
            if len(faces) == 0:
                return None, "no_face"
            row = faces[0]
            h, w = frame_bgr.shape[:2]
            if (row[2] * row[3]) / float(w * h) < min_area:
                return None, "face_too_small"
            if _yaw_ratio(row) > max_yaw_ratio:
                return None, "face_not_frontal"
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            if cv2.Laplacian(gray, cv2.CV_64F).var() < min_sharpness:
                return None, "too_blurry"
            return [float(x) for x in self._embed_row(frame_bgr, row)], None

    def score(
        self, frame_bgr: np.ndarray, reference: Sequence[float], min_sharpness: float, min_area: float, max_yaw_ratio: float
    ) -> tuple[Optional[float], Optional[str]]:
        """(similarity, None) or (None, reason_skipped)."""
        with self.lock:
            faces = self._detect(frame_bgr)
            if len(faces) == 0:
                return None, "no_face"
            row = faces[0]
            h, w = frame_bgr.shape[:2]
            if (row[2] * row[3]) / float(w * h) < min_area:
                return None, "face_too_small"
            if _yaw_ratio(row) > max_yaw_ratio:
                return None, "face_not_frontal"
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            if cv2.Laplacian(gray, cv2.CV_64F).var() < min_sharpness:
                return None, "too_blurry"
            emb = self._embed_row(frame_bgr, row)
        return cosine(emb, reference), None
