"""
Short evidence clips, kept only around flagged events.

Design constraint from the product decision: store *events* always, and a few
seconds of media only when something was flagged -- never a continuous
recording. So this holds a tiny rolling window in memory (the last ~12
downscaled frames and ~8 s of audio), and only when a rule raises an alert
with `evidence=True` does it copy a snapshot out. Anything not attached is
simply overwritten seconds later; nothing is written to disk here.

Clips are bounded three ways: severity floor, per-session count cap, and
size (frames downscaled to 320 px wide JPEG, audio limited to a few seconds).
"""

from __future__ import annotations

import base64
import io
import uuid
import wave
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from observations import SEVERITIES, AlertEvent, BoundingBox
from pipeline_config import PipelineConfig

_SAMPLE_RATE = 16000
_MAX_FRAMES = 12
_FRAME_WIDTH = 320
_JPEG_QUALITY = 55


def _rank(severity: str) -> int:
    return SEVERITIES.index(severity)


# Marker colour (BGR): a saturated orange-red that stands out on a webcam image without hiding it.
_MARK = (40, 90, 255)


def draw_annotations(frame_bgr: np.ndarray, annotations: Sequence[Tuple[str, BoundingBox]]) -> np.ndarray:
    """A copy of `frame_bgr` with each labelled box drawn on it."""
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    for label, box in annotations:
        x1 = int(max(0.0, box.x) * w)
        y1 = int(max(0.0, box.y) * h)
        x2 = int(min(1.0, box.x + box.width) * w)
        y2 = int(min(1.0, box.y + box.height) * h)
        if x2 <= x1 or y2 <= y1:
            continue
        cv2.rectangle(out, (x1, y1), (x2, y2), _MARK, 2)
        text = label.upper()
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        top = max(0, y1 - th - 6)
        cv2.rectangle(out, (x1, top), (min(w - 1, x1 + tw + 6), top + th + 6), _MARK, -1)
        cv2.putText(out, text, (x1 + 3, top + th + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return out


class EvidenceBuffer:
    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        self._frames: Deque[Tuple[float, bytes]] = deque(maxlen=_MAX_FRAMES)
        self._audio: Deque[Tuple[float, bytes]] = deque()
        self._audio_bytes = 0
        self._max_audio_bytes = int(8.0 * _SAMPLE_RATE * 2)
        self.attached = 0

    # ------------------------------------------------------------------
    def add_frame(self, ts: float, frame_bgr: np.ndarray, annotations: Sequence[Tuple[str, BoundingBox]] = ()) -> None:
        """Keeps a small copy of the frame. `annotations` are boxes (label, normalised box) drawn onto that
        copy -- the phone, the second person -- so a reviewer sees *what* was detected without having to find
        it. Only the stored thumbnail is marked; the analysed frame is untouched."""
        h, w = frame_bgr.shape[:2]
        if w > _FRAME_WIDTH:
            frame_bgr = cv2.resize(frame_bgr, (_FRAME_WIDTH, max(1, int(h * _FRAME_WIDTH / w))), interpolation=cv2.INTER_AREA)
        if annotations:
            frame_bgr = draw_annotations(frame_bgr, annotations)
        ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY])
        if ok:
            self._frames.append((ts, buf.tobytes()))

    def add_audio(self, ts: float, pcm: bytes) -> None:
        self._audio.append((ts, pcm))
        self._audio_bytes += len(pcm)
        while self._audio and self._audio_bytes > self._max_audio_bytes:
            _, dropped = self._audio.popleft()
            self._audio_bytes -= len(dropped)

    # ------------------------------------------------------------------
    def build(self, event: AlertEvent) -> Optional[Dict[str, Any]]:
        """A clip for `event`, or None if it shouldn't carry one."""
        if not event.evidence:
            return None
        if _rank(event.severity) < _rank(self.cfg.evidence_min_severity):
            return None
        if self.attached >= self.cfg.evidence_max_per_session:
            return None

        want_video = event.evidence_kind in ("video", "both")
        want_audio = event.evidence_kind in ("audio", "both")
        frames = self._pick_frames(event.ts) if want_video else []
        audio = self._audio_clip() if want_audio else None
        if not frames and audio is None:
            return None

        self.attached += 1
        clip: Dict[str, Any] = {"id": uuid.uuid4().hex, "frames": frames}
        if audio is not None:
            clip["audio"] = audio
        return clip

    def _pick_frames(self, ts: float) -> list:
        if not self._frames:
            return []
        n = self.cfg.evidence_frame_count
        # The newest frame, then frames roughly 2 s and 4 s earlier -- enough
        # to show what led up to the alert.
        targets = [ts - 2.0 * i for i in range(n)]
        chosen: Dict[float, bytes] = {}
        for target in targets:
            t, jpeg = min(self._frames, key=lambda f: abs(f[0] - target))
            chosen[t] = jpeg
        return [
            {"ts": t, "jpeg_b64": base64.b64encode(j).decode("ascii")}
            for t, j in sorted(chosen.items())
        ]

    def _audio_clip(self) -> Optional[Dict[str, Any]]:
        if not self._audio:
            return None
        want = int(self.cfg.evidence_audio_seconds * _SAMPLE_RATE * 2)
        pcm = b"".join(chunk for _, chunk in self._audio)[-want:]
        if not pcm:
            return None
        out = io.BytesIO()
        with wave.open(out, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(_SAMPLE_RATE)
            w.writeframes(pcm)
        return {
            "sample_rate": _SAMPLE_RATE,
            "seconds": round(len(pcm) / 2 / _SAMPLE_RATE, 2),
            "wav_b64": base64.b64encode(out.getvalue()).decode("ascii"),
        }
