"""
Microphone / audio rules.

Speech by the candidate is *not itself suspicious* (people read aloud, mutter,
sigh), which is why the previous `voice_detected` alert -- one per second of
speech -- was noise. The rules here are about patterns: speech that keeps
going, speech that isn't the candidate's, speech that's been deliberately
lowered, and the microphone itself misbehaving.

Correlation with what the candidate is doing on the machine (speaking while
not typing or clicking) lives on the desktop side, which owns input-idle
time; this service only knows about audio.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Deque, List, Optional, Tuple

from observations import AlertEvent, AudioObservation
from pipeline_config import PipelineConfig
from rules.episodes import AlertGate, RatioEpisode, RollingCounter


class SpeechRule:
    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        self._gate = AlertGate()
        self._episode = RatioEpisode(
            window_s=cfg.speaking_window_s,
            on_ratio=cfg.speaking_on_ratio,
            min_samples=cfg.speaking_min_samples,
            off_ratio=0.2,
            off_window_s=4.0,
        )
        self._escalated = False

    def on_audio(self, obs: AudioObservation) -> List[AlertEvent]:
        events: List[AlertEvent] = []

        if obs.is_speech and self._gate.allow("speech", obs.ts, self.cfg.speech_detected_cooldown_s):
            events.append(
                AlertEvent(
                    alert_type="VOICE_DETECTED",
                    severity="LOW",
                    ts=obs.ts,
                    description="Speech was detected on the microphone.",
                    details={"level_dbfs": round(obs.rms_dbfs, 1)},
                )
            )

        transition = self._episode.update(obs.is_speech, obs.ts)
        if transition == "start":
            self._escalated = False
            events.append(
                AlertEvent(
                    alert_type="SUSTAINED_SPEAKING",
                    severity="MEDIUM",
                    ts=obs.ts,
                    description="Sustained speech was detected on the microphone.",
                    details={"since": self._episode.active_since},
                    evidence=True,
                    evidence_kind="audio",
                )
            )
        elif transition == "end":
            self._escalated = False
        elif (
            self._episode.is_active
            and not self._escalated
            and self._episode.duration(obs.ts) >= self.cfg.speaking_escalate_after_s
        ):
            self._escalated = True
            events.append(
                AlertEvent(
                    alert_type="SUSTAINED_SPEAKING",
                    severity="HIGH",
                    ts=obs.ts,
                    description="Speech has continued for an extended period.",
                    details={"duration": round(self._episode.duration(obs.ts), 1), "escalated": True},
                    evidence=True,
                    evidence_kind="audio",
                )
            )
        return events

    @property
    def speaking(self) -> bool:
        return self._episode.is_active


class WhisperRule:
    """Whispering: energy clearly above the room's noise floor, but quiet,
    unvoiced (noise-like spectrum), and *missed by the VAD* -- a whisper is
    exactly what speech detectors tend to skip. Heuristic by nature, so it is
    capped at MEDIUM and carries a low confidence."""

    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        self._floor: Deque[float] = deque(maxlen=90)
        self._episode = RatioEpisode(
            window_s=cfg.whisper_window_s,
            on_ratio=cfg.whisper_on_ratio,
            min_samples=5,
            off_ratio=0.2,
            off_window_s=4.0,
        )

    def _noise_floor(self) -> Optional[float]:
        if len(self._floor) < 20:
            return None  # not enough history to know what "quiet room" looks like
        ordered = sorted(self._floor)
        return ordered[int(len(ordered) * 0.1)]

    def on_audio(self, obs: AudioObservation) -> List[AlertEvent]:
        floor = self._noise_floor()
        whisper_like = (
            floor is not None
            and not obs.is_speech
            and obs.rms_dbfs < self.cfg.whisper_max_dbfs
            and obs.rms_dbfs > floor + self.cfg.whisper_min_above_floor_db
            and obs.spectral_flatness >= self.cfg.whisper_min_flatness
        )
        # Only learn the floor from windows that aren't whisper-like or
        # speech, or the whisper itself would raise the "floor" and hide.
        if not obs.is_speech and not whisper_like and math.isfinite(obs.rms_dbfs):
            self._floor.append(obs.rms_dbfs)

        if self._episode.update(whisper_like, obs.ts) == "start":
            return [
                AlertEvent(
                    alert_type="WHISPER_SUSPECTED",
                    severity="MEDIUM",
                    ts=obs.ts,
                    description="Quiet, unvoiced sound consistent with whispering was detected.",
                    details={"level_dbfs": round(obs.rms_dbfs, 1), "noise_floor_dbfs": round(floor or 0.0, 1)},
                    confidence=0.4,
                    evidence=True,
                    evidence_kind="audio",
                )
            ]
        return []


class MicrophoneStateRule:
    """The microphone going dead, dropping in and out, or vanishing."""

    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        self._silent = RatioEpisode(
            window_s=cfg.mic_silent_window_s,
            on_ratio=cfg.mic_silent_on_ratio,
            min_samples=8,
            off_ratio=0.5,
            off_window_s=3.0,
        )
        self._interruptions = RollingCounter(cfg.audio_interruption_window_s)
        self._gate = AlertGate()
        self._last_audio: Optional[float] = None
        self._lost = False

    def on_audio(self, obs: AudioObservation) -> List[AlertEvent]:
        events: List[AlertEvent] = []
        if self._lost:
            self._lost = False
            events.append(
                AlertEvent(
                    alert_type="AUDIO_FEED_RESTORED",
                    severity="LOW",
                    ts=obs.ts,
                    description="Audio is arriving again.",
                    details={"gap_seconds": round(obs.ts - (self._last_audio or obs.ts), 1)},
                )
            )
        self._last_audio = obs.ts

        transition = self._silent.update(obs.rms_dbfs < self.cfg.mic_silent_dbfs, obs.ts)
        if transition == "start":
            events.append(
                AlertEvent(
                    alert_type="MICROPHONE_SILENT",
                    severity="MEDIUM",
                    ts=obs.ts,
                    description="The microphone is producing no signal (muted or disabled).",
                    details={"level_dbfs": round(obs.rms_dbfs, 1)},
                )
            )
            count = self._interruptions.add(obs.ts)
            if count >= self.cfg.audio_interruption_count and self._gate.allow(
                "interruptions", obs.ts, self.cfg.audio_interruption_window_s
            ):
                events.append(
                    AlertEvent(
                        alert_type="AUDIO_INTERRUPTIONS",
                        severity="MEDIUM",
                        ts=obs.ts,
                        description="The microphone has repeatedly gone silent.",
                        details={"count": count, "window_s": self.cfg.audio_interruption_window_s},
                    )
                )
        elif transition == "end":
            events.append(
                AlertEvent(
                    alert_type="MICROPHONE_RESTORED",
                    severity="LOW",
                    ts=obs.ts,
                    description="The microphone is producing signal again.",
                    details={"silent_seconds": round(self._silent.last_duration, 1)},
                )
            )
        return events

    def on_tick(self, ts: float) -> List[AlertEvent]:
        if self._last_audio is None or self._lost:
            return []
        gap = ts - self._last_audio
        if gap >= self.cfg.audio_feed_lost_s:
            self._lost = True
            return [
                AlertEvent(
                    alert_type="AUDIO_FEED_LOST",
                    severity="MEDIUM",
                    ts=ts,
                    description="No audio has been received.",
                    details={"gap_seconds": round(gap, 1)},
                )
            ]
        return []


def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return 0.0 if na == 0 or nb == 0 else dot / (na * nb)


class MultipleVoicesRule:
    """A second, different voice. Requires speaker embeddings (see
    speaker_embedding.py) -- without an embedder the perception layer never
    fills `speaker_embedding` and this rule is simply inert, rather than
    guessing from pitch.

    Online clustering: each embedded speech window joins the nearest existing
    voice if it's similar enough, otherwise starts a new one. Two voices each
    seen `voice_other_needed` times within the window => MULTIPLE_VOICES. One
    odd-sounding cough can't start a "second speaker"."""

    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        self._centroids: List[List[float]] = []
        self._counts: List[int] = []
        self._recent: Deque[Tuple[float, int]] = deque()
        self._gate = AlertGate()

    def _assign(self, emb: List[float]) -> int:
        best, best_sim = -1, -1.0
        for i, c in enumerate(self._centroids):
            s = _cosine(emb, c)
            if s > best_sim:
                best, best_sim = i, s
        if best >= 0 and best_sim >= self.cfg.voice_same_speaker_above:
            n = self._counts[best]
            merged = [(c * n + e) / (n + 1) for c, e in zip(self._centroids[best], emb)]
            norm = math.sqrt(sum(x * x for x in merged)) or 1.0
            self._centroids[best] = [x / norm for x in merged]
            self._counts[best] = n + 1
            return best
        self._centroids.append(list(emb))
        self._counts.append(1)
        return len(self._centroids) - 1

    def on_audio(self, obs: AudioObservation) -> List[AlertEvent]:
        if obs.speaker_embedding is None or obs.speech_ratio < self.cfg.voice_min_speech_ratio:
            return []
        cluster = self._assign(obs.speaker_embedding)
        self._recent.append((obs.ts, cluster))
        cutoff = obs.ts - self.cfg.voice_window_s
        while self._recent and self._recent[0][0] < cutoff:
            self._recent.popleft()

        tally: dict[int, int] = {}
        for _, c in self._recent:
            tally[c] = tally.get(c, 0) + 1
        voices = [c for c, n in tally.items() if n >= self.cfg.voice_other_needed]
        if len(voices) >= 2 and self._gate.allow("voices", obs.ts, 120.0):
            return [
                AlertEvent(
                    alert_type="MULTIPLE_VOICES",
                    severity="HIGH",
                    ts=obs.ts,
                    description="More than one distinct voice was detected.",
                    details={"distinct_voices": len(voices), "window_s": self.cfg.voice_window_s},
                    evidence=True,
                    evidence_kind="audio",
                )
            ]
        return []
