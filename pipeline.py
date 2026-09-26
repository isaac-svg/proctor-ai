"""
ProctorPipeline: composes every rule over one session's observations.

Feed it observations (`on_frame` / `on_audio`) and a periodic `on_tick`; it
returns `AlertEvent`s. No model, no I/O, no wall clock -- so a whole exam can
be replayed against it in a unit test.

Cross-rule behaviour that individual rules can't know about lives here, e.g.
a covered camera makes "no face"/"looking away" meaningless, so those rules
are muted while the camera is covered instead of piling three alerts onto
one cause.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from gaze import gaze_bucket
from observations import AlertEvent, AudioObservation, FrameObservation
from pipeline_config import PipelineConfig
from rules.attention import LookAwayRule
from rules.audio import MicrophoneStateRule, MultipleVoicesRule, SpeechRule, WhisperRule
from rules.camera import CameraCoveredRule, CameraFeedWatchdog, CameraFrozenRule
from rules.identity import IdentityRule
from rules.objects import ProhibitedObjectRule
from rules.presence import FacePresenceRule, FramingRule, MultiplePersonsRule


class ProctorPipeline:
    def __init__(self, cfg: Optional[PipelineConfig] = None) -> None:
        self.cfg = cfg or PipelineConfig()
        c = self.cfg

        self.camera_covered = CameraCoveredRule(c)
        self.camera_frozen = CameraFrozenRule(c)
        self.camera_watchdog = CameraFeedWatchdog(c)
        self.presence = FacePresenceRule(c)
        self.multiple = MultiplePersonsRule(c)
        self.framing = FramingRule(c)
        self.attention = LookAwayRule(c)
        self.identity = IdentityRule(c)
        self.objects = ProhibitedObjectRule(c)

        self.speech = SpeechRule(c)
        self.whisper = WhisperRule(c)
        self.microphone = MicrophoneStateRule(c)
        self.voices = MultipleVoicesRule(c)

        self._last_frame: Optional[FrameObservation] = None
        self._last_audio: Optional[AudioObservation] = None

    # ------------------------------------------------------------------
    def on_frame(self, obs: FrameObservation) -> List[AlertEvent]:
        self._last_frame = obs
        events: List[AlertEvent] = []

        events += self.camera_watchdog.on_frame(obs)
        events += self.camera_covered.on_frame(obs)
        events += self.camera_frozen.on_frame(obs, covered=self.camera_covered.reason_for(obs) is not None)

        # Everything below reasons about *what is in the picture*. If the
        # picture is a covered lens, that reasoning is meaningless (and would
        # raise face-absent / look-away noise for the one underlying cause),
        # so hold those rules until the lens is clear again.
        if self.camera_covered.is_covered:
            return events

        presence_events = self.presence.on_frame(obs)
        events += presence_events
        for e in presence_events:
            if e.alert_type == "FACE_RETURNED":
                # Whoever came back is compared with whoever left, with a shorter run of evidence.
                self.identity.on_face_returned(obs.ts, float(e.details.get("absent_seconds", 0.0)))
        events += self.multiple.on_frame(obs)
        events += self.framing.on_frame(obs)
        events += self.attention.on_frame(obs)
        events += self.identity.on_frame(obs)
        events += self.objects.on_frame(obs)
        return events

    def annotations(self) -> List[tuple]:
        """Boxes worth marking on the evidence frame for the observation just processed: confirmed prohibited
        objects, and -- while more than one person is in view -- the people, so a reviewer can see who was counted."""
        marks: List[tuple] = list(self.objects.confirmed_boxes())
        f = self._last_frame
        if f is not None and self.multiple.person_count >= 2:
            marks += [("person", p.box) for p in f.persons if p.confidence >= self.cfg.person_min_confidence and p.box.area >= self.cfg.person_min_area]
        return marks

    def on_audio(self, obs: AudioObservation) -> List[AlertEvent]:
        self._last_audio = obs
        events: List[AlertEvent] = []
        events += self.microphone.on_audio(obs)
        events += self.speech.on_audio(obs)
        events += self.whisper.on_audio(obs)
        events += self.voices.on_audio(obs)
        return events

    def on_tick(self, ts: float) -> List[AlertEvent]:
        """Call every ~2s. Detects things that are visible only as an
        *absence* of input (feeds going quiet)."""
        return self.camera_watchdog.on_tick(ts) + self.microphone.on_tick(ts)

    # ------------------------------------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        """A cheap instantaneous view for the periodic AI_ANALYSIS message --
        NOT the persisted cheat score (that's shepherd-backend's
        risk-score.ts, computed from accumulated evidence_events)."""
        f = self._last_frame
        if f is None or not f.faces:
            return {
                "faces_detected": 0,
                "persons_in_frame": self.multiple.person_count if f else 0,
                "gaze_direction": "CENTER",
                "head_pose": {"yaw": 0, "pitch": 0, "roll": 0},
                "camera_covered": self.camera_covered.is_covered,
                "risk_score": 0.6 if (f is not None and self.presence.face_absent) else 0.0,
            }
        p = f.faces[0]
        cfg = self.cfg
        turned = abs(p.yaw) > cfg.yaw_threshold_deg or abs(p.pitch) > cfg.pitch_threshold_deg
        gaze = gaze_bucket(p.gaze_x, p.gaze_y)
        risk = min(
            1.0,
            0.3 * turned
            + 0.2 * (gaze != "CENTER")
            + 0.4 * ("cell phone" in self.objects.confirmed)
            + 0.5 * (self.multiple.person_count >= 2)
            + 0.5 * self.camera_covered.is_covered,
        )
        snap: Dict[str, Any] = {
            "faces_detected": len(f.faces),
            "persons_in_frame": self.multiple.person_count,
            "face_bounding_box": p.box.to_dict(),
            "gaze_direction": gaze,
            "head_pose": {"yaw": p.yaw, "pitch": p.pitch, "roll": p.roll},
            "camera_covered": self.camera_covered.is_covered,
            "risk_score": round(risk, 2),
        }
        if self.identity.last_similarity is not None:
            snap["identity_similarity"] = round(self.identity.last_similarity, 3)
        if self._last_audio is not None:
            snap["audio_level_dbfs"] = round(self._last_audio.rms_dbfs, 1)
            snap["speaking"] = self.speech.speaking
        return snap
