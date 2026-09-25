"""
Face presence, extra people, and framing.

Covers: face present/absent, the candidate repeatedly leaving the frame,
more than one person in view, and face position/size.
"""

from __future__ import annotations

from typing import List, Optional

from observations import AlertEvent, FrameObservation
from pipeline_config import PipelineConfig
from rules.episodes import AlertGate, RatioEpisode, RollingCounter


def _absence_severity(seconds: float) -> str:
    if seconds >= 60:
        return "HIGH"
    if seconds >= 15:
        return "MEDIUM"
    return "LOW"


class FacePresenceRule:
    """FACE_ABSENT when no face is visible for most of a window; FACE_RETURNED
    (with the duration) when they're back; FREQUENT_FRAME_EXIT when they keep
    leaving. A person who leaves for 3 seconds is not reported at all."""

    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        self._absent = RatioEpisode(
            window_s=cfg.face_absent_window_s,
            on_ratio=cfg.face_absent_on_ratio,
            min_samples=cfg.face_absent_min_samples,
            off_ratio=0.2,
            off_window_s=2.0,
        )
        self._exits = RollingCounter(cfg.frame_exit_window_s)
        self._gate = AlertGate()

    @property
    def face_absent(self) -> bool:
        return self._absent.is_active

    def on_frame(self, obs: FrameObservation) -> List[AlertEvent]:
        events: List[AlertEvent] = []
        transition = self._absent.update(active=not obs.faces, ts=obs.ts)

        if transition == "start":
            events.append(
                AlertEvent(
                    alert_type="FACE_ABSENT",
                    severity="MEDIUM",
                    ts=obs.ts,
                    description="No face has been visible to the camera.",
                    details={"since": self._absent.active_since},
                    evidence=True,
                )
            )
            exits = self._exits.add(obs.ts)
            if exits >= self.cfg.frame_exit_count and self._gate.allow(
                "frame_exit", obs.ts, self.cfg.frame_exit_window_s
            ):
                events.append(
                    AlertEvent(
                        alert_type="FREQUENT_FRAME_EXIT",
                        severity="HIGH",
                        ts=obs.ts,
                        description="The candidate has repeatedly left the camera frame.",
                        details={"exits": exits, "window_s": self.cfg.frame_exit_window_s},
                        evidence=True,
                    )
                )
        elif transition == "end":
            seconds = self._absent.last_duration
            events.append(
                AlertEvent(
                    alert_type="FACE_RETURNED",
                    severity=_absence_severity(seconds),
                    ts=obs.ts,
                    description="The candidate returned to the camera frame.",
                    details={"absent_seconds": round(seconds, 1)},
                )
            )
        return events


class MultiplePersonsRule:
    """More than one person in view.

    Counts the primary face plus any *secondary* face that is a plausible
    person (not a tiny poster/photo), and separately YOLO "person" boxes --
    the latter catches someone whose face is turned away from the camera.
    Requires the condition for most of a short window, so a passer-by in the
    background for one frame is ignored."""

    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        self._episode = RatioEpisode(
            window_s=cfg.multi_person_window_s,
            on_ratio=cfg.multi_person_on_ratio,
            min_samples=cfg.multi_person_min_samples,
            off_ratio=0.2,
            off_window_s=3.0,
        )
        self._latest_count = 1
        self._latest_source = "face"

    def _count(self, obs: FrameObservation) -> tuple[int, str]:
        face_count = 0
        if obs.faces:
            primary_area = obs.faces[0].box.area
            face_count = 1
            for f in obs.faces[1:]:
                if primary_area > 0 and f.box.area / primary_area >= self.cfg.secondary_face_min_area_ratio:
                    face_count += 1
        person_count = sum(
            1
            for p in obs.persons
            if p.confidence >= self.cfg.person_min_confidence and p.box.area >= self.cfg.person_min_area
        )
        if face_count >= person_count:
            return face_count, "face"
        return person_count, "body"

    def on_frame(self, obs: FrameObservation) -> List[AlertEvent]:
        count, source = self._count(obs)
        self._latest_count, self._latest_source = count, source
        transition = self._episode.update(active=count >= 2, ts=obs.ts)
        if transition != "start":
            return []
        return [
            AlertEvent(
                alert_type="MULTIPLE_PERSONS",
                severity="HIGH",
                ts=obs.ts,
                description=f"Detected {count} people in frame.",
                details={"person_count": count, "source": source},
                evidence=True,
            )
        ]

    @property
    def person_count(self) -> int:
        return self._latest_count


class FramingRule:
    """Face position and size: too far, too close, off-centre, partly out of
    frame. Advisory (LOW) -- the point is to let the student fix their setup
    before it becomes a FACE_ABSENT, and to give a proctor context."""

    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        self._episodes = {
            reason: RatioEpisode(
                window_s=cfg.framing_window_s,
                on_ratio=cfg.framing_on_ratio,
                min_samples=5,
                off_ratio=0.2,
                off_window_s=5.0,
            )
            for reason in ("too_far", "too_close", "off_center", "partially_out_of_frame")
        }
        self._gate = AlertGate()

    def _issue(self, obs: FrameObservation) -> Optional[str]:
        primary = obs.primary
        if primary is None:
            return None
        box = primary.box
        if box.clipped_area_ratio() < self.cfg.face_min_visible_ratio:
            return "partially_out_of_frame"
        if box.area < self.cfg.face_too_far_area:
            return "too_far"
        if box.area > self.cfg.face_too_close_area:
            return "too_close"
        cx, cy = box.center
        if abs(cx - 0.5) > self.cfg.face_off_center or abs(cy - 0.5) > self.cfg.face_off_center:
            return "off_center"
        return None

    def on_frame(self, obs: FrameObservation) -> List[AlertEvent]:
        if not obs.faces:
            # Absence is FacePresenceRule's job; don't double-report it here.
            for ep in self._episodes.values():
                ep.reset()
            return []
        current = self._issue(obs)
        events: List[AlertEvent] = []
        for reason, ep in self._episodes.items():
            transition = ep.update(active=(reason == current), ts=obs.ts)
            if transition == "start" and self._gate.allow(
                f"framing:{reason}", obs.ts, self.cfg.framing_cooldown_s
            ):
                box = obs.faces[0].box
                events.append(
                    AlertEvent(
                        alert_type="FACE_POSITION_ISSUE",
                        severity="LOW",
                        ts=obs.ts,
                        description=f"The candidate's face is {reason.replace('_', ' ')}.",
                        details={
                            "reason": reason,
                            "face_area": round(box.area, 4),
                            "center": [round(c, 3) for c in box.center],
                        },
                    )
                )
        return events
