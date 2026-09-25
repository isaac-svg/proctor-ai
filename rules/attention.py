"""
Head and eye attention.

Deliberately forgiving: looking away is normal while thinking. An alert needs
the candidate to be turned away for *most of a six-second window*, and
FREQUENT_LOOK_AWAY needs several *separate* episodes -- one long think is one
episode, not six.
"""

from __future__ import annotations

from typing import List

from observations import AlertEvent, FrameObservation
from pipeline_config import PipelineConfig
from rules.episodes import AlertGate, RatioEpisode, RollingCounter


class LookAwayRule:
    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        self._head = RatioEpisode(
            window_s=cfg.look_away_window_s,
            on_ratio=cfg.look_away_on_ratio,
            min_samples=cfg.look_away_min_samples,
            off_ratio=cfg.look_away_off_ratio,
            off_window_s=3.0,
        )
        self._gaze = RatioEpisode(
            window_s=cfg.gaze_window_s,
            on_ratio=cfg.gaze_on_ratio,
            min_samples=cfg.look_away_min_samples,
            off_ratio=cfg.look_away_off_ratio,
            off_window_s=3.0,
        )
        self._episodes = RollingCounter(cfg.frequent_look_away_window_s)
        self._gate = AlertGate()

    def _head_turned(self, obs: FrameObservation) -> bool:
        p = obs.primary
        if p is None:
            return False
        return abs(p.yaw) > self.cfg.yaw_threshold_deg or abs(p.pitch) > self.cfg.pitch_threshold_deg

    def _gaze_off(self, obs: FrameObservation) -> bool:
        p = obs.primary
        if p is None:
            return False
        head_still = abs(p.yaw) < self.cfg.gaze_head_still_yaw_deg
        eyes_off = abs(p.gaze_x) > self.cfg.gaze_threshold or abs(p.gaze_y) > self.cfg.gaze_threshold
        return head_still and eyes_off

    def on_frame(self, obs: FrameObservation) -> List[AlertEvent]:
        events: List[AlertEvent] = []
        primary = obs.primary

        # No face is FacePresenceRule's territory. Don't count it as
        # "looking away" -- and don't feed the episodes a sample either, so a
        # face dropping out for a moment can't fake or break an episode.
        if primary is None:
            return events

        head = self._head.update(self._head_turned(obs), obs.ts)
        if head == "start":
            events.append(
                AlertEvent(
                    alert_type="SUSTAINED_LOOK_AWAY",
                    severity="MEDIUM",
                    ts=obs.ts,
                    description="The candidate has been looking away from the screen.",
                    details={
                        "yaw": round(primary.yaw, 1),
                        "pitch": round(primary.pitch, 1),
                        "since": self._head.active_since,
                    },
                    evidence=False,
                )
            )
            count = self._episodes.add(obs.ts)
            if count >= self.cfg.frequent_look_away_count and self._gate.allow(
                "frequent", obs.ts, self.cfg.frequent_look_away_window_s
            ):
                events.append(
                    AlertEvent(
                        alert_type="FREQUENT_LOOK_AWAY",
                        severity="HIGH",
                        ts=obs.ts,
                        description="The candidate has looked away from the screen repeatedly.",
                        details={
                            "count": count,
                            "window_s": self.cfg.frequent_look_away_window_s,
                        },
                        evidence=True,
                    )
                )

        gaze = self._gaze.update(self._gaze_off(obs), obs.ts)
        if gaze == "start":
            events.append(
                AlertEvent(
                    alert_type="GAZE_AWAY",
                    severity="LOW",
                    ts=obs.ts,
                    description="Eyes have moved away from the screen without a head movement.",
                    details={"gaze_x": round(primary.gaze_x, 2), "gaze_y": round(primary.gaze_y, 2)},
                )
            )
        return events
