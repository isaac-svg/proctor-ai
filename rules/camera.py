"""
Camera integrity: covered, blocked, frozen/looped, or gone.

A candidate can defeat every face-based rule by simply not giving it a face
to look at, so these fire on the *image itself* rather than on what's in it.
"""

from __future__ import annotations

from typing import List, Optional

from observations import AlertEvent, FrameObservation
from pipeline_config import PipelineConfig
from rules.episodes import RatioEpisode


class CameraCoveredRule:
    """A dark or featureless frame for most of a few seconds.

    Deliberately narrow. Two signals are unambiguous enough to accuse on:
      * "dark"    -- the frame is nearly black (lens covered, camera off, or
                     the lights out);
      * "blocked" -- almost no contrast *and* almost no edges at all (a
                     finger, cloth or paper flush against the lens).
    A candidate stepping away from a plain wall also produces a low-detail
    frame, but that's an empty chair, not a covered lens: it is reported by
    FacePresenceRule as FACE_ABSENT instead of being accused here."""

    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        self._episode = RatioEpisode(
            window_s=cfg.camera_covered_window_s,
            on_ratio=cfg.camera_covered_on_ratio,
            min_samples=3,
            off_ratio=0.2,
            off_window_s=3.0,
        )
        self._reason: Optional[str] = None

    @property
    def is_covered(self) -> bool:
        return self._episode.is_active

    def reason_for(self, obs: FrameObservation) -> Optional[str]:
        q = obs.quality
        if q.mean_luma < self.cfg.camera_dark_luma:
            return "dark"
        if q.luma_std < self.cfg.camera_blocked_std and q.sharpness < self.cfg.camera_blocked_sharpness:
            return "blocked"
        return None

    def on_frame(self, obs: FrameObservation) -> List[AlertEvent]:
        reason = self.reason_for(obs)
        if reason:
            self._reason = reason
        transition = self._episode.update(active=reason is not None, ts=obs.ts)
        if transition == "start":
            q = obs.quality
            return [
                AlertEvent(
                    alert_type="CAMERA_COVERED",
                    severity="HIGH",
                    ts=obs.ts,
                    description="The camera view appears to be covered or blocked.",
                    details={
                        "reason": self._reason,
                        "mean_luma": round(q.mean_luma, 1),
                        "contrast": round(q.luma_std, 1),
                        "sharpness": round(q.sharpness, 1),
                    },
                    evidence=True,
                )
            ]
        return []


class CameraFrozenRule:
    """The picture stopped changing. A live sensor always has some noise, so
    many consecutive near-identical frames mean a frozen driver, a still
    image, or a short looped clip fed through a virtual camera."""

    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        self._still_since: Optional[float] = None
        self._reported = False

    def on_frame(self, obs: FrameObservation, covered: bool = False) -> List[AlertEvent]:
        motion = obs.quality.motion
        if covered or motion is None:
            # A dark/covered lens barely changes either, but that is already
            # reported as CAMERA_COVERED -- one cause, one alert.
            self._still_since = None
            self._reported = False
            return []
        if motion < self.cfg.camera_frozen_motion:
            if self._still_since is None:
                self._still_since = obs.ts
            elif not self._reported and obs.ts - self._still_since >= self.cfg.camera_frozen_seconds:
                self._reported = True
                return [
                    AlertEvent(
                        alert_type="CAMERA_FROZEN",
                        severity="HIGH",
                        ts=obs.ts,
                        description="The camera feed has stopped changing (frozen or looped video).",
                        details={"still_seconds": round(obs.ts - self._still_since, 1)},
                        evidence=True,
                    )
                ]
        else:
            self._still_since = None
            self._reported = False
        return []


class CameraFeedWatchdog:
    """Frames stopped arriving. Called from `tick()`, since by definition no
    frame is around to trigger it. The cause can be the camera, the encoder,
    or the network -- the alert says only what proctor-ai can actually
    know: it stopped receiving video."""

    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        self._last_frame: Optional[float] = None
        self._lost = False

    def on_frame(self, obs: FrameObservation) -> List[AlertEvent]:
        events: List[AlertEvent] = []
        if self._lost:
            events.append(
                AlertEvent(
                    alert_type="CAMERA_FEED_RESTORED",
                    severity="LOW",
                    ts=obs.ts,
                    description="Video frames are arriving again.",
                    details={"gap_seconds": round(obs.ts - (self._last_frame or obs.ts), 1)},
                )
            )
            self._lost = False
        self._last_frame = obs.ts
        return events

    def on_tick(self, ts: float) -> List[AlertEvent]:
        if self._last_frame is None or self._lost:
            return []
        gap = ts - self._last_frame
        if gap >= self.cfg.camera_feed_lost_s:
            self._lost = True
            return [
                AlertEvent(
                    alert_type="CAMERA_FEED_LOST",
                    severity="HIGH",
                    ts=ts,
                    description="No video frames have been received.",
                    details={"gap_seconds": round(gap, 1)},
                )
            ]
        return []
