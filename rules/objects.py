"""
Prohibited objects (phones, books, second laptops, and -- when an extra model
provides them -- earbuds, tablets, notes, smartwatches).

Detections are followed from frame to frame (rules/object_tracker.py) and a
label is only reported once a *track* is confirmed:

  * seen in `object_confirm_hits` of the last `object_confirm_window` frames
    with confidences that add up to `object_confirm_score`, or
  * a single sighting confident enough on its own (`object_single_shot_confidence`).

That is more forgiving of a detector that only catches a half-hidden phone
every other frame than "N frames in a row", and just as hard to fool with a
one-frame ghost. Alert types stay CELLPHONE_DETECTED for phones (existing
policies key on it) and PROHIBITED_OBJECT_DETECTED for everything else.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set

from observations import AlertEvent, BoundingBox, FrameObservation
from pipeline_config import PipelineConfig
from rules.episodes import AlertGate
from rules.object_tracker import ObjectTracker, Track


def _near_face(box: BoundingBox, obs: FrameObservation, distance: float) -> bool:
    """Is the object being held up to the candidate's face?"""
    primary = obs.primary
    if primary is None or primary.box.height <= 0:
        return False
    (fx, fy), (ox, oy) = primary.box.center, box.center
    reach = distance * primary.box.height
    return ((fx - ox) ** 2 + (fy - oy) ** 2) ** 0.5 <= reach


class ProhibitedObjectRule:
    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        self._tracker = ObjectTracker(cfg.object_track_iou, cfg.object_track_max_gap_s)
        self._gate = AlertGate()
        self._phone_since: Optional[float] = None
        self._phone_last_seen: Optional[float] = None
        self._phone_escalated = False
        self.confirmed: Set[str] = set()

    # ------------------------------------------------------------------
    def _confirmed(self, track: Track, ts: float) -> bool:
        cfg = self.cfg
        if track.max_confidence >= cfg.object_single_shot_confidence:
            return True
        # The window is counted in frames, but frames arrive at ~1/s and can be
        # missing, so it is measured in time: the last N frames' worth of seconds.
        recent = track.recent(ts - cfg.object_confirm_window)
        return len(recent) >= cfg.object_confirm_hits and sum(c for _, c in recent) >= cfg.object_confirm_score

    def on_frame(self, obs: FrameObservation) -> List[AlertEvent]:
        events: List[AlertEvent] = []
        cfg = self.cfg

        eligible = []
        for o in obs.objects:
            rule = cfg.object_rules.get(o.label)
            if rule is not None and o.confidence >= rule[0]:
                eligible.append(o)
        seen_tracks = self._tracker.update(obs.ts, eligible)

        confirmed_now: Dict[str, Track] = {}
        for track in self._tracker.tracks:
            if self._confirmed(track, obs.ts):
                best = confirmed_now.get(track.label)
                if best is None or track.max_confidence > best.max_confidence:
                    confirmed_now[track.label] = track
        self.confirmed = set(confirmed_now)

        in_this_frame = {t.id for t in seen_tracks}
        for label, track in confirmed_now.items():
            _, severity = cfg.object_rules[label]
            is_phone = label == "cell phone"
            if not self._gate.allow(f"object:{label}", obs.ts, cfg.object_cooldown_s):
                continue
            details = {
                "object": label,
                "box": track.box.to_dict(),
                "sightings": len(track.sightings),
                "tracked_seconds": round(track.age, 1),
                "near_face": _near_face(track.box, obs, cfg.near_face_distance),
            }
            events.append(
                AlertEvent(
                    alert_type="CELLPHONE_DETECTED" if is_phone else "PROHIBITED_OBJECT_DETECTED",
                    severity=severity,
                    ts=obs.ts,
                    description=("A cellphone was detected in frame." if is_phone else f"A {label} was detected in frame."),
                    details=details,
                    confidence=round(track.max_confidence, 2),
                    evidence=True,
                )
            )

        # A phone that *stays* in view is worse than one that merely appeared.
        phone = confirmed_now.get("cell phone")
        if phone is not None:
            if self._phone_since is None:
                self._phone_since = phone.first_ts
            self._phone_last_seen = obs.ts if phone.id in in_this_frame else self._phone_last_seen
            if not self._phone_escalated and obs.ts - self._phone_since >= cfg.phone_sustained_s:
                self._phone_escalated = True
                events.append(
                    AlertEvent(
                        alert_type="CELLPHONE_DETECTED",
                        severity="CRITICAL",
                        ts=obs.ts,
                        description="A cellphone has stayed in frame.",
                        details={
                            "object": "cell phone",
                            "duration": round(obs.ts - self._phone_since, 1),
                            "sustained": True,
                            "box": phone.box.to_dict(),
                            "near_face": _near_face(phone.box, obs, cfg.near_face_distance),
                        },
                        evidence=True,
                    )
                )
        else:
            self._phone_since = None
            self._phone_last_seen = None
            self._phone_escalated = False

        return events

    # Boxes the evidence buffer should draw for the current frame: confirmed objects only.
    def confirmed_boxes(self) -> List[tuple]:
        return [(t.label, t.box) for t in self._tracker.tracks if t.label in self.confirmed and self._confirmed_now(t)]

    def _confirmed_now(self, track: Track) -> bool:
        return track.label in self.confirmed
