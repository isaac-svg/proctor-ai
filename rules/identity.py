"""
Identity: is the person at the desk the person who checked in, and are they
still the same person they were a few minutes ago?

Two independent checks, because each catches what the other cannot:

  1. Against the check-in reference (IDENTITY_MISMATCH). The enrolled face
     against the face now. Catches a stand-in who was there from the start of
     the exam. It compares two different sessions of the camera and light, so
     its threshold has to be forgiving.

  2. Against *this session's own history* (PERSON_CHANGED). A running centroid
     of the candidate's face, built from the exam so far. Same camera, same room,
     same lighting: the same person scores 0.5-0.9 here, so a score under
     `continuity_change_below` is a genuinely different face -- a far stricter
     test than (1), and one that needs no enrolment at all. Catches the swap
     mid-exam, and the classic "leaves the frame, someone else sits down": right
     after an absence a shorter run of evidence is enough.

When the two disagree the rule is careful, because "you are not who you say you
are" is the most serious thing this system can say:

  * Doesn't match the check-in photo, but matches the person the session has been
    tracking (and that person *was* verified against the photo earlier): the same
    candidate looks different -- glasses, a mask, a hat, the light. That is
    APPEARANCE_CHANGED for a human to look at, not an accusation.
  * Every check needs a run of consecutive low scores, and any good score resets it.
  * Frames the perception layer judged unusable (turned away, small, blurry)
    carry no score, so they neither confirm nor refute anything.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

from observations import AlertEvent, FrameObservation
from pipeline_config import PipelineConfig
from rules.episodes import AlertGate


def _norm(v: Sequence[float]) -> List[float]:
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n > 0 else list(v)


def _cos(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(_norm(a), _norm(b)))


def _mean(vectors: Sequence[Sequence[float]]) -> List[float]:
    return _norm([sum(col) / len(vectors) for col in zip(*vectors)])


# Cosine similarity above which a face is "the same person" for the purpose of
# saying the tracked person was verified against the check-in reference
# (OpenCV's recommended SFace cut-off).
_VERIFIED_SIMILARITY = 0.363
_VERIFIED_HITS = 3
_APPEARANCE_COOLDOWN_S = 600.0


class IdentityRule:
    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        self._gate = AlertGate()
        self.last_similarity: Optional[float] = None
        self.last_continuity: Optional[float] = None

        # Against the reference.
        self._ref_run = 0
        self._ref_scores: List[float] = []

        # Against the session's own history.
        self._boot: List[List[float]] = []
        self._centroid: Optional[List[float]] = None
        self._verify_hits = 0
        self._break_run = 0
        self._break_scores: List[float] = []
        self._recheck_until = float("-inf")
        self._absence_seconds: Optional[float] = None

    # ------------------------------------------------------------------
    @property
    def tracking(self) -> bool:
        """Has the session built up a face to compare against?"""
        return self._centroid is not None

    @property
    def tracked_person_verified(self) -> bool:
        return self._centroid is not None and self._verify_hits >= _VERIFIED_HITS

    def on_face_returned(self, ts: float, absent_seconds: float) -> None:
        """The candidate is back in frame after being out of it. Whoever it is now
        is checked against who left, with a shorter run of evidence."""
        self._recheck_until = ts + self.cfg.continuity_after_absence_window_s
        self._absence_seconds = absent_seconds

    # ------------------------------------------------------------------
    def _continuity(self, emb: Sequence[float], ref_sim: Optional[float]) -> Optional[float]:
        """Updates the session's face model, and returns how well `emb` matches it
        (None while it is still being built)."""
        cfg = self.cfg
        if self._centroid is None:
            self._boot.append(list(emb))
            if len(self._boot) >= cfg.continuity_bootstrap:
                mean = _mean(self._boot)
                keep = [e for e in self._boot if _cos(e, mean) >= cfg.continuity_same_above]
                # Only trust the start of a model made of faces that agree with one another;
                # otherwise drop the oldest and keep listening.
                if len(keep) >= cfg.continuity_bootstrap - 1:
                    self._centroid = _mean(keep)
                    self._boot.clear()
                else:
                    self._boot = self._boot[1:]
            return None

        similarity = _cos(emb, self._centroid)
        if similarity >= cfg.continuity_same_above:
            a = cfg.continuity_update_alpha
            self._centroid = _norm([(1 - a) * c + a * e for c, e in zip(self._centroid, _norm(emb))])
            if ref_sim is not None and ref_sim >= _VERIFIED_SIMILARITY:
                self._verify_hits += 1
        return similarity

    def _reset_tracking(self) -> None:
        self._centroid = None
        self._boot.clear()
        self._verify_hits = 0
        self._break_run = 0
        self._break_scores.clear()

    # ------------------------------------------------------------------
    def on_frame(self, obs: FrameObservation) -> List[AlertEvent]:
        cfg = self.cfg
        ref_sim = obs.identity_similarity
        emb = obs.identity_embedding
        if ref_sim is None and emb is None:
            return []  # nothing scored (no reference and no usable face): neither confirms nor refutes

        if ref_sim is not None:
            self.last_similarity = ref_sim
            if ref_sim >= cfg.identity_mismatch_below:
                self._ref_run = 0
                self._ref_scores.clear()
            else:
                self._ref_run += 1
                self._ref_scores.append(ref_sim)

        cont: Optional[float] = None
        if emb is not None:
            cont = self._continuity(emb, ref_sim)
            self.last_continuity = cont
            if cont is not None:
                if cont < cfg.continuity_change_below:
                    self._break_run += 1
                    self._break_scores.append(cont)
                else:
                    self._break_run = 0
                    self._break_scores.clear()

        after_absence = obs.ts <= self._recheck_until
        needed = cfg.continuity_after_absence_consecutive if after_absence else cfg.continuity_consecutive
        ref_trigger = self._ref_run >= cfg.identity_mismatch_consecutive
        cont_trigger = self._break_run >= needed
        if not (ref_trigger or cont_trigger):
            return []
        return self._raise(obs, ref_trigger, cont_trigger, cont, after_absence)

    # ------------------------------------------------------------------
    def _raise(self, obs: FrameObservation, ref_trigger: bool, cont_trigger: bool, cont: Optional[float], after_absence: bool) -> List[AlertEvent]:
        cfg = self.cfg
        ref_scores, self._ref_scores, self._ref_run = self._ref_scores, [], 0
        break_scores, self._break_scores, self._break_run = self._break_scores, [], 0
        was_verified = self.tracked_person_verified

        if ref_trigger:
            # Not the person in the check-in photo. But if the session has been following one face all
            # along, that face was verified against the photo earlier, and it is still the face on
            # screen, the candidate is the same person who merely looks different now.
            same_as_tracked = was_verified and not cont_trigger and cont is not None and cont >= cfg.continuity_same_above
            if same_as_tracked:
                if not self._gate.allow("appearance", obs.ts, _APPEARANCE_COOLDOWN_S):
                    return []
                return [
                    AlertEvent(
                        alert_type="APPEARANCE_CHANGED",
                        severity="MEDIUM",
                        ts=obs.ts,
                        description="The candidate no longer looks like their check-in photo, but is the same person who has been at the desk. Their appearance may have changed (glasses, a mask, lighting).",
                        details={
                            "similarity_to_checkin_mean": round(sum(ref_scores) / len(ref_scores), 3),
                            "continuity_with_session": round(cont, 3),
                            "threshold": cfg.identity_mismatch_below,
                        },
                        evidence=True,
                    )
                ]
            if not self._gate.allow("identity", obs.ts, cfg.identity_cooldown_s):
                return []
            self._gate.allow("person_change", obs.ts, cfg.person_change_cooldown_s)  # one alert for one swap
            changed_during_exam = cont_trigger
            self._reset_tracking()
            details = {
                "similarity_mean": round(sum(ref_scores) / len(ref_scores), 3),
                "similarity_min": round(min(ref_scores), 3),
                "checks": len(ref_scores),
                "threshold": cfg.identity_mismatch_below,
                "changed_during_exam": changed_during_exam,
            }
            if break_scores:
                details["continuity_mean"] = round(sum(break_scores) / len(break_scores), 3)
            if changed_during_exam and after_absence and self._absence_seconds is not None:
                details["after_absence_seconds"] = round(self._absence_seconds, 1)
            return [
                AlertEvent(
                    alert_type="IDENTITY_MISMATCH",
                    severity="CRITICAL",
                    ts=obs.ts,
                    description="The person on camera does not match the candidate who checked in.",
                    details=details,
                    # No `confidence`: a cosine similarity is not a probability, and inventing one for an
                    # accusation this serious would be false precision. The raw scores are in `details`.
                    evidence=True,
                )
            ]

        # The check-in photo still matches (or there was none), but the face changed mid-session.
        if not self._gate.allow("person_change", obs.ts, cfg.person_change_cooldown_s):
            return []
        self._reset_tracking()
        details = {
            "continuity_mean": round(sum(break_scores) / len(break_scores), 3),
            "continuity_min": round(min(break_scores), 3),
            "checks": len(break_scores),
            "threshold": cfg.continuity_change_below,
            "matches_checkin": self.last_similarity is not None and self.last_similarity >= cfg.identity_mismatch_below,
        }
        if after_absence and self._absence_seconds is not None:
            details["after_absence_seconds"] = round(self._absence_seconds, 1)
        return [
            AlertEvent(
                alert_type="PERSON_CHANGED",
                severity="HIGH",
                ts=obs.ts,
                description="A different person appears to be at the desk than earlier in the exam.",
                details=details,
                evidence=True,
            )
        ]
