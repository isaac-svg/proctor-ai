"""
Identity consistency: is the person at the desk the person who checked in?

This is a *consistency* check against a reference captured at check-in (the
design choice for this deployment), so it catches a swap or a stand-in taking
over mid-exam. It does not prove the checked-in person is who they claim to
be -- that needs an independent reference photo.

Guarded hard against false accusations, because "you are not who you say you
are" is the most serious thing this system can say:
  * the perception layer only scores frames where the face is frontal, large
    and sharp enough (otherwise `identity_similarity` is None),
  * a *run* of consecutive low scores is required, and any good score resets
    it,
  * the threshold for accusing (0.30) sits below the "same person" cut-off
    (0.363), so ordinary lighting/pose variation lands in the no-alert band.
"""

from __future__ import annotations

from typing import List, Optional

from observations import AlertEvent, FrameObservation
from pipeline_config import PipelineConfig
from rules.episodes import AlertGate


class IdentityRule:
    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        self._low_run = 0
        self._scores: List[float] = []
        self._gate = AlertGate()
        self.last_similarity: Optional[float] = None

    def on_frame(self, obs: FrameObservation) -> List[AlertEvent]:
        sim = obs.identity_similarity
        if sim is None:
            return []  # nothing scored (no reference, or unusable frame): neither confirms nor refutes
        self.last_similarity = sim

        if sim >= self.cfg.identity_mismatch_below:
            self._low_run = 0
            self._scores.clear()
            return []

        self._low_run += 1
        self._scores.append(sim)
        if self._low_run < self.cfg.identity_mismatch_consecutive:
            return []

        if not self._gate.allow("identity", obs.ts, self.cfg.identity_cooldown_s):
            return []
        scores, self._scores, self._low_run = self._scores, [], 0
        return [
            AlertEvent(
                alert_type="IDENTITY_MISMATCH",
                severity="CRITICAL",
                ts=obs.ts,
                description="The person on camera does not match the candidate who checked in.",
                details={
                    "similarity_mean": round(sum(scores) / len(scores), 3),
                    "similarity_min": round(min(scores), 3),
                    "checks": len(scores),
                    "threshold": self.cfg.identity_mismatch_below,
                },
                # No `confidence`: a cosine similarity is not a probability,
                # and inventing one for an accusation this serious would be
                # false precision. The raw scores are in `details`.
                evidence=True,
            )
        ]
