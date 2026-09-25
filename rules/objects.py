"""
Prohibited objects (phones, books, second laptops).

YOLO at low confidence produces the occasional one-frame ghost, so a label
must be *confirmed* -- seen in `object_confirm_hits` of the last
`object_confirm_window` frames -- before it's reported. Alert types stay
CELLPHONE_DETECTED for phones (existing policies key on it) and
PROHIBITED_OBJECT_DETECTED for the rest.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Optional, Set

from observations import AlertEvent, FrameObservation
from pipeline_config import PipelineConfig
from rules.episodes import AlertGate


class ProhibitedObjectRule:
    def __init__(self, cfg: PipelineConfig) -> None:
        self.cfg = cfg
        self._history: Dict[str, Deque[bool]] = {
            label: deque(maxlen=cfg.object_confirm_window) for label in cfg.object_rules
        }
        self._gate = AlertGate()
        self._phone_since: Optional[float] = None
        self._phone_escalated = False
        self.confirmed: Set[str] = set()

    def on_frame(self, obs: FrameObservation) -> List[AlertEvent]:
        events: List[AlertEvent] = []

        seen: Dict[str, float] = {}
        for o in obs.objects:
            rule = self.cfg.object_rules.get(o.label)
            if rule is None or o.confidence < rule[0]:
                continue
            seen[o.label] = max(seen.get(o.label, 0.0), o.confidence)

        confirmed_now: Set[str] = set()
        for label, history in self._history.items():
            history.append(label in seen)
            if sum(history) >= self.cfg.object_confirm_hits:
                confirmed_now.add(label)
        self.confirmed = confirmed_now

        for label in confirmed_now:
            _, severity = self.cfg.object_rules[label]
            is_phone = label == "cell phone"
            if not self._gate.allow(f"object:{label}", obs.ts, self.cfg.object_cooldown_s):
                continue
            events.append(
                AlertEvent(
                    alert_type="CELLPHONE_DETECTED" if is_phone else "PROHIBITED_OBJECT_DETECTED",
                    severity=severity,
                    ts=obs.ts,
                    description=(
                        "A cellphone was detected in frame."
                        if is_phone
                        else f"A {label} was detected in frame."
                    ),
                    details={"object": label},
                    confidence=round(seen.get(label, 0.0), 2),
                    evidence=True,
                )
            )

        # A phone that *stays* in view is worse than one that merely appeared.
        if "cell phone" in confirmed_now:
            if self._phone_since is None:
                self._phone_since = obs.ts
            elif not self._phone_escalated and obs.ts - self._phone_since >= self.cfg.phone_sustained_s:
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
                        },
                        evidence=True,
                    )
                )
        else:
            self._phone_since = None
            self._phone_escalated = False

        return events
