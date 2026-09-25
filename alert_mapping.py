"""
Maps events.py's EventDetector event dicts onto the AI_ALERT wire shape
shepherd-backend's video-stream-relay.ts / shepherd-ai's stream-client.ts
expect (API_CONTRACTS.md §3), including a severity table that also becomes
each occurrence's evidence_events.severity once shepherd-ai turns it back
into a SECURITY_EVENT (see shepherd-ai/docs/AI_INTEGRATION.md and
shepherd-backend/src/db.ts's seeded exam policy for where these alert_type
strings need matching event_behaviors entries).

Two proctor-ai event types collapse onto the same alert_type at different
severities (a phone merely appearing vs. staying up for a while) -- severity
is a property of the *occurrence*, not a hardcoded-per-type constant, same
as every other detector in this system already works.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Tuple

if TYPE_CHECKING:
    from observations import AlertEvent

# event["type"] -> (alert_type, severity)
ALERT_MAP: Dict[str, Tuple[str, str]] = {
    "cellphone_detected":          ("CELLPHONE_DETECTED", "HIGH"),
    "sustained_cellphone_usage":   ("CELLPHONE_DETECTED", "CRITICAL"),
    "sustained_look_away":         ("SUSTAINED_LOOK_AWAY", "MEDIUM"),
    "frequent_look_away":          ("FREQUENT_LOOK_AWAY", "HIGH"),
    "gaze_without_head_movement":  ("GAZE_AWAY", "LOW"),
    "voice_detected":              ("VOICE_DETECTED", "LOW"),
    "sustained_speaking":          ("SUSTAINED_SPEAKING", "MEDIUM"),
}

# Human-readable description per alert_type, independent of severity.
_DESCRIPTIONS: Dict[str, str] = {
    "CELLPHONE_DETECTED": "A cellphone was detected in frame.",
    "SUSTAINED_LOOK_AWAY": "Student looked away from the screen for a sustained period.",
    "FREQUENT_LOOK_AWAY": "Student has looked away from the screen repeatedly.",
    "GAZE_AWAY": "Eyes moved away from center without a corresponding head movement.",
    "VOICE_DETECTED": "Voice activity detected on the microphone.",
    "SUSTAINED_SPEAKING": "Sustained speech detected on the microphone.",
}


def map_event(ev: Dict[str, Any]) -> Dict[str, Any] | None:
    """
    Maps one events.py event dict to an AI_ALERT body (everything but the
    envelope's type/session_id/timestamp, which service.py adds). Returns
    None for an event type this table doesn't recognize, rather than
    raising -- a forward-compatible EventDetector plugin (register_sub_detector)
    producing an unmapped type shouldn't crash the service, just go
    unrelayed until this table is extended.
    """
    mapping = ALERT_MAP.get(ev.get("type"))
    if mapping is None:
        return None
    alert_type, severity = mapping
    details = {k: v for k, v in ev.items() if k not in ("type", "ts")}
    return {
        "alert_type": alert_type,
        "severity": severity,
        "confidence": None,
        "description": _DESCRIPTIONS.get(alert_type, alert_type.replace("_", " ").title()),
        "details": details,
    }


# ---------------------------------------------------------------------------
# Rules-layer events (pipeline.py). Unlike map_event() above, AlertEvent
# already carries its own alert_type, severity and description -- severity is
# decided per occurrence by the rule, not looked up from a table -- so this
# is just serialisation.
# ---------------------------------------------------------------------------


def alert_to_message(session_id: str, ev: "AlertEvent", clip: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """The AI_ALERT wire message (API_CONTRACTS.md §3) for one AlertEvent.
    `clip`, when present, is the evidence attachment (evidence.py); the
    backend's video-stream relay peels it off and stores it, so it never
    travels on to the desktop app."""
    msg: Dict[str, Any] = {
        "type": "AI_ALERT",
        "session_id": session_id,
        "timestamp": int(ev.ts),
        "alert_type": ev.alert_type,
        "severity": ev.severity,
        "confidence": ev.confidence,
        "description": ev.description,
        "details": dict(ev.details),
    }
    if clip is not None:
        msg["evidence"] = clip
    return msg
