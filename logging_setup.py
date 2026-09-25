"""
Structured logging for the analysis service.

One JSON object per line on stdout (what Docker, Coolify and log shippers
expect):

    {"time": "...", "level": "info", "logger": "proctor-ai", "msg": "session_closed",
     "session_id": "...", "frames": 412, ...}

Rules this module enforces, each the answer to a way logging goes wrong:

  * Logging never breaks the service. The formatter falls back to a plain line
    if a field cannot be serialised.
  * Every line inside a websocket session carries its session_id automatically
    (a context variable), so one candidate's whole story is one grep -- without
    each call site having to remember to pass it.
  * Secrets and media never appear. Sensitive-looking field names are redacted,
    and bytes (frames, audio, embeddings' raw payloads) are logged as sizes.
  * A fault that repeats every frame does not become a flood. Warnings and
    errors are rate limited per (logger, message, exception type); the next
    line that gets through says how many repeats were suppressed.
  * Health checks (polled every few seconds) stay out of the access log.
  * Uncaught exceptions -- in the main thread, worker threads and the event
    loop -- are logged as structured errors instead of raw tracebacks on stderr.

Configuration (environment):
    LOG_LEVEL    debug | info | warning | error   (default info)
    LOG_FORMAT   json | text                       (default json; text when run in a terminal)
"""

from __future__ import annotations

import contextvars
import datetime as _dt
import json
import logging
import os
import re
import sys
import threading
import time
import traceback
from typing import Any, Dict, Optional, Tuple

# Set by service.py for the duration of a websocket session.
session_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("session_id", default=None)

_SENSITIVE = re.compile(r"password|passwd|secret|authorization|cookie|token|api_?key|embedding", re.I)
_RESERVED = {"time", "level", "logger", "msg", "session_id"}
_STANDARD_ATTRS = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime", "fields"}
_MAX_STR = 2000
_MAX_DEPTH = 5
_MAX_ITEMS = 50


def sanitize(value: Any, depth: int = 0) -> Any:
    """Makes a value JSON-safe, bounded, and free of secrets and raw media."""
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if value == value and value not in (float("inf"), float("-inf")) else str(value)
    if isinstance(value, str):
        return value if len(value) <= _MAX_STR else f"{value[:_MAX_STR]}...[+{len(value) - _MAX_STR} chars]"
    if isinstance(value, (bytes, bytearray, memoryview)):
        return f"[{len(value)} bytes]"
    if depth >= _MAX_DEPTH:
        return "[object]"
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for i, (k, v) in enumerate(value.items()):
            if i >= _MAX_ITEMS:
                out["[truncated]"] = f"{len(value) - _MAX_ITEMS} more keys"
                break
            key = str(k)
            out[key] = "[REDACTED]" if _SENSITIVE.search(key) and v is not None else sanitize(v, depth + 1)
        return out
    if isinstance(value, (list, tuple, set, frozenset)):
        items = list(value)
        out_list = [sanitize(v, depth + 1) for v in items[:_MAX_ITEMS]]
        if len(items) > _MAX_ITEMS:
            out_list.append(f"[+{len(items) - _MAX_ITEMS} more]")
        return out_list
    if isinstance(value, BaseException):
        return {"type": type(value).__name__, "message": sanitize(str(value))}
    try:
        return sanitize(str(value), depth + 1)
    except Exception:  # noqa: BLE001 - logging must never raise
        return "[unprintable]"


def _error_field(exc_info: Tuple[Any, Any, Any]) -> Dict[str, Any]:
    exc_type, exc, tb = exc_info
    return {
        "type": getattr(exc_type, "__name__", str(exc_type)),
        "message": sanitize(str(exc)),
        "stack": sanitize("".join(traceback.format_exception(exc_type, exc, tb)), 0),
    }


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        try:
            payload: Dict[str, Any] = {
                "time": _dt.datetime.fromtimestamp(record.created, _dt.timezone.utc).isoformat(timespec="milliseconds"),
                "level": record.levelname.lower().replace("warning", "warn").replace("critical", "fatal"),
                "logger": record.name,
                "msg": record.getMessage(),
            }
            sid = getattr(record, "session_id", None) or session_id_var.get()
            if sid:
                payload["session_id"] = sid
            fields = getattr(record, "fields", None)
            if isinstance(fields, dict):
                for key, val in sanitize(fields).items():
                    payload[key if key not in _RESERVED else f"field_{key}"] = val
            # Anything passed through logging's own `extra=` (not `fields`).
            for key, val in record.__dict__.items():
                if key not in _STANDARD_ATTRS and key not in payload and not key.startswith("_"):
                    payload[key] = sanitize(val)
            if record.exc_info and record.exc_info[0] is not None:
                payload["error"] = _error_field(record.exc_info)
            return json.dumps(payload, ensure_ascii=False, default=str)
        except Exception as exc:  # noqa: BLE001 - last resort, never raise from logging
            return json.dumps({"level": "error", "msg": "log_format_failed", "reason": str(exc), "original": str(record.msg)[:200]})


class TextFormatter(logging.Formatter):
    """Readable single lines for a terminal."""

    def format(self, record: logging.LogRecord) -> str:
        base = f"{time.strftime('%H:%M:%S', time.localtime(record.created))} {record.levelname:<7} {record.name}: {record.getMessage()}"
        sid = getattr(record, "session_id", None) or session_id_var.get()
        if sid:
            base += f" session_id={sid}"
        fields = getattr(record, "fields", None)
        if isinstance(fields, dict) and fields:
            base += " " + " ".join(f"{k}={v}" for k, v in sanitize(fields).items())
        if record.exc_info and record.exc_info[0] is not None:
            base += "\n" + "".join(traceback.format_exception(*record.exc_info)).rstrip()
        return base


class RateLimitFilter(logging.Filter):
    """Suppresses a warning/error that repeats within `window` seconds.

    The key is (logger, message template, exception type), so "frame analysis
    failed" for a thousand frames is one line plus a count -- while a *different*
    problem still gets through immediately."""

    def __init__(self, window: float = 30.0, clock=time.monotonic) -> None:
        super().__init__()
        self.window = window
        self._clock = clock
        self._seen: Dict[Tuple[str, str, str], Tuple[float, int]] = {}
        self._lock = threading.Lock()

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno < logging.WARNING:
            return True
        exc_name = record.exc_info[0].__name__ if record.exc_info and record.exc_info[0] else ""
        key = (record.name, str(record.msg), exc_name)
        now = self._clock()
        with self._lock:
            last, suppressed = self._seen.get(key, (None, 0))
            if last is not None and now - last < self.window:
                self._seen[key] = (last, suppressed + 1)
                return False
            self._seen[key] = (now, 0)
            if len(self._seen) > 500:  # bound memory
                oldest = min(self._seen, key=lambda k: self._seen[k][0])
                del self._seen[oldest]
        if suppressed:
            fields = dict(getattr(record, "fields", None) or {})
            fields["suppressed_repeats"] = suppressed
            record.fields = fields
        return True


class HealthAccessFilter(logging.Filter):
    """Drops uvicorn's access line for /health: it is polled constantly."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            args = record.args
            return not (isinstance(args, tuple) and len(args) >= 3 and str(args[2]).split("?")[0] == "/health")
        except Exception:  # noqa: BLE001
            return True


def log_event(logger: logging.Logger, level: int, event: str, /, *, exc_info: Any = None, **fields: Any) -> None:
    """The one way to log: a constant event name plus structured fields."""
    logger.log(level, event, exc_info=exc_info, extra={"fields": fields})


_configured = False


def configure_logging(level: Optional[str] = None, fmt: Optional[str] = None, stream=None, force: bool = False) -> None:
    """Idempotent. Routes root, uvicorn and application logs through one handler."""
    global _configured
    if _configured and not force:
        return
    _configured = True

    level_name = (level or os.environ.get("LOG_LEVEL", "info")).upper()
    numeric = logging.getLevelName(level_name)
    if not isinstance(numeric, int):
        numeric = logging.INFO
    out = stream or sys.stdout
    chosen = (fmt or os.environ.get("LOG_FORMAT") or ("text" if getattr(out, "isatty", lambda: False)() else "json")).lower()

    handler = logging.StreamHandler(out)
    handler.setFormatter(TextFormatter() if chosen == "text" else JsonFormatter())
    handler.addFilter(RateLimitFilter())

    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(handler)
    root.setLevel(numeric)

    # uvicorn installs its own handlers; take them over so its lines are JSON too.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"):
        lg = logging.getLogger(name)
        lg.handlers.clear()
        lg.propagate = True
        lg.setLevel(numeric)
    logging.getLogger("uvicorn.access").addFilter(HealthAccessFilter())

    # Chatty native/ML libraries: warnings and worse only.
    for noisy in ("absl", "matplotlib", "PIL", "asyncio", "websockets", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(max(numeric, logging.WARNING))

    install_exception_hooks()


def install_exception_hooks() -> None:
    """Uncaught exceptions become structured errors instead of stderr tracebacks."""
    log = logging.getLogger("proctor-ai")

    def _excepthook(exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        log.critical("uncaught_exception", exc_info=(exc_type, exc, tb))

    def _threadhook(args):
        log.error("uncaught_thread_exception", exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
                  extra={"fields": {"thread": getattr(args.thread, "name", None)}})

    sys.excepthook = _excepthook
    threading.excepthook = _threadhook
