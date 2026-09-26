"""Structured logging: format, redaction, rate limiting, health filter, hooks.

Pure standard library -- runs anywhere (including the slim CI job)."""

import io
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging_setup
from logging_setup import HealthAccessFilter, JsonFormatter, RateLimitFilter, log_event, sanitize, session_id_var


def _logger(stream=None, filters=()):
    stream = stream or io.StringIO()
    lg = logging.getLogger(f"test.{id(stream)}")
    lg.handlers.clear()
    lg.propagate = False
    lg.setLevel(logging.DEBUG)
    h = logging.StreamHandler(stream)
    h.setFormatter(JsonFormatter())
    for f in filters:
        h.addFilter(f)
    lg.addHandler(h)
    return lg, stream


def _lines(stream):
    return [json.loads(l) for l in stream.getvalue().splitlines() if l]


def test_one_json_object_per_line_with_core_fields():
    lg, out = _logger()
    log_event(lg, logging.INFO, "session_connected", client="10.0.0.1", frames=3)
    (line,) = _lines(out)
    assert line["msg"] == "session_connected"
    assert line["level"] == "info"
    assert line["client"] == "10.0.0.1" and line["frames"] == 3
    assert line["time"].endswith("+00:00")


def test_warning_and_critical_use_the_backends_level_names():
    lg, out = _logger()
    lg.warning("w")
    lg.critical("c")
    assert [l["level"] for l in _lines(out)] == ["warn", "fatal"]


def test_session_id_is_attached_automatically_inside_a_session():
    lg, out = _logger()
    token = session_id_var.set("abc-123")
    try:
        lg.info("inside")
    finally:
        session_id_var.reset(token)
    lg.info("outside")
    inside, outside = _lines(out)
    assert inside["session_id"] == "abc-123"
    assert "session_id" not in outside


def test_a_field_cannot_overwrite_the_core_keys():
    lg, out = _logger()
    log_event(lg, logging.INFO, "real", msg="fake", level="debug", time="never")
    (line,) = _lines(out)
    assert line["msg"] == "real" and line["level"] == "info" and line["time"] != "never"
    assert line["field_msg"] == "fake"


def test_secrets_and_embeddings_are_redacted_at_any_depth():
    lg, out = _logger()
    log_event(lg, logging.INFO, "x", body={"user": "a", "password": "hunter2", "n": [{"token": "eyJ.secret"}]}, embedding=[0.7391] * 128)
    raw = out.getvalue()
    for leaked in ("hunter2", "eyJ.secret", "0.7391"):
        assert leaked not in raw
    assert "[REDACTED]" in raw and '"user": "a"' in raw


def test_media_bytes_are_logged_as_sizes_never_content():
    assert sanitize(b"\xff\xd8" * 500) == "[1000 bytes]"
    lg, out = _logger()
    log_event(lg, logging.INFO, "frame", data=b"\xff" * 4096)
    assert "[4096 bytes]" in out.getvalue()


def test_exceptions_become_a_structured_error_field():
    lg, out = _logger()
    try:
        raise OSError("libEGL.so.1: cannot open shared object file")
    except OSError:
        lg.error("session_create_failed", exc_info=True)
    (line,) = _lines(out)
    assert line["error"]["type"] == "OSError"
    assert "libEGL.so.1" in line["error"]["message"]
    assert "Traceback" in line["error"]["stack"]


def test_unserialisable_and_hostile_values_never_raise():
    lg, out = _logger()

    class Evil:
        def __str__(self):
            raise RuntimeError("no")

    circular = {}
    circular["self"] = circular
    log_event(lg, logging.INFO, "x", evil=Evil(), circular=circular, nan=float("nan"), big="y" * 10_000)
    (line,) = _lines(out)
    assert line["evil"] == "[unprintable]" or isinstance(line["evil"], str)
    assert line["nan"] == "nan"
    assert len(line["big"]) < 2100


class TestRateLimit:
    def test_a_fault_that_repeats_every_frame_is_one_line_plus_a_count(self):
        now = [0.0]
        flt = RateLimitFilter(window=30, clock=lambda: now[0])
        lg, out = _logger(filters=[flt])
        for _ in range(100):
            lg.error("frame analysis failed")
            now[0] += 0.1  # 10 s in total, all inside the window
        assert len(_lines(out)) == 1
        now[0] += 60
        lg.error("frame analysis failed")
        first, second = _lines(out)
        assert second["suppressed_repeats"] == 99
        assert "suppressed_repeats" not in first

    def test_a_different_problem_still_gets_through_immediately(self):
        flt = RateLimitFilter(window=30, clock=lambda: 0.0)
        lg, out = _logger(filters=[flt])
        lg.error("frame analysis failed")
        lg.error("vad failed")
        assert len(_lines(out)) == 2

    def test_info_lines_are_never_suppressed(self):
        flt = RateLimitFilter(window=30, clock=lambda: 0.0)
        lg, out = _logger(filters=[flt])
        for _ in range(5):
            lg.info("alert_emitted")
        assert len(_lines(out)) == 5


def test_health_checks_stay_out_of_the_access_log():
    flt = HealthAccessFilter()

    def rec(path):
        return logging.LogRecord("uvicorn.access", logging.INFO, "", 0, '%s - "%s %s HTTP/%s" %d', ("1.2.3.4:1", "GET", path, "1.1", 200), None)

    assert flt.filter(rec("/health")) is False
    assert flt.filter(rec("/health?probe=1")) is False
    assert flt.filter(rec("/analyze/abc")) is True


def test_configure_logging_routes_uvicorn_through_the_same_handler_and_is_idempotent():
    stream = io.StringIO()
    logging_setup.configure_logging(level="info", fmt="json", stream=stream, force=True)
    logging.getLogger("uvicorn.error").info("Started server process")
    logging.getLogger("proctor-ai").warning("hello")
    lines = _lines(stream)
    assert [l["msg"] for l in lines] == ["Started server process", "hello"]
    logging_setup.configure_logging(level="error", fmt="json", stream=io.StringIO())  # no force: ignored
    assert logging.getLogger().level == logging.INFO


def test_uncaught_thread_exceptions_are_logged_not_printed():
    import threading

    stream = io.StringIO()
    logging_setup.configure_logging(level="info", fmt="json", stream=stream, force=True)

    def boom():
        raise ValueError("worker died")

    t = threading.Thread(target=boom, name="worker-1")
    t.start()
    t.join()
    line = next(l for l in _lines(stream) if l["msg"] == "uncaught_thread_exception")
    assert line["level"] == "error" and line["thread"] == "worker-1"
    assert line["error"]["message"] == "worker died"
