"""Service-level behaviour: honest /health, and a session that cannot be created
(the "libEGL.so.1: cannot open shared object file" outage) failing cleanly and loudly.

Needs the service's real dependencies; skipped in the slim CI job."""

import io
import json
import logging
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("mediapipe")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging_setup
import service
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect


@pytest.fixture()
def logs():
    stream = io.StringIO()
    logging_setup.configure_logging(level="debug", fmt="json", stream=stream, force=True)
    yield lambda: [json.loads(l) for l in stream.getvalue().splitlines() if l]


@pytest.fixture()
def client():
    return TestClient(service.app)


@pytest.fixture()
def shared_state():
    """Restores the module-level state the tests below manipulate."""
    shared = service.manager.shared
    saved = (dict(shared.errors), service._warm["state"])
    yield shared
    shared.errors.clear()
    shared.errors.update(saved[0])
    service._warm["state"] = saved[1]


def test_health_is_503_FAILED_when_a_critical_capability_is_missing(client, shared_state):
    service._warm["state"] = "ready"
    shared_state.errors["face_tracking"] = "OSError: libEGL.so.1: cannot open shared object file"
    response = client.get("/health")
    body = response.json()
    assert response.status_code == 503
    assert body["status"] == "FAILED"
    assert body["capabilities"]["face_tracking"] is False
    assert "libEGL.so.1" in body["errors"]["face_tracking"]


def test_health_stays_200_while_models_are_still_loading(client, shared_state):
    service._warm["state"] = "loading"
    shared_state.errors["face_tracking"] = "not checked yet would not be here; simulate a slow start"
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "LOADING"


def test_health_is_DEGRADED_but_200_when_only_an_optional_capability_is_missing(client, shared_state):
    service._warm["state"] = "ready"
    shared_state.errors.pop("face_tracking", None)
    shared_state.errors["speaker_diarization"] = "onnxruntime or model unavailable"
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "DEGRADED"


def test_health_reports_counters(client):
    body = client.get("/health").json()
    assert set(body["metrics"]) == {"sessions_total", "sessions_failed", "frames_processed", "alerts_emitted"}
    assert body["uptime_seconds"] >= 0


def test_a_session_that_cannot_be_created_is_closed_1011_with_one_clear_error(client, logs, monkeypatch):
    def broken(_session_id):
        raise OSError("libEGL.so.1: cannot open shared object file: No such file or directory")

    monkeypatch.setattr(service.manager, "get_or_create", broken)
    before = service._metrics["sessions_failed"]

    with client.websocket_connect("/analyze/sess-egl-1") as ws:
        with pytest.raises(WebSocketDisconnect) as closed:
            ws.receive_text()
    assert closed.value.code == 1011
    assert "analysis unavailable" in closed.value.reason

    lines = logs()
    failed = [l for l in lines if l["msg"] == "session_create_failed"]
    assert len(failed) == 1
    assert failed[0]["level"] == "error"
    assert failed[0]["session_id"] == "sess-egl-1"
    assert "libEGL.so.1" in failed[0]["error"]["message"]
    # ...and the session summary still says why it ended.
    summary = [l for l in lines if l["msg"] == "session_closed"][0]
    assert summary["reason"] == "session_create_failed" and summary["session_id"] == "sess-egl-1"
    assert service._metrics["sessions_failed"] == before + 1


def test_a_normal_session_logs_connect_and_a_closing_summary(client, logs):
    with client.websocket_connect("/analyze/sess-ok-1") as ws:
        ws.send_json({"type": "CONNECT"})
        ws.send_json({"type": "DISCONNECT"})
    lines = [l for l in logs() if l.get("session_id") == "sess-ok-1"]
    assert [l["msg"] for l in lines][:1] == ["session_connected"]
    closed = [l for l in lines if l["msg"] == "session_closed"][0]
    assert closed["reason"] == "client_requested"
    assert closed["frames"] == 0 and closed["alerts"] == {}
    assert isinstance(closed["duration_seconds"], float)


def test_the_health_endpoint_is_not_written_to_the_access_log(client, logs):
    client.get("/health")
    assert not [l for l in logs() if "/health" in l.get("msg", "")]
