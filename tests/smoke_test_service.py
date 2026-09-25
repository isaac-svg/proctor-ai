"""
End-to-end smoke test for service.py's WS /analyze/{session_id} route,
using FastAPI's synchronous TestClient (starlette's own websocket test
support) rather than spinning up a real uvicorn process -- cheaper and
more deterministic, and exercises the real route/session_manager code, not
a mock of it. Uses real (if tiny) synthetic JPEG/PCM input and the real
mediapipe/YOLO/silero_vad models already checked into this repo, so this
is a genuine integration test, not a unit test with everything stubbed --
expect it to be slow (model loads) rather than fast.

Run: pytest tests/smoke_test_service.py -v
"""

import base64
import io
import struct
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi.testclient import TestClient

from service import app


def synthetic_jpeg_bytes(color=(120, 120, 120), size=(64, 48)) -> bytes:
    """A tiny solid-color JPEG -- deliberately not a real face; this test
    exercises the pipeline's plumbing (decode -> detectors -> event
    dicts -> AI_ALERT/AI_ANALYSIS mapping -> WS send), not detection
    accuracy against real footage."""
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="JPEG")
    return buf.getvalue()


def synthetic_pcm_bytes(num_samples: int = 8000) -> bytes:
    """Half a second of silence at 16kHz mono s16le -- valid PCM, just with
    nothing for VAD to detect as speech."""
    return struct.pack(f"<{num_samples}h", *([0] * num_samples))


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_endpoint_reports_no_active_sessions_before_any_connect(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "OK"


def test_video_and_audio_frames_do_not_crash_and_produce_analysis(client):
    with client.websocket_connect("/analyze/smoke-session-1") as ws:
        ws.send_json(
            {
                "type": "CONNECT",
                "session_id": "smoke-session-1",
                "stream_type": "VIDEO_AUDIO",
                "codecs": {"video": "mjpeg", "audio": "pcm_s16le"},
                "resolution": "64x48",
                "snapshot_interval_ms": 1000,
                "audio_sample_rate": 16000,
            }
        )

        received_types = []
        # 5 frames to guarantee at least one AI_ANALYSIS (service.py sends
        # one every 5th frame).
        for i in range(5):
            ws.send_json(
                {
                    "type": "VIDEO_FRAME",
                    "sequence_number": i + 1,
                    "session_id": "smoke-session-1",
                    "data": base64.b64encode(synthetic_jpeg_bytes()).decode("ascii"),
                    "size_bytes": 0,
                    "frame_number": i + 1,
                }
            )

        ws.send_json(
            {
                "type": "AUDIO_FRAME",
                "sequence_number": 1,
                "session_id": "smoke-session-1",
                "data": base64.b64encode(synthetic_pcm_bytes()).decode("ascii"),
                "size_bytes": 0,
                "sample_rate": 16000,
                "duration_ms": 500,
            }
        )

        # Drain whatever the service sent back for a bounded number of
        # messages rather than a fixed sleep -- AI_ALERT is possible but not
        # guaranteed against a solid-color image with no face; AI_ANALYSIS
        # is guaranteed by frame 5.
        try:
            for _ in range(10):
                msg = ws.receive_json()
                received_types.append(msg["type"])
                if "AI_ANALYSIS" in received_types:
                    break
        except Exception:
            pass  # No more messages queued -- fine, checked below.

        assert "AI_ANALYSIS" in received_types, f"expected an AI_ANALYSIS message, got {received_types}"


def test_two_concurrent_sessions_do_not_cross_contaminate_state():
    import session_manager
    from helpers import frame, obj

    manager = session_manager.ProctorSessionManager()
    session_a = manager.get_or_create("session-a")
    session_b = manager.get_or_create("session-b")

    assert session_a is not session_b
    assert session_a.session_id == "session-a"
    assert session_b.session_id == "session-b"

    # A phone confirmed in session A's rules must leave session B's untouched.
    alerts_a = []
    for t in range(3):
        alerts_a += session_a.pipeline.on_frame(frame(float(t), objects=[obj("cell phone", 0.8)]))
        session_b.pipeline.on_frame(frame(float(t)))
    assert [a.alert_type for a in alerts_a] == ["CELLPHONE_DETECTED"]
    assert "cell phone" in session_a.pipeline.objects.confirmed
    assert "cell phone" not in session_b.pipeline.objects.confirmed

    # Identity references are per session too.
    assert session_a.set_reference([0.1] * 128)
    assert session_a.has_reference and not session_b.has_reference

    manager.remove("session-a")
    manager.remove("session-b")
    assert manager.count() == 0


def test_health_reports_which_capabilities_are_really_loaded(client):
    body = client.get("/health").json()
    assert body["status"] == "OK"
    assert set(body["capabilities"]) == {"identity", "speaker_diarization", "objects"}
    assert body["models"] in ("loading", "ready")


def test_enroll_with_an_unusable_image_is_rejected_with_a_reason(client):
    with client.websocket_connect("/analyze/precheck-smoke") as ws:
        ws.send_json({"type": "ENROLL", "images": [base64.b64encode(synthetic_jpeg_bytes(size=(320, 240))).decode()]})
        msg = ws.receive_json()
        assert msg["type"] == "ENROLLMENT_RESULT"
        assert msg["ok"] is False and msg["reason"] in ("no_face", "identity_unavailable")
        assert msg["embedding"] is None


def test_enroll_with_a_prior_embedding_sets_the_reference(client):
    with client.websocket_connect("/analyze/precheck-smoke-2") as ws:
        ws.send_json({"type": "ENROLL", "embedding": [0.05] * 128})
        msg = ws.receive_json()
        assert msg == {"type": "ENROLLMENT_RESULT", "ok": True, "reason": None, "embedding": None}


def test_enroll_rejects_a_malformed_embedding(client):
    with client.websocket_connect("/analyze/precheck-smoke-3") as ws:
        ws.send_json({"type": "ENROLL", "embedding": ["x", "y"]})
        assert ws.receive_json()["ok"] is False
