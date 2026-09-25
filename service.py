"""
FastAPI/websockets service exposing one endpoint per exam session:
WS /analyze/{session_id}. shepherd-backend's video-stream-relay.ts is the
only expected client -- it authenticates the real Electron connection and
proxies messages through to here, then relays whatever this service sends
back (AI_ANALYSIS / AI_ALERT / FRAME_ACK / ENROLLMENT_RESULT) to Electron.
See shepherd-ai's docs/AI_INTEGRATION.md for the full pipeline this is one
piece of, and README.md (this repo) for setup.

Messages from the client:
  CONNECT        handshake (no reply)
  ENROLL         {images: [b64 jpeg, ...]} or {embedding: [floats]} -- set the
                 identity reference for this session. Replies ENROLLMENT_RESULT.
  VIDEO_FRAME    one base64 JPEG snapshot
  AUDIO_FRAME    base64 raw PCM (s16le, mono, 16 kHz)
  DISCONNECT

Run: uvicorn service:app --port 8901
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import threading
import time
from collections import Counter
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List

from fastapi import FastAPI, Response, WebSocket, WebSocketDisconnect

# First, so everything below (including library import-time output) is captured
# by the structured handler. Quiet the native ML libraries' own stderr chatter.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "2")
from logging_setup import configure_logging, log_event, session_id_var  # noqa: E402

configure_logging()

from alert_mapping import alert_to_message
from session_manager import Emission, ProctorSessionManager, SharedModels

log = logging.getLogger("proctor-ai")

manager = ProctorSessionManager(SharedModels())

# The contract only says AI_ANALYSIS is "periodic" -- every Nth video frame
# keeps the informational stream light without making it useless.
_ANALYSIS_EVERY_N_FRAMES = 5
_FRAME_ACK_EVERY_N_FRAMES = 10
# Watchdog cadence: how often a session checks whether its feeds went quiet.
_TICK_SECONDS = 2.0
# Refuse absurd payloads instead of decoding them.
_MAX_IMAGE_BYTES = 2 * 1024 * 1024
_MAX_ENROLL_IMAGES = 5

_warm = {"state": "loading"}
_STARTED_AT = time.time()
# A frame slower than this is logged (rate limited): it means the box is
# overloaded and the candidate's analysis is falling behind real time.
_SLOW_FRAME_SECONDS = float(os.environ.get("SLOW_FRAME_SECONDS", "1.5"))
# Without these the service cannot do its job; /health then answers 503 so the
# orchestrator shows it as unhealthy instead of "OK" while every session fails.
_CRITICAL_CAPABILITIES = ("face_tracking",)

# Process-wide counters, exposed on /health. Only touched from the event loop.
_metrics = {"sessions_total": 0, "sessions_failed": 0, "frames_processed": 0, "alerts_emitted": 0}


@dataclass
class _SessionStats:
    started: float = field(default_factory=time.monotonic)
    frames: int = 0
    audio_chunks: int = 0
    invalid_messages: int = 0
    slow_frames: int = 0
    alerts: Counter = field(default_factory=Counter)


def _warm_up() -> None:
    started = time.monotonic()
    log_event(log, logging.INFO, "warm_up_started")
    try:
        manager.shared.warm_up()
    except Exception:  # noqa: BLE001 - the service must stay up to report what failed
        log_event(log, logging.ERROR, "warm_up_failed", exc_info=True)
    finally:
        _warm["state"] = "ready"
        missing = [c for c in _CRITICAL_CAPABILITIES if not manager.shared.capabilities().get(c)]
        log_event(
            log,
            logging.ERROR if missing else logging.INFO,
            "warm_up_finished",
            seconds=round(time.monotonic() - started, 1),
            capabilities=manager.shared.capabilities(),
            unavailable=dict(manager.shared.errors),
            missing_critical=missing,
        )


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Loading models (and, the first time, downloading them) takes long enough
    # that doing it on the first real frame would stall that exam's video.
    log_event(log, logging.INFO, "service_starting", pid=os.getpid(), log_level=logging.getLevelName(logging.getLogger().level))
    threading.Thread(target=_warm_up, name="proctor-ai-warmup", daemon=True).start()
    yield
    log_event(log, logging.INFO, "service_stopping", uptime_seconds=int(time.time() - _STARTED_AT), **_metrics)


app = FastAPI(title="proctor-ai", lifespan=lifespan)


@app.get("/health")
def health(response: Response) -> Dict[str, Any]:
    """Liveness plus honest readiness.

    `status` is OK when the service can do its job, LOADING while models are
    still loading (HTTP 200: it is starting, not broken), DEGRADED when an
    optional capability is missing (HTTP 200: still useful, and `unavailable`
    says what), and FAILED with HTTP 503 when a capability the service cannot
    work without is missing -- so the platform reports it unhealthy instead of
    "OK" while every candidate connection fails."""
    capabilities = manager.shared.capabilities()
    ready = _warm["state"] == "ready"
    missing_critical = [c for c in _CRITICAL_CAPABILITIES if not capabilities.get(c)]
    if ready and missing_critical:
        status = "FAILED"
        response.status_code = 503
    elif not ready:
        status = "LOADING"
    elif manager.shared.errors:
        status = "DEGRADED"
    else:
        status = "OK"
    return {
        "status": status,
        "active_sessions": manager.count(),
        "models": _warm["state"],
        "capabilities": capabilities,
        "unavailable": sorted(manager.shared.errors),
        "errors": dict(manager.shared.errors),
        "uptime_seconds": int(time.time() - _STARTED_AT),
        "metrics": dict(_metrics),
    }


class _Sender:
    """Serialises sends: the receive loop and the watchdog ticker are two
    tasks writing to one socket."""

    def __init__(self, websocket: WebSocket) -> None:
        self._ws = websocket
        self._lock = asyncio.Lock()

    async def send(self, message: Dict[str, Any]) -> None:
        async with self._lock:
            await self._ws.send_text(json.dumps(message))


def _b64(data: Any, limit: int = _MAX_IMAGE_BYTES) -> bytes | None:
    if not isinstance(data, str):
        return None
    try:
        raw = base64.b64decode(data, validate=False)
    except Exception:
        return None
    return raw if 0 < len(raw) <= limit else None


async def _send_emissions(
    sender: _Sender, session_id: str, emissions: List[Emission], stats: "_SessionStats | None" = None
) -> None:
    for emission in emissions:
        event = emission.event
        _metrics["alerts_emitted"] += 1
        if stats is not None:
            stats.alerts[event.alert_type] += 1
        # The alert is the product: one line each, with what it was and how sure
        # we are. The evidence itself is only logged as a yes/no.
        log_event(
            log,
            logging.INFO,
            "alert_emitted",
            alert=event.alert_type,
            severity=event.severity,
            confidence=event.confidence,
            evidence=emission.clip is not None,
        )
        await sender.send(alert_to_message(session_id, event, emission.clip))


async def _ticker(
    sender: _Sender, session, session_id: str, loop: asyncio.AbstractEventLoop, stats: "_SessionStats"
) -> None:
    while True:
        await asyncio.sleep(_TICK_SECONDS)
        emissions = await loop.run_in_executor(None, session.on_tick)
        await _send_emissions(sender, session_id, emissions, stats)


@app.websocket("/analyze/{session_id}")
async def analyze(websocket: WebSocket, session_id: str) -> None:
    # Every log line below -- including those from other modules -- now carries
    # this session_id automatically.
    id_token = session_id_var.set(session_id)
    stats = _SessionStats()
    reason = "client_disconnect"
    ticker = None
    session_created = False
    try:
        await websocket.accept()
        _metrics["sessions_total"] += 1
        client = websocket.client.host if websocket.client else None
        log_event(log, logging.INFO, "session_connected", client=client)

        try:
            session = manager.get_or_create(session_id)
            session_created = True
        except Exception:  # noqa: BLE001
            # e.g. a missing system library or model: say so once, clearly, and
            # close with a reason the caller (the backend relay) can log --
            # instead of an unhandled ASGI traceback per connection attempt.
            _metrics["sessions_failed"] += 1
            reason = "session_create_failed"
            log_event(log, logging.ERROR, "session_create_failed", exc_info=True,
                      unavailable=dict(manager.shared.errors))
            await _close_quietly(websocket, 1011, "analysis unavailable")
            return

        sender = _Sender(websocket)
        loop = asyncio.get_running_loop()
        ticker = asyncio.create_task(_ticker(sender, session, session_id, loop, stats))

        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                stats.invalid_messages += 1
                continue
            if not isinstance(msg, dict):
                stats.invalid_messages += 1
                continue

            msg_type = msg.get("type")

            if msg_type == "CONNECT":
                # No response required by the contract's video-stream
                # handshake beyond staying connected.
                continue

            if msg_type == "ENROLL":
                await _handle_enroll(sender, session, msg, loop)
                continue

            if msg_type == "VIDEO_FRAME":
                jpeg_bytes = _b64(msg.get("data"))
                if jpeg_bytes is None:
                    stats.invalid_messages += 1
                    continue

                # Inference is CPU-bound; the executor keeps one session's
                # frame from blocking every other session's messages.
                began = time.monotonic()
                emissions = await loop.run_in_executor(None, session.on_video_frame, jpeg_bytes)
                took = time.monotonic() - began
                if took > _SLOW_FRAME_SECONDS:
                    stats.slow_frames += 1
                    log_event(log, logging.WARNING, "slow_frame", seconds=round(took, 2),
                              active_sessions=manager.count())
                await _send_emissions(sender, session_id, emissions, stats)

                stats.frames += 1
                _metrics["frames_processed"] += 1
                if stats.frames % _ANALYSIS_EVERY_N_FRAMES == 0:
                    snapshot = await loop.run_in_executor(None, session.analysis_snapshot)
                    await sender.send(
                        {
                            "type": "AI_ANALYSIS",
                            "session_id": session_id,
                            "frame_number": msg.get("frame_number", stats.frames),
                            "timestamp": int(time.time()),
                            "analysis": snapshot,
                        }
                    )

                sequence_number = msg.get("sequence_number")
                if sequence_number is not None and stats.frames % _FRAME_ACK_EVERY_N_FRAMES == 0:
                    await sender.send(
                        {"type": "FRAME_ACK", "sequence_number": sequence_number, "timestamp": int(time.time())}
                    )
                continue

            if msg_type == "AUDIO_FRAME":
                pcm_bytes = _b64(msg.get("data"), limit=1024 * 1024)
                if pcm_bytes is None:
                    stats.invalid_messages += 1
                    continue
                emissions = await loop.run_in_executor(None, session.on_audio_chunk, pcm_bytes)
                stats.audio_chunks += 1
                await _send_emissions(sender, session_id, emissions, stats)
                continue

            if msg_type == "DISCONNECT":
                reason = "client_requested"
                break

            # Unknown message type -- ignore rather than crash the
            # connection, matching ws-relay.ts's own default: case.
            stats.invalid_messages += 1

    except WebSocketDisconnect as exc:
        reason = f"client_disconnect_{exc.code}"
    except Exception:  # noqa: BLE001 - one session's failure must not reach the server
        reason = "error"
        _metrics["sessions_failed"] += 1
        log_event(log, logging.ERROR, "session_failed", exc_info=True)
        await _close_quietly(websocket, 1011, "internal error")
    finally:
        if ticker is not None:
            ticker.cancel()
        if session_created:
            manager.remove(session_id)
        log_event(
            log,
            logging.INFO,
            "session_closed",
            reason=reason,
            duration_seconds=round(time.monotonic() - stats.started, 1),
            frames=stats.frames,
            audio_chunks=stats.audio_chunks,
            slow_frames=stats.slow_frames,
            invalid_messages=stats.invalid_messages,
            alerts=dict(stats.alerts),
        )
        session_id_var.reset(id_token)


async def _close_quietly(websocket: WebSocket, code: int, reason: str) -> None:
    """Closes a socket that may already be closed, without raising."""
    try:
        await websocket.close(code=code, reason=reason)
    except Exception:  # noqa: BLE001
        pass


async def _handle_enroll(sender: _Sender, session, msg: Dict[str, Any], loop: asyncio.AbstractEventLoop) -> None:
    """Sets the session's identity reference. The reference *images* are used
    only to compute an embedding and are never stored; a client that already
    holds an embedding (from an earlier check-in) can send that instead."""
    embedding = msg.get("embedding")
    if isinstance(embedding, list):
        ok = session.set_reference(embedding)
        log_event(log, logging.INFO, "enrollment_reference_set", ok=ok, source="embedding")
        await sender.send(
            {"type": "ENROLLMENT_RESULT", "ok": ok, "reason": None if ok else "invalid_embedding", "embedding": None}
        )
        return

    images = msg.get("images")
    if not isinstance(images, list):
        await sender.send({"type": "ENROLLMENT_RESULT", "ok": False, "reason": "no_images", "embedding": None})
        return
    jpegs = [b for b in (_b64(i) for i in images[:_MAX_ENROLL_IMAGES]) if b is not None]
    result = await loop.run_in_executor(None, session.enroll, jpegs)
    # Outcome only: never the images, and never the embedding.
    log_event(log, logging.INFO, "enrollment_result", ok=result.ok, reason=result.reason,
              images=len(jpegs), faces_seen=result.faces_seen)
    await sender.send(
        {
            "type": "ENROLLMENT_RESULT",
            "ok": result.ok,
            "reason": result.reason,
            "faces_seen": result.faces_seen,
            # Returned so the client can re-establish the reference on a
            # reconnect without re-capturing the candidate.
            "embedding": result.embedding,
        }
    )
