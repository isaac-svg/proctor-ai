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
import threading
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

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


def _warm_up() -> None:
    try:
        manager.shared.warm_up()
    finally:
        _warm["state"] = "ready"


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Loading models (and, the first time, downloading them) takes long enough
    # that doing it on the first real frame would stall that exam's video.
    threading.Thread(target=_warm_up, name="proctor-ai-warmup", daemon=True).start()
    yield


app = FastAPI(title="proctor-ai", lifespan=lifespan)


@app.get("/health")
def health() -> Dict[str, Any]:
    """`status: OK` as soon as the process is serving (what
    run-shepherd-stack.sh waits on). `models` says whether model loading has
    finished, and `capabilities` says which detections are really available --
    a deployment can see, e.g., that speaker diarization is off because
    onnxruntime isn't installed, instead of silently getting no alerts."""
    return {
        "status": "OK",
        "active_sessions": manager.count(),
        "models": _warm["state"],
        "capabilities": manager.shared.capabilities(),
        "unavailable": sorted(manager.shared.errors),
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


async def _send_emissions(sender: _Sender, session_id: str, emissions: List[Emission]) -> None:
    for emission in emissions:
        await sender.send(alert_to_message(session_id, emission.event, emission.clip))


async def _ticker(sender: _Sender, session, session_id: str, loop: asyncio.AbstractEventLoop) -> None:
    while True:
        await asyncio.sleep(_TICK_SECONDS)
        emissions = await loop.run_in_executor(None, session.on_tick)
        await _send_emissions(sender, session_id, emissions)


@app.websocket("/analyze/{session_id}")
async def analyze(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()
    session = manager.get_or_create(session_id)
    sender = _Sender(websocket)
    loop = asyncio.get_running_loop()
    frame_count = 0
    ticker = asyncio.create_task(_ticker(sender, session, session_id, loop))

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(msg, dict):
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
                    continue

                # Inference is CPU-bound; the executor keeps one session's
                # frame from blocking every other session's messages.
                emissions = await loop.run_in_executor(None, session.on_video_frame, jpeg_bytes)
                await _send_emissions(sender, session_id, emissions)

                frame_count += 1
                if frame_count % _ANALYSIS_EVERY_N_FRAMES == 0:
                    snapshot = await loop.run_in_executor(None, session.analysis_snapshot)
                    await sender.send(
                        {
                            "type": "AI_ANALYSIS",
                            "session_id": session_id,
                            "frame_number": msg.get("frame_number", frame_count),
                            "timestamp": int(time.time()),
                            "analysis": snapshot,
                        }
                    )

                sequence_number = msg.get("sequence_number")
                if sequence_number is not None and frame_count % _FRAME_ACK_EVERY_N_FRAMES == 0:
                    await sender.send(
                        {"type": "FRAME_ACK", "sequence_number": sequence_number, "timestamp": int(time.time())}
                    )
                continue

            if msg_type == "AUDIO_FRAME":
                pcm_bytes = _b64(msg.get("data"), limit=1024 * 1024)
                if pcm_bytes is None:
                    continue
                emissions = await loop.run_in_executor(None, session.on_audio_chunk, pcm_bytes)
                await _send_emissions(sender, session_id, emissions)
                continue

            if msg_type == "DISCONNECT":
                break

            # Unknown message type -- ignore rather than crash the
            # connection, matching ws-relay.ts's own default: case.

    except WebSocketDisconnect:
        pass
    finally:
        ticker.cancel()
        manager.remove(session_id)


async def _handle_enroll(sender: _Sender, session, msg: Dict[str, Any], loop: asyncio.AbstractEventLoop) -> None:
    """Sets the session's identity reference. The reference *images* are used
    only to compute an embedding and are never stored; a client that already
    holds an embedding (from an earlier check-in) can send that instead."""
    embedding = msg.get("embedding")
    if isinstance(embedding, list):
        ok = session.set_reference(embedding)
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
