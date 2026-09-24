"""
FastAPI/websockets service exposing one endpoint per exam session:
WS /analyze/{session_id}. shepherd-backend's video-stream-relay.ts is the
only expected client -- it authenticates the real Electron connection and
proxies VIDEO_FRAME/AUDIO_FRAME messages through to here unmodified (a thin
proxy, per its own comments), then relays whatever this service sends back
(AI_ANALYSIS/AI_ALERT/FRAME_ACK) to Electron. See shepherd-ai's
docs/AI_INTEGRATION.md for the full pipeline this is one piece of, and
README.md (this repo) for setup.

Run: uvicorn service:app --port 8901
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from typing import Any, Dict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from alert_mapping import map_event
from session_manager import ProctorSessionManager

app = FastAPI(title="proctor-ai")
manager = ProctorSessionManager()

# The contract only says AI_ANALYSIS is "periodic" -- every Nth video frame
# keeps the informational stream light without making it useless. Frame
# cadence itself is whatever shepherd-ai's snapshot_interval_ms sends at
# (see ffmpeg-args.ts), typically ~1/sec, so this is roughly one AI_ANALYSIS
# every few seconds.
_ANALYSIS_EVERY_N_FRAMES = 5
_FRAME_ACK_EVERY_N_FRAMES = 10


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "OK", "active_sessions": manager.count()}


async def _send(websocket: WebSocket, message: Dict[str, Any]) -> None:
    await websocket.send_text(json.dumps(message))


@app.websocket("/analyze/{session_id}")
async def analyze(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()
    session = manager.get_or_create(session_id)
    loop = asyncio.get_event_loop()
    frame_count = 0

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            msg_type = msg.get("type")

            if msg_type == "CONNECT":
                # No response required by the contract's video-stream
                # handshake beyond staying connected -- unlike /ws/sessions,
                # this path has no CONNECTED ack defined.
                continue

            if msg_type == "VIDEO_FRAME":
                data_b64 = msg.get("data")
                if not isinstance(data_b64, str):
                    continue
                try:
                    jpeg_bytes = base64.b64decode(data_b64)
                except Exception:
                    continue

                # mediapipe/YOLO inference is CPU-bound; offloading to the
                # default thread-pool executor keeps this session's frame
                # from blocking every other session's messages on the same
                # event loop. Doesn't fix the underlying single-process
                # concurrency ceiling (see proctor-ai/README.md) -- just
                # keeps one slow frame from stalling the whole service.
                events = await loop.run_in_executor(None, session.on_video_frame, jpeg_bytes)

                for ev in events:
                    alert = map_event(ev)
                    if alert is None:
                        continue
                    await _send(
                        websocket,
                        {
                            "type": "AI_ALERT",
                            "session_id": session_id,
                            "timestamp": int(time.time()),
                            **alert,
                        },
                    )

                frame_count += 1

                if frame_count % _ANALYSIS_EVERY_N_FRAMES == 0:
                    await _send(
                        websocket,
                        {
                            "type": "AI_ANALYSIS",
                            "session_id": session_id,
                            "frame_number": msg.get("frame_number", frame_count),
                            "timestamp": int(time.time()),
                            "analysis": session.analysis_snapshot(),
                        },
                    )

                sequence_number = msg.get("sequence_number")
                if sequence_number is not None and frame_count % _FRAME_ACK_EVERY_N_FRAMES == 0:
                    await _send(
                        websocket,
                        {
                            "type": "FRAME_ACK",
                            "sequence_number": sequence_number,
                            "timestamp": int(time.time()),
                        },
                    )
                continue

            if msg_type == "AUDIO_FRAME":
                data_b64 = msg.get("data")
                if not isinstance(data_b64, str):
                    continue
                try:
                    pcm_bytes = base64.b64decode(data_b64)
                except Exception:
                    continue
                await loop.run_in_executor(None, session.on_audio_chunk, pcm_bytes)
                continue

            if msg_type == "DISCONNECT":
                break

            # Unknown message type -- ignore rather than crash the
            # connection, matching ws-relay.ts's own default: case.

    except WebSocketDisconnect:
        pass
    finally:
        manager.remove(session_id)
