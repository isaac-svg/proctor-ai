# proctor-ai

Vision/audio proctoring detection core: MediaPipe head-pose + gaze tracking (`facetracking.py`),
YOLO cellphone detection (`obd.py`), Silero VAD voice-activity detection (`vad.py`), unified into
typed suspicious-activity events by `events.py`'s `EventDetector`.

**Provenance**: this is a separate upstream project (`github.com/martin-kagya/proctor-ai`),
checked out here as a sibling of [`shepherd-ai`](../shepherd-ai)/[`shepherd-backend`](../shepherd-backend)/
[`shepherd-dashboard`](../shepherd-dashboard) but not owned by that project — see
`shepherd-ai/docs/PLATFORM_OWNER_GUIDE.md` §5. There is no upstream `LICENSE` file; confirm terms
with the author before this code ships in a real product, independent of the technical integration
below.

## Two ways to run this

**1. Local demo** (`main.py` / `facetracking.py`'s own `main()`) — opens your machine's webcam and
mic directly, shows a live on-screen overlay. Nothing is sent anywhere; this is what the original
project did before any of the files below existed.

```bash
python main.py
```

**2. Network service** (`service.py`) — the piece that makes this useful to Shepherd. Exposes
`WS /analyze/{session_id}`, one connection per exam session, receiving JPEG snapshot frames +
raw PCM audio chunks and returning `AI_ANALYSIS`/`AI_ALERT` messages. See
`shepherd-ai/docs/AI_INTEGRATION.md` for the full pipeline this is one piece of:
[shepherd-ai](../shepherd-ai) captures video/audio → [shepherd-backend](../shepherd-backend)'s
`/ws/video-stream/{id}` authenticates and proxies to this service → this service runs detection →
alerts flow back through the same path to Electron, which turns them into ordinary
`SECURITY_EVENT`s through its existing evidence/policy pipeline.

```bash
pip install -r requirements-service.txt
uvicorn service:app --port 8901
```

`GET /health` reports `{"status": "OK", "active_sessions": <n>}` — used by
[`run-shepherd-stack.sh`](../run-shepherd-stack.sh)'s startup wait.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-service.txt
```

`requirements.txt` (upstream's own file) reads like a full `pip freeze` dump of a shared conda
environment — 246 lines, including unresolvable local build paths (`archspec @ file:///home/conda/
...`) that make `pip install -r requirements.txt` fail outright, confirmed by trying it. Left
as-is since it's upstream's file, not rewritten here, but **not** a usable install path.
`requirements-service.txt` is self-contained instead: everything `facetracking.py`/`obd.py`/
`vad.py`/`service.py` actually import, pinned loosely so pip resolves current wheels.

**Heads up on first run**: `mediapipe` + `torch`/`torchvision`/`torchaudio` + `ultralytics` is a
multi-GB install, and `facetracking.py`'s `ensure_model()` downloads a ~1MB FaceLandmarker model
from Google's CDN the first time `HeadPoseEstimator` is constructed if `face_landmarker.task`
isn't already present (it is, checked into this repo). `obd.py` loads `yolo26n.pt` (also checked
in) at import time. Both add real latency to the very first request this service handles after a
fresh start — this shows up as the service being slow to report `/health` as ready, not as an
error.

## What's additive vs. original

`main.py`, `facetracking.py`'s `main()`, and the original single-session, module-global design in
those files are untouched in behavior — every change below is additive (new optional parameters
with defaults that preserve old behavior, or new files):

- `vad.py` — `VoiceActivityDectector.__init__` gained an `open_stream: bool = True` flag; passing
  `False` skips opening a real microphone (what `session_manager.py`'s server-side instances do).
  The old `read_chunk()` behavior is unchanged; its logic moved into a new `analyze_pcm(pcm_bytes)`
  method that accepts an external buffer instead of reading `self.stream` directly.
- `facetracking.py` — `HeadPoseEstimator` gained an optional `num_faces` constructor param
  (default `1`, unchanged behavior); `PoseResult` gained optional `face_bounding_box`/`num_faces`
  fields; a new `gaze_bucket()` function buckets continuous gaze into `CENTER`/`LEFT`/`RIGHT`/
  `UP`/`DOWN`.
- `events.py` — untouched; `EventDetector`/sub-detectors were already decoupled from capture
  source.
- `alert_mapping.py`, `session_manager.py`, `service.py` — new files, the network-service layer.

## Known limitations (v1, flagged not hidden)

- **Single-process, thread-pool-offloaded inference**: `service.py` offloads each frame's
  mediapipe/YOLO/VAD work to a thread-pool executor so one session's inference doesn't block
  another's WebSocket messages, but this is still one Python process — CPU-bound inference across
  many concurrent sessions will contend for the same cores. Fine for a couple of concurrent
  exam-takers in dev; not a production-scale design.
- **State on disconnect**: a `ProctorSession`'s detector state (rolling look-away/phone-usage
  timers) is torn down immediately when its WebSocket closes. A reconnect starts cold, so a
  drop spanning a "sustained" pattern under-detects it.
- **JPEG snapshot cadence**: `sustained_*`/`frequent_*` detectors were tuned for near-continuous
  frame sampling; at ~1 snapshot/sec (shepherd-ai's default `snapshot_interval_ms`) a 2.5s
  "sustained look-away" threshold only gets 2-3 samples to confirm against. Revisit the cadence
  after a first end-to-end test.
