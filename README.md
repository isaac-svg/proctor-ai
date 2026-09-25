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

## Architecture (service): perception → observations → rules

The service is split in two so the interesting logic can be tested and tuned without any model:

- **Perception** (`facetracking.py`, `obd.py`, `face_identity.py`, `vad.py`, `speaker_embedding.py`,
  `frame_quality.py`, `audio_features.py`) — MediaPipe head pose/gaze, YOLO (people, phones, laptops,
  books), OpenCV YuNet+SFace face recognition, Silero VAD, WeSpeaker speaker embeddings, plain image
  and audio statistics. Turns one JPEG / one second of PCM into *observations* (`observations.py`).
- **Rules** (`rules/*.py`, `pipeline.py`, `pipeline_config.py`) — pure logic over observations and
  timestamps: presence, multiple people, framing, attention, camera covered/frozen/lost, identity,
  objects, speech, whispering, microphone state, multiple voices. No model imports; every threshold
  is in `pipeline_config.py` with its rationale.

Every "sustained" behaviour is a **ratio over a time window with hysteresis** (`rules/episodes.py`),
so one glance, blink or dropped frame can't fire or cancel it. See
[`shepherd-ai/docs/ANTI_CHEAT.md`](../shepherd-ai/docs/ANTI_CHEAT.md) for the full alert catalogue,
what each is worth, and privacy handling.

**Evidence clips** (`evidence.py`): a rolling in-memory buffer (~12 small keyframes, ~8 s of audio).
A clip is copied out only when a rule flags an alert as needing one, capped per session — never a
continuous recording. shepherd-backend's relay strips the clip off the alert and stores it.

**Identity** (`face_identity.py`): `ENROLL` validates check-in photos strictly and returns an
*embedding*; the photos are not kept. The embedding lives in session memory only and is dropped when
the connection closes.

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

`GET /health` reports `{"status": "OK", "active_sessions": <n>, "models": "loading|ready", "capabilities": {identity, speaker_diarization, objects}, "unavailable": [...]}` — so you can see which detections are really available (e.g. speaker diarization is off if `onnxruntime` isn't installed) rather than silently getting no alerts. Also used by
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

## Wire protocol additions

Client → service: `ENROLL` (`{images: [b64 jpeg…]}` or `{embedding: […]}`) → `ENROLLMENT_RESULT`
(`{ok, reason, faces_seen, embedding}`). Service → client: `AI_ALERT` now carries many more
`alert_type`s and may include an `evidence` clip (which shepherd-backend replaces with an
`evidence_id`). The service also sends its own watchdog alerts when video or audio stops arriving.

## Tests

```bash
pytest tests --ignore=tests/smoke_test_service.py --ignore=tests/test_perception_integration.py \
       --ignore=tests/test_audio_integration.py          # rules layer: light deps only, runs in <1 s
pytest tests/smoke_test_service.py                       # real models, real WebSocket route (~1 min first time)

# Real-media scenarios (skipped unless you point them at your own files):
PROCTOR_TEST_FACES_DIR=/path pytest tests/test_perception_integration.py   # person_a.jpg, person_b.jpg
PROCTOR_TEST_VOICES_DIR=/path pytest tests/test_audio_integration.py      # voice_a.wav, voice_b.wav (16 kHz mono)
```

Model files not checked into the repo (`face_detection_yunet`, `face_recognition_sface`, the
speaker model) are downloaded to `./models/` on first use and **verified against a pinned SHA-256**
(`model_assets.py`). Pre-populate `./models/` for air-gapped installs.

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
