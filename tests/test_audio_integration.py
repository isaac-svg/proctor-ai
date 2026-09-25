"""
Audio scenarios through the whole stack (real Silero VAD + WeSpeaker speaker
embeddings) using real speech.

    PROCTOR_TEST_VOICES_DIR=/path/to/dir pytest tests/test_audio_integration.py

The directory holds `voice_a.wav` and `voice_b.wav`: 16 kHz mono 16-bit WAV,
two *different* speakers, each at least ~20 s of continuous speech (macOS:
`say -v Samantha "..." -o a.aiff && afconvert -f WAVE -d LEI16@16000 -c 1 a.aiff voice_a.wav`).
Skipped when unset or when the heavy dependencies are missing.
"""

import os
import wave
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("silero_vad")
pytest.importorskip("mediapipe")

VOICES = os.environ.get("PROCTOR_TEST_VOICES_DIR")
pytestmark = pytest.mark.skipif(not VOICES, reason="set PROCTOR_TEST_VOICES_DIR to run audio scenarios")

WINDOW = 32000  # 1 s of 16 kHz s16le


def pcm(name):
    with wave.open(str(Path(VOICES) / name)) as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1
        return w.readframes(w.getnframes())


def seconds(data, start_s, n_s):
    return data[start_s * WINDOW : (start_s + n_s) * WINDOW]


class Clock:
    t = 1000.0

    def __call__(self):
        return self.t


@pytest.fixture(scope="module")
def shared():
    from session_manager import SharedModels

    s = SharedModels()
    s.warm_up()
    return s


@pytest.fixture
def session(shared):
    from session_manager import ProctorSession

    clock = Clock()
    s = ProctorSession("audio-scenario", shared, clock=clock)
    yield s, clock
    s.close()


def feed(session_clock, chunks):
    s, clock = session_clock
    out = []
    for chunk in chunks:
        clock.t += 1.0
        out += s.on_audio_chunk(chunk)
    return out


def kinds(emissions):
    return [e.event.alert_type for e in emissions]


def test_speaker_diarization_is_available(shared):
    assert shared.capabilities()["speaker_diarization"], shared.errors


def test_one_person_talking_for_a_while_is_sustained_speaking_but_never_multiple_voices(session):
    a = pcm("voice_a.wav")
    got = kinds(feed(session, [seconds(a, i, 1) for i in range(20)]))
    assert "VOICE_DETECTED" in got
    assert "SUSTAINED_SPEAKING" in got
    assert "MULTIPLE_VOICES" not in got
    assert got.count("VOICE_DETECTED") == 1  # not one per second


def test_two_different_voices_raise_multiple_voices_with_audio_evidence(session):
    a, b = pcm("voice_a.wav"), pcm("voice_b.wav")
    chunks = []
    for turn in range(3):  # conversational turns of ~4 s each, alternating
        chunks += [seconds(a, turn * 4 + i, 1) for i in range(4)]
        chunks += [seconds(b, turn * 4 + i, 1) for i in range(4)]
    emissions = feed(session, chunks)
    voices = [e for e in emissions if e.event.alert_type == "MULTIPLE_VOICES"]
    assert len(voices) == 1, kinds(emissions)
    assert voices[0].clip and voices[0].clip["audio"]["wav_b64"]


def test_a_dead_microphone_is_reported(session):
    silence = b"\x00\x00" * 16000
    got = kinds(feed(session, [silence] * 25))
    assert got == ["MICROPHONE_SILENT"]


def test_a_quiet_room_is_not_a_dead_microphone_and_raises_nothing(session):
    rng = np.random.default_rng(1)
    quiet = lambda: (rng.normal(0, 40, 16000)).astype("<i2").tobytes()  # ~ -58 dBFS hiss
    assert kinds(feed(session, [quiet() for _ in range(40)])) == []
