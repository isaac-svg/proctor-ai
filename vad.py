from silero_vad import load_silero_vad, get_speech_timestamps
import threading
import numpy as np

# pyaudio (a PortAudio binding, awkward to build on a server) is only needed
# for the local-demo microphone path, so it's imported lazily in __init__
# when open_stream=True -- the network service never needs it.
FORMAT = 8  # pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 16000

_model = None
_model_lock = threading.Lock()
_infer_lock = threading.Lock()


def _shared_model():
    """One Silero model for the whole process instead of one copy per exam
    session. The model carries internal recurrent state that
    get_speech_timestamps() resets and mutates, so it is NOT safe to call
    concurrently -- every use goes through _infer_lock below."""
    global _model
    with _model_lock:
        if _model is None:
            _model = load_silero_vad()
        return _model


class VoiceActivityDectector:
    def __init__(self, open_stream: bool = True):
        self.format = FORMAT
        self.rate = RATE
        self.chunk = CHUNK
        self.channels = 1
        self.model = _shared_model()

        # open_stream=False skips grabbing a real microphone entirely --
        # for a server-side instance (proctor-ai/session_manager.py) fed
        # externally-supplied PCM chunks over a network connection, opening
        # the host machine's default mic would be both wrong (whose mic?)
        # and, in a multi-session service, a hardware resource only one
        # instance could ever hold. read_chunk()/close() stay meaningful
        # only when open_stream=True (the local-demo path, main.py).
        self.pyaudio_instance = None
        self.stream = None
        if open_stream:
            import pyaudio

            self.format = pyaudio.paInt16
            self.pyaudio_instance = pyaudio.PyAudio()
            self.stream = self.pyaudio_instance.open(format=self.format,
            input=True,
            rate=self.rate,
            frames_per_buffer = self.chunk,
            channels = self.channels
            )
            print("Microphone Access successful")

    def analyze_pcm(self, pcm_bytes: bytes) -> bool:
        """
        Run VAD on an externally-supplied buffer of raw PCM samples (format/
        rate/channels must match FORMAT/RATE/CHANNELS above -- s16le mono
        16kHz). Doesn't touch self.stream, so this is safe to call on an
        instance constructed with open_stream=False.
        """
        return self.speech_ratio(pcm_bytes) > 0.0

    def speech_ratio(self, pcm_bytes: bytes) -> float:
        """Fraction (0..1) of the buffer's duration the VAD calls speech --
        richer than analyze_pcm's yes/no, so rules can tell a syllable from
        a sentence."""
        total = len(pcm_bytes) / 2 / self.rate
        if total <= 0:
            return 0.0
        # Build the tensor straight from the PCM. (The previous version wrote a
        # WAV into a BytesIO and called silero's read_audio(), which needs a
        # torchaudio audio-file backend -- absent on many installs, where it
        # raised on the first real second of audio.)
        import torch

        samples = np.frombuffer(pcm_bytes[: (len(pcm_bytes) // 2) * 2], dtype="<i2").astype(np.float32) / 32768.0
        wav = torch.from_numpy(samples)
        with _infer_lock:
            timestamps = get_speech_timestamps(wav, self.model, sampling_rate=self.rate, return_seconds=True)
        speech = sum(max(0.0, seg["end"] - seg["start"]) for seg in timestamps)
        return min(1.0, speech / total)

    def read_chunk(self) -> bool:
        self.raw = self.stream.read(self.chunk)
        return self.analyze_pcm(self.raw)

    def close(self):
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
        if self.pyaudio_instance is not None:
            self.pyaudio_instance.terminate()

