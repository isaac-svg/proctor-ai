import pyaudio
import pyaudio
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
import numpy as np
import wave 
from io import BytesIO

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 16000

class VoiceActivityDectector:
    def __init__(self, open_stream: bool = True):
        self.format = FORMAT
        self.rate = RATE
        self.chunk = CHUNK
        self.channels = 1
        self.model = load_silero_vad()
        self.buffer = BytesIO()

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
        self.buffer.seek(0)
        self.buffer.truncate(0)

        with wave.open(self.buffer, "wb") as wave_file:
            wave_file.setframerate(self.rate)
            wave_file.setnchannels(self.channels)
            wave_file.setsampwidth(2)
            wave_file.writeframes(pcm_bytes)
        self.buffer.seek(0)
        wav = read_audio(self.buffer)
        timestamp = get_speech_timestamps(wav, self.model, return_seconds=True)

        return bool(timestamp)

    def read_chunk(self) -> bool:
        self.raw = self.stream.read(self.chunk)
        return self.analyze_pcm(self.raw)

    def close(self):
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
        if self.pyaudio_instance is not None:
            self.pyaudio_instance.terminate()

