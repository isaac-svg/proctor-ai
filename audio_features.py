"""
Model-free audio features for one analysis window of 16 kHz s16le mono PCM.
"""

from __future__ import annotations

import math

import numpy as np

from observations import AudioObservation

SAMPLE_RATE = 16000
_FRAME = 400  # 25 ms
_HOP = 160  # 10 ms
_SILENCE_DBFS = -120.0


def pcm_to_float(pcm_bytes: bytes) -> np.ndarray:
    # An odd trailing byte (a truncated chunk) is dropped, not an error.
    n = len(pcm_bytes) // 2
    return np.frombuffer(pcm_bytes[: n * 2], dtype="<i2").astype(np.float32) / 32768.0


def _spectral_flatness(x: np.ndarray) -> float:
    """Mean spectral flatness over 25 ms frames: 0 = tonal (voiced speech),
    1 = noise-like (whisper, breath, hiss)."""
    if len(x) < _FRAME:
        return 0.0
    window = np.hanning(_FRAME).astype(np.float32)
    values = []
    for start in range(0, len(x) - _FRAME + 1, _HOP * 4):
        seg = x[start : start + _FRAME] * window
        power = np.abs(np.fft.rfft(seg)) ** 2 + 1e-12
        values.append(float(np.exp(np.mean(np.log(power))) / np.mean(power)))
    return float(np.mean(values)) if values else 0.0


def analyze_window(pcm_bytes: bytes, ts: float, speech_ratio: float = 0.0, speaker_embedding=None) -> AudioObservation:
    x = pcm_to_float(pcm_bytes)
    if len(x) == 0:
        return AudioObservation(ts=ts, duration_s=0.0, rms_dbfs=_SILENCE_DBFS, peak=0.0)

    rms = float(np.sqrt(np.mean(x * x)))
    rms_dbfs = 20.0 * math.log10(rms) if rms > 1e-7 else _SILENCE_DBFS
    peak = float(np.max(np.abs(x)))
    clipping = float(np.mean(np.abs(x) >= 0.999))
    zcr = float(np.mean(np.abs(np.diff(np.signbit(x).astype(np.int8)))))

    return AudioObservation(
        ts=ts,
        duration_s=len(x) / SAMPLE_RATE,
        rms_dbfs=rms_dbfs,
        peak=peak,
        clipping_ratio=clipping,
        zero_crossing_rate=zcr,
        spectral_flatness=_spectral_flatness(x),
        speech_ratio=speech_ratio,
        speaker_embedding=speaker_embedding,
    )
