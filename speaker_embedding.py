"""
Speaker embeddings, for telling whether more than one person is talking.

WeSpeaker ResNet34 (Apache-2.0, ONNX) over 80-bin Kaldi fbank features,
computed with torchaudio (already a dependency for Silero) and run with
onnxruntime. This is optional: if onnxruntime or the model is unavailable,
`load_speaker_embedder()` returns None, the MULTIPLE_VOICES rule simply never
fires, and the service reports the capability as off -- it never guesses
from pitch instead.
"""

from __future__ import annotations

import logging
import threading
from typing import List, Optional

import numpy as np

from model_assets import WESPEAKER, ensure_asset

log = logging.getLogger("proctor-ai.speaker")

SAMPLE_RATE = 16000
# Embeddings from under ~1.5 s of audio are unreliable.
MIN_SAMPLES = int(1.5 * SAMPLE_RATE)


class SpeakerEmbedder:
    def __init__(self) -> None:
        import onnxruntime as ort  # deferred: optional dependency
        import torch
        import torchaudio

        self._torch = torch
        self._fbank = torchaudio.compliance.kaldi.fbank
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1  # many sessions share the box; don't oversubscribe
        self._session = ort.InferenceSession(
            str(ensure_asset(WESPEAKER)), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._lock = threading.Lock()

    def embed(self, pcm_s16le: bytes) -> Optional[List[float]]:
        """L2-normalised 256-d embedding for 16 kHz mono s16le audio, or None
        if there isn't enough audio to be reliable."""
        n = len(pcm_s16le) // 2
        if n < MIN_SAMPLES:
            return None
        wave = np.frombuffer(pcm_s16le[: n * 2], dtype="<i2").astype(np.float32)
        feats = self._fbank(
            self._torch.from_numpy(wave).unsqueeze(0),  # int16-scale, as the model was trained
            num_mel_bins=80,
            frame_length=25,
            frame_shift=10,
            dither=0.0,
            sample_frequency=SAMPLE_RATE,
            window_type="hamming",
            use_energy=False,
        )
        feats = (feats - feats.mean(dim=0, keepdim=True)).numpy()[None].astype(np.float32)
        with self._lock:
            out = self._session.run(None, {"feats": feats})[0].ravel()
        norm = float(np.linalg.norm(out))
        if norm == 0.0:
            return None
        return [float(x) for x in out / norm]


def load_speaker_embedder() -> Optional[SpeakerEmbedder]:
    try:
        return SpeakerEmbedder()
    except Exception as exc:  # ImportError, download failure, bad model...
        log.warning("speaker embeddings unavailable (%s); MULTIPLE_VOICES detection is off", exc)
        return None
