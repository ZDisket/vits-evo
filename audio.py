from pathlib import Path
import wave

import numpy as np


def _to_pcm16(audio) -> np.ndarray:
    waveform = np.asarray(audio, dtype=np.float32)
    waveform = np.squeeze(waveform)
    if waveform.ndim != 1:
        raise ValueError(f"Expected mono audio after squeeze, got shape {waveform.shape}.")
    waveform = np.clip(waveform, -1.0, 1.0)
    return (waveform * 32767.0).astype(np.int16)


def save_wav(path, audio, sampling_rate: int) -> Path:
    target = Path(path)
    pcm16 = _to_pcm16(audio)

    with wave.open(str(target), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sampling_rate))
        handle.writeframes(pcm16.tobytes())

    return target
