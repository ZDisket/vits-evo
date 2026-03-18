from pathlib import Path
from typing import Optional, Union, cast
import struct

import librosa
import numpy as np
from scipy.ndimage import binary_dilation

from .hparams import *

int16_max = (2 ** 15) - 1
_warned_missing_webrtcvad = False


def _trim_silence_fallback(wav: np.ndarray):
    wav = np.asarray(wav, dtype=np.float32)
    if wav.size == 0:
        return wav

    samples_per_window = max(1, (vad_window_length * sampling_rate) // 1000)
    frame_count = int(np.ceil(len(wav) / samples_per_window))
    padded = np.pad(wav, (0, frame_count * samples_per_window - len(wav)), mode="constant")
    frames = padded.reshape(frame_count, samples_per_window)
    frame_rms = np.sqrt(np.mean(frames * frames, axis=1))

    peak_rms = float(frame_rms.max())
    if peak_rms <= 0.0 or not np.isfinite(peak_rms):
        return wav

    threshold = max(1e-4, peak_rms * 0.1)
    active = frame_rms > threshold
    if not np.any(active):
        return wav

    start_frame = max(0, int(np.argmax(active)) - 1)
    end_frame = min(frame_count, int(frame_count - np.argmax(active[::-1])) + 1)
    start = start_frame * samples_per_window
    end = min(len(wav), end_frame * samples_per_window)
    return wav[start:end]


def preprocess_wav(fpath_or_wav: Union[str, Path, np.ndarray], source_sr: Optional[int] = None):
    if isinstance(fpath_or_wav, (str, Path)):
        wav, source_sr = cast(tuple, librosa.load(str(fpath_or_wav), sr=None))  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
    else:
        wav = np.asarray(fpath_or_wav, dtype=np.float32)

    if source_sr is not None and source_sr != sampling_rate:
        wav = librosa.resample(wav, orig_sr=source_sr, target_sr=sampling_rate)

    wav = normalize_volume(wav, audio_norm_target_dBFS, increase_only=True)
    wav = trim_long_silences(wav)
    return np.asarray(wav, dtype=np.float32)


def wav_to_mel_spectrogram(wav: np.ndarray):
    frames = librosa.feature.melspectrogram(
        y=np.asarray(wav, dtype=np.float32),
        sr=sampling_rate,
        n_fft=int(sampling_rate * mel_window_length / 1000),
        hop_length=int(sampling_rate * mel_window_step / 1000),
        n_mels=mel_n_channels,
        power=2.0,
        center=True,
        pad_mode="constant",
        htk=False,
        norm="slaney",
    )
    return frames.astype(np.float32).T


def trim_long_silences(wav: np.ndarray):
    global _warned_missing_webrtcvad
    try:
        import webrtcvad
    except ImportError:
        if not _warned_missing_webrtcvad:
            print(
                "Warning: webrtcvad is unavailable, so using a simple amplitude-based silence trim fallback. "
                "Install webrtcvad or webrtcvad-wheels for the original preprocessing behavior."
            )
            _warned_missing_webrtcvad = True
        return _trim_silence_fallback(wav)

    samples_per_window = (vad_window_length * sampling_rate) // 1000
    wav = np.asarray(wav, dtype=np.float32)
    wav = wav[:len(wav) - (len(wav) % samples_per_window)]
    if len(wav) == 0:
        return wav

    pcm_wave = struct.pack("%dh" % len(wav), *(np.round(wav * int16_max)).astype(np.int16))

    voice_flags = []
    vad = webrtcvad.Vad(mode=3)
    for window_start in range(0, len(wav), samples_per_window):
        window_end = window_start + samples_per_window
        voice_flags.append(vad.is_speech(pcm_wave[window_start * 2:window_end * 2],
                                         sample_rate=sampling_rate))
    voice_flags = np.array(voice_flags)

    def moving_average(array, width):
        array_padded = np.concatenate((np.zeros((width - 1) // 2), array, np.zeros(width // 2)))
        ret = np.cumsum(array_padded, dtype=float)
        ret[width:] = ret[width:] - ret[:-width]
        return ret[width - 1:] / width

    audio_mask = moving_average(voice_flags, vad_moving_average_width)
    audio_mask = np.round(audio_mask).astype(bool)
    audio_mask = binary_dilation(audio_mask, np.ones(vad_max_silence_length + 1))
    audio_mask = np.repeat(audio_mask, samples_per_window)
    return wav[audio_mask == True]


def normalize_volume(wav: np.ndarray, target_dBFS, increase_only=False, decrease_only=False):
    if increase_only and decrease_only:
        raise ValueError("Both increase only and decrease only are set")
    wav = np.asarray(wav, dtype=np.float32)
    rms = np.sqrt(np.mean((wav * int16_max) ** 2))
    if rms == 0:
        return wav
    wave_dBFS = 20 * np.log10(rms / int16_max)
    dBFS_change = target_dBFS - wave_dBFS
    if dBFS_change < 0 and increase_only or dBFS_change > 0 and decrease_only:
        return wav
    return wav * (10 ** (dBFS_change / 20))
