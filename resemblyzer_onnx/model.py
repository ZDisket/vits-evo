from pathlib import Path
from typing import Dict, Optional, Union

import torch
import torchaudio.functional as audio_functional
from torch import nn
from torch.nn import functional as F

from .hparams import *


_N_FFT = int(sampling_rate * mel_window_length / 1000)
_HOP_LENGTH = int(sampling_rate * mel_window_step / 1000)
_F_MAX = sampling_rate / 2
_MEL_FRAME_START = 1
_MEL_FRAME_STOP = _MEL_FRAME_START + partials_n_frames
DEFAULT_TORCH_WEIGHTS_FPATH = Path(__file__).resolve().parent.joinpath("pretrained.pt")


def get_default_torch_weights_fpath() -> Path:
    return DEFAULT_TORCH_WEIGHTS_FPATH


def _resolve_weights_fpath(weights_fpath: Optional[Union[Path, str]]) -> Path:
    weights_fpath = Path(weights_fpath) if weights_fpath is not None else DEFAULT_TORCH_WEIGHTS_FPATH
    if not weights_fpath.exists():
        raise FileNotFoundError("Couldn't find the torch voice encoder model at %s." % weights_fpath)
    return weights_fpath


def _load_encoder_state(weights_fpath: Optional[Union[Path, str]]) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(_resolve_weights_fpath(weights_fpath), map_location="cpu")
    model_state = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint
    return {
        key: value
        for key, value in model_state.items()
        if not key.startswith("similarity_")
    }


class OnnxVoiceEncoderModel(nn.Module):
    def __init__(self, weights_fpath: Optional[Union[Path, str]] = None):
        super().__init__()

        self.lstm = nn.LSTM(mel_n_channels, model_hidden_size, model_num_layers, batch_first=True)
        self.linear = nn.Linear(model_hidden_size, model_embedding_size)
        self.relu = nn.ReLU()
        self.register_buffer("stft_window", torch.hann_window(_N_FFT))
        self.register_buffer(
            "mel_filterbank",
            audio_functional.melscale_fbanks(
                n_freqs=_N_FFT // 2 + 1,
                f_min=0.0,
                f_max=_F_MAX,
                n_mels=mel_n_channels,
                sample_rate=sampling_rate,
                norm="slaney",
                mel_scale="slaney",
            ),
        )

        state_dict = _load_encoder_state(weights_fpath)
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        allowed_missing = {"stft_window", "mel_filterbank"}
        missing_keys = set(missing_keys)
        if unexpected_keys or missing_keys - allowed_missing:
            raise RuntimeError(
                "Unexpected checkpoint contents: missing=%s unexpected=%s" %
                (sorted(missing_keys), sorted(unexpected_keys))
            )

    def _wav_to_mel_spectrogram(self, wav: torch.Tensor):
        stft = torch.stft(
            wav,
            n_fft=_N_FFT,
            hop_length=_HOP_LENGTH,
            win_length=_N_FFT,
            window=self.stft_window,
            center=True,
            pad_mode="constant",
            return_complex=False,
        )
        power = stft.pow(2).sum(dim=-1)
        return torch.matmul(power.transpose(1, 2), self.mel_filterbank)

    def _embed_partials(self, partial_mels: torch.Tensor):
        _, (hidden, _) = self.lstm(partial_mels)
        embeds_raw = self.relu(self.linear(hidden[-1]))
        return F.normalize(embeds_raw, p=2, dim=1, eps=1e-12)

    def forward(self, wav: torch.Tensor):
        if wav.ndim != 2:
            raise ValueError("Expected waveforms with shape (batch, num_samples)")

        wav = wav.to(dtype=torch.float32)
        mel = self._wav_to_mel_spectrogram(wav)
        mel = mel[:, _MEL_FRAME_START:_MEL_FRAME_STOP, :]
        return self._embed_partials(mel.to(dtype=torch.float32))
