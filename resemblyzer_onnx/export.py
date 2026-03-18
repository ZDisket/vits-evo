from pathlib import Path
from typing import Optional, Union

import torch

from .hparams import mel_window_step, partials_n_frames, sampling_rate
from .model import OnnxVoiceEncoderModel


DEFAULT_ONNX_WEIGHTS_FPATH = Path(__file__).resolve().parent.joinpath("pretrained.onnx")


def get_default_onnx_weights_fpath() -> Path:
    return DEFAULT_ONNX_WEIGHTS_FPATH


def export_voice_encoder_to_onnx(onnx_fpath: Optional[Union[str, Path]] = None,
                                 weights_fpath: Optional[Union[str, Path]] = None,
                                 rate=1.3, min_coverage=0.75, opset_version=17):
    _ = rate, min_coverage
    model = OnnxVoiceEncoderModel(weights_fpath=weights_fpath).eval()
    onnx_fpath = Path(onnx_fpath) if onnx_fpath is not None else DEFAULT_ONNX_WEIGHTS_FPATH
    onnx_fpath.parent.mkdir(parents=True, exist_ok=True)

    partial_num_samples = partials_n_frames * int(sampling_rate * mel_window_step / 1000)
    dummy_waveforms = torch.zeros((2, partial_num_samples), dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy_waveforms,
        str(onnx_fpath),
        opset_version=opset_version,
        input_names=["waveforms"],
        output_names=["embeddings"],
        dynamic_axes={
            "waveforms": {0: "batch_size", 1: "num_samples"},
            "embeddings": {0: "batch_size"},
        },
    )
    return onnx_fpath
