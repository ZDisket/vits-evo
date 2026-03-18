from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, cast

import numpy as np
import onnxruntime as ort

from audio import save_wav
from cleaners import collapse_whitespace
from config import InferenceConfig, load_inference_config
from frontend import build_frontend
from tokenizer import Tokenizer

_TEXT_INPUT_NAMES = ("text", "input", "x")
_TEXT_LENGTH_INPUT_NAMES = ("text_lengths", "input_lengths", "x_lengths")
_KNOWN_INTEGER_SPEAKER_INPUT_NAMES = (
    "sid",
    "speaker_id",
)
_KNOWN_FLOAT_SPEAKER_INPUT_NAMES = (
    "speaker_embedding",
    "spk_emb",
    "g",
    "speaker",
)


@dataclass
class SessionInputSpec:
    name: str
    type: str
    shape: Sequence[object]

    @property
    def is_integer(self) -> bool:
        return "int" in self.type

    @property
    def is_float(self) -> bool:
        return "float" in self.type or "double" in self.type


def _default_providers() -> Sequence[str]:
    available = ort.get_available_providers()
    preferred = ["CUDAExecutionProvider", "ROCMExecutionProvider", "CPUExecutionProvider"]
    selected = [provider for provider in preferred if provider in available]
    if selected:
        return selected
    if available:
        return [available[0]]
    return ["CPUExecutionProvider"]


def _normalize_providers(providers: Optional[Iterable[str]]) -> Sequence[str]:
    if providers is None:
        return _default_providers()

    requested = list(providers)
    available = set(ort.get_available_providers())
    selected = [provider for provider in requested if provider in available]
    if not selected:
        raise ValueError(
            f"None of the requested providers are available. Requested={requested}, available={sorted(available)}"
        )
    return selected


def _select_input_name(inputs: Dict[str, SessionInputSpec], candidates: Sequence[str], fallback_index: int) -> str:
    for name in candidates:
        if name in inputs:
            return name
    try:
        return list(inputs.keys())[fallback_index]
    except IndexError as exc:
        raise ValueError("The ONNX model does not expose the expected text inputs.") from exc


def _select_speaker_input(
    inputs: Dict[str, SessionInputSpec],
    text_name: str,
    text_lengths_name: str,
    *,
    prefer_embedding: bool,
):
    preferred_names = (
        _KNOWN_FLOAT_SPEAKER_INPUT_NAMES + _KNOWN_INTEGER_SPEAKER_INPUT_NAMES
        if prefer_embedding
        else _KNOWN_INTEGER_SPEAKER_INPUT_NAMES + _KNOWN_FLOAT_SPEAKER_INPUT_NAMES
    )
    for name in preferred_names:
        if name in inputs:
            spec = inputs[name]
            if prefer_embedding and spec.is_float:
                return spec
            if not prefer_embedding and spec.is_integer:
                return spec

    if prefer_embedding:
        for name in _KNOWN_FLOAT_SPEAKER_INPUT_NAMES:
            if name in inputs:
                return inputs[name]
    else:
        for name in _KNOWN_INTEGER_SPEAKER_INPUT_NAMES:
            if name in inputs:
                return inputs[name]

    for name in preferred_names:
        if name in inputs:
            return inputs[name]

    extras = [
        spec
        for name, spec in inputs.items()
        if name not in {text_name, text_lengths_name}
    ]
    if len(extras) == 1:
        return extras[0]
    if len(extras) > 1:
        raise ValueError(
            f"Unsupported ONNX signature with multiple extra inputs: {[spec.name for spec in extras]}"
        )
    return None


class VitsEvo:
    def __init__(
        self,
        model_path: str,
        *,
        config_path: Optional[str] = None,
        metadata_path: Optional[str] = None,
        sampling_rate: Optional[int] = None,
        add_blank: Optional[bool] = None,
        cleaned_text: Optional[bool] = None,
        text_cleaners: Optional[Iterable[str]] = None,
        n_speakers: Optional[int] = None,
        zero_shot_speakers: Optional[bool] = None,
        frontend: Optional[str] = None,
        phonemizer_checkpoint: Optional[str] = None,
        phonemizer_lang: Optional[str] = None,
        providers: Optional[Iterable[str]] = None,
        strict_symbols: bool = True,
    ):
        self.model_path = model_path
        self.providers = _normalize_providers(providers)
        self.session = ort.InferenceSession(model_path, providers=list(self.providers))
        self.config = load_inference_config(
            config_path=config_path,
            metadata_path=metadata_path,
            sampling_rate=sampling_rate,
            add_blank=add_blank,
            cleaned_text=cleaned_text,
            text_cleaners=text_cleaners,
            n_speakers=n_speakers,
            zero_shot_speakers=zero_shot_speakers,
            frontend=frontend,
            phonemizer_checkpoint=phonemizer_checkpoint,
            phonemizer_lang=phonemizer_lang,
        )
        self.tokenizer = Tokenizer(add_blank=self.config.add_blank, strict=strict_symbols)

        self.input_specs = {
            item.name: SessionInputSpec(name=item.name, type=item.type, shape=item.shape)
            for item in self.session.get_inputs()
        }
        self.output_name = self.session.get_outputs()[0].name
        self.text_input_name = _select_input_name(self.input_specs, _TEXT_INPUT_NAMES, 0)
        self.text_lengths_input_name = _select_input_name(self.input_specs, _TEXT_LENGTH_INPUT_NAMES, 1)
        self.speaker_input = _select_speaker_input(
            self.input_specs,
            self.text_input_name,
            self.text_lengths_input_name,
            prefer_embedding=self.config.zero_shot_speakers,
        )

        self.frontend_kind = self.config.frontend or (
            "deepphonemizer" if self.config.cleaned_text else "cleaners"
        )
        self._frontend = None

    @property
    def sampling_rate(self) -> int:
        return self.config.sampling_rate

    @property
    def speaker_mode(self) -> Optional[str]:
        if self.speaker_input is None:
            return None
        if self.speaker_input.is_integer:
            return "speaker_id"
        if self.speaker_input.is_float:
            return "speaker_embedding"
        return "unknown"

    def describe(self) -> Dict[str, object]:
        return {
            "model_path": self.model_path,
            "providers": list(self.providers),
            "sampling_rate": self.sampling_rate,
            "frontend": self.frontend_kind,
            "speaker_mode": self.speaker_mode,
            "speaker_input_name": None if self.speaker_input is None else self.speaker_input.name,
            "speaker_input_type": None if self.speaker_input is None else self.speaker_input.type,
            "inputs": {
                name: {"type": spec.type, "shape": list(spec.shape)}
                for name, spec in self.input_specs.items()
            },
        }

    def _get_frontend(self):
        if self._frontend is None:
            self._frontend = build_frontend(
                self.frontend_kind,
                cleaner_names=self.config.text_cleaners,
                phonemizer_checkpoint=self.config.phonemizer_checkpoint,
                phonemizer_lang=self.config.phonemizer_lang,
            )
        return self._frontend

    def phonemize(self, text: str) -> str:
        return self._get_frontend().phonemize(text)

    def encode_phonemes(self, phonemes: str):
        normalized = collapse_whitespace(phonemes)
        if not isinstance(normalized, str):
            raise TypeError("Expected a single phoneme string.")
        return self.tokenizer.encode_phonemes(cast(str, normalized))

    def _prepare_speaker_inputs(
        self,
        *,
        speaker_id: Optional[int] = None,
        speaker_embedding=None,
    ):
        provided = [
            speaker_id is not None,
            speaker_embedding is not None,
        ]
        if sum(provided) > 1:
            raise ValueError("Pass only one of speaker_id or speaker_embedding.")

        if self.speaker_input is None:
            if speaker_id is not None or speaker_embedding is not None:
                raise ValueError("This ONNX model does not expose a speaker input.")
            return {}

        speaker_mode = self.speaker_mode

        if speaker_mode == "speaker_id":
            if speaker_id is None:
                raise ValueError(
                    f"Model input '{self.speaker_input.name}' expects a speaker ID ({self.speaker_input.type})."
                )
            return {self.speaker_input.name: np.asarray([speaker_id], dtype=np.int64)}

        if speaker_mode == "speaker_embedding":
            if speaker_embedding is None:
                raise ValueError(
                    f"Model input '{self.speaker_input.name}' expects a speaker embedding ({self.speaker_input.type})."
                )
            embedding = np.asarray(speaker_embedding, dtype=np.float32)
            if embedding.ndim == 1:
                embedding = embedding[None, :]
            if embedding.ndim != 2 or embedding.shape[0] != 1:
                raise ValueError(
                    "speaker_embedding must have shape [channels] or [1, channels]."
                )
            return {self.speaker_input.name: embedding}

        raise ValueError(
            f"Unsupported speaker input type '{self.speaker_input.type}' for input '{self.speaker_input.name}'."
        )

    def prepare_inputs(
        self,
        *,
        text: Optional[str] = None,
        phonemes: Optional[str] = None,
        speaker_id: Optional[int] = None,
        speaker_embedding=None,
        return_phonemes: bool = False,
    ):
        if (text is None) == (phonemes is None):
            raise ValueError("Pass exactly one of text or phonemes.")

        if phonemes is not None:
            cleaned = phonemes
        else:
            if text is None:
                raise ValueError("text is required when phonemes are not provided.")
            cleaned = self.phonemize(cast(str, text))
        encoded = self.encode_phonemes(cleaned)

        inputs: Dict[str, object] = {
            self.text_input_name: np.asarray([encoded], dtype=np.int64),
            self.text_lengths_input_name: np.asarray([len(encoded)], dtype=np.int64),
        }

        speaker_inputs = self._prepare_speaker_inputs(
            speaker_id=speaker_id,
            speaker_embedding=speaker_embedding,
        )
        inputs.update(speaker_inputs)

        if return_phonemes:
            return inputs, cleaned
        return inputs

    def _run(self, inputs) -> np.ndarray:
        audio = self.session.run([self.output_name], inputs)[0]
        audio = np.asarray(audio, dtype=np.float32)
        audio = np.squeeze(audio)
        if audio.ndim != 1:
            raise ValueError(f"Expected mono waveform output, got shape {audio.shape}.")
        return audio

    def synthesize(
        self,
        text: str,
        *,
        speaker_id: Optional[int] = None,
        speaker_embedding=None,
        return_phonemes: bool = False,
    ):
        inputs, cleaned = self.prepare_inputs(
            text=text,
            speaker_id=speaker_id,
            speaker_embedding=speaker_embedding,
            return_phonemes=True,
        )
        audio = self._run(inputs)
        if return_phonemes:
            return audio, cleaned
        return audio

    def synthesize_phonemes(
        self,
        phonemes: str,
        *,
        speaker_id: Optional[int] = None,
        speaker_embedding=None,
    ) -> np.ndarray:
        inputs = self.prepare_inputs(
            phonemes=phonemes,
            speaker_id=speaker_id,
            speaker_embedding=speaker_embedding,
        )
        return self._run(inputs)

    def save_wav(self, path: str, audio) -> str:
        return str(save_wav(path, audio, self.sampling_rate))


MinimalOnnxTTS = VitsEvo
