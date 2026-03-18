from pathlib import Path
from time import perf_counter as timer
from typing import Optional, Sequence, Tuple, cast, Union

import numpy as np

from .export import DEFAULT_ONNX_WEIGHTS_FPATH
from .preprocessing import OnnxPreparedUtterance, OnnxVoiceEncoderPreprocessor


def _resolve_runtime_device(providers):
    primary = providers[0]
    if primary == "CUDAExecutionProvider":
        return "cuda"
    if primary == "ROCMExecutionProvider":
        return "rocm"
    return "cpu"


def _is_audio_sr_pair(item):
    return (
        isinstance(item, (tuple, list))
        and len(item) == 2
        and not isinstance(item[0], (str, Path))
        and np.isscalar(item[1])
    )


def _coerce_embedding_sources(audio):
    if isinstance(audio, (str, Path)) or _is_audio_sr_pair(audio):
        return [audio]
    if isinstance(audio, (tuple, list)):
        sources = list(audio)
        if not sources:
            raise ValueError("Expected at least one audio source")
        return sources
    raise TypeError(
        "audio must be a file path, an (audio_array, sampling_rate) tuple, "
        "or a non-empty list of those sources"
    )


def _normalize_factors(factors, num_sources):
    if factors is None:
        return np.full((num_sources,), 1.0 / float(num_sources), dtype=np.float32)

    weights = np.asarray(factors, dtype=np.float32).reshape(-1)
    if weights.shape[0] != num_sources:
        raise ValueError(f"Expected {num_sources} mixing factors, got {weights.shape[0]}")
    if not np.isfinite(weights).all():
        raise ValueError("Mixing factors must be finite")
    if (weights < 0).any():
        raise ValueError("Mixing factors must be non-negative")

    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("Mixing factors must sum to a positive value")
    return weights / total


class OnnxVoiceEncoderInference:
    def __init__(self, device: Optional[Union[str, Path]] = None, verbose=True,
                 weights_fpath: Optional[Union[str, Path]] = None,
                 providers: Optional[Sequence[str]] = None, session_options=None):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required to run ONNX voice encoder inference. "
                "Install onnxruntime or onnxruntime-gpu."
            ) from exc

        if weights_fpath is None and isinstance(device, (str, Path)) and str(device) not in {"cpu", "cuda"}:
            weights_fpath = device
            device = None
        self.onnx_fpath = Path(weights_fpath) if weights_fpath is not None else DEFAULT_ONNX_WEIGHTS_FPATH
        if not self.onnx_fpath.exists():
            raise FileNotFoundError("Couldn't find the ONNX voice encoder model at %s." % self.onnx_fpath)

        available_providers = ort.get_available_providers()
        if providers is None:
            if device is None:
                providers = ["CUDAExecutionProvider", "ROCMExecutionProvider", "CPUExecutionProvider"]
            elif device == "cpu":
                providers = ["CPUExecutionProvider"]
            elif device == "cuda":
                providers = ["CUDAExecutionProvider", "ROCMExecutionProvider", "CPUExecutionProvider"]
            elif device == "rocm":
                providers = ["ROCMExecutionProvider", "CPUExecutionProvider"]
            else:
                raise ValueError("Unsupported device %r. Use 'cpu', 'cuda', 'rocm' or explicit providers." % device)

        self.providers = [provider for provider in providers if provider in available_providers]
        if not self.providers:
            raise RuntimeError(
                "None of the requested ONNX Runtime providers are available. "
                "Requested=%s available=%s" % (list(providers), available_providers)
            )

        start = timer()
        self.session = ort.InferenceSession(
            str(self.onnx_fpath),
            providers=self.providers,
            sess_options=session_options,
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.device = _resolve_runtime_device(self.providers)

        if verbose:
            print("Loaded the ONNX voice encoder model on %s in %.2f seconds." %
                  (self.device, timer() - start))

    def embed_partials(self, waveforms, batch_size=None):
        waveforms = np.asarray(waveforms, dtype=np.float32)
        if waveforms.ndim == 1:
            waveforms = waveforms.reshape(1, -1)
        if waveforms.ndim != 2:
            raise ValueError("Expected partial waveforms with shape (batch, num_samples)")

        if batch_size is None or batch_size <= 0:
            return self.session.run([self.output_name], {self.input_name: waveforms})[0]

        outputs = []
        for start in range(0, waveforms.shape[0], batch_size):
            batch = waveforms[start:start + batch_size]
            outputs.append(self.session.run([self.output_name], {self.input_name: batch})[0])
        return np.concatenate(outputs, axis=0)


class OnnxVoiceEncoder:
    def __init__(self, device: Optional[Union[str, Path]] = None, verbose=True,
                 weights_fpath: Optional[Union[str, Path]] = None,
                 providers: Optional[Sequence[str]] = None, session_options=None,
                 rate=1.3, min_coverage=0.75):
        self.preprocessor = OnnxVoiceEncoderPreprocessor(rate=rate, min_coverage=min_coverage)
        self.inference = OnnxVoiceEncoderInference(
            device=device,
            verbose=verbose,
            weights_fpath=weights_fpath,
            providers=providers,
            session_options=session_options,
        )

        self.rate = self.preprocessor.rate
        self.min_coverage = self.preprocessor.min_coverage
        self.device = self.inference.device
        self.onnx_fpath = self.inference.onnx_fpath
        self.providers = self.inference.providers

    @staticmethod
    def preprocess_wav(fpath_or_wav, source_sr=None):
        return OnnxVoiceEncoderPreprocessor.preprocess_wav(fpath_or_wav, source_sr=source_sr)

    @staticmethod
    def compute_partial_slices(n_samples: int, rate, min_coverage):
        return OnnxVoiceEncoderPreprocessor.compute_partial_slices(n_samples, rate, min_coverage)

    def prepare_utterance(self, wav, source_sr=None, preprocess=None):
        return self.preprocessor.prepare_utterance(wav, source_sr=source_sr, preprocess=preprocess)

    def prepare_utterances(self, wavs, source_srs=None, preprocess=None, lengths=None):
        return self.preprocessor.prepare_utterances(
            wavs,
            source_srs=source_srs,
            preprocess=preprocess,
            lengths=lengths,
        )

    def embed_prepared_utterance(self, prepared: OnnxPreparedUtterance, return_partials=False,
                                 batch_size=None):
        partial_embeds = self.inference.embed_partials(prepared.partials, batch_size=batch_size)
        embed = self.preprocessor.aggregate_partial_embeddings(
            partial_embeds,
            [prepared.partials.shape[0]],
        )[0]
        if return_partials:
            return embed.astype(np.float32), partial_embeds.astype(np.float32), prepared.wav_slices
        return embed.astype(np.float32)

    def embed_prepared_utterances(self, prepared_utterances, batch_size=None, return_partials=False):
        prepared_utterances = list(prepared_utterances)
        flat_partials, counts = self.preprocessor.collate_prepared_utterances(prepared_utterances)
        partial_embeddings = self.inference.embed_partials(flat_partials, batch_size=batch_size)
        utterance_embeddings = self.preprocessor.aggregate_partial_embeddings(partial_embeddings, counts)
        if return_partials:
            split_partials = self.preprocessor.split_partial_embeddings(partial_embeddings, counts)
            return utterance_embeddings, split_partials
        return utterance_embeddings

    def embed_utterances(self, wavs, source_srs=None, preprocess=None, lengths=None,
                         rate=1.3, min_coverage=0.75, batch_size=None):
        if not np.isclose(rate, self.rate) or not np.isclose(min_coverage, self.min_coverage):
            raise ValueError(
                "This encoder was initialized with fixed rate=%s and min_coverage=%s" %
                (self.rate, self.min_coverage)
            )

        prepared_utterances = self.prepare_utterances(
            wavs,
            source_srs=source_srs,
            preprocess=preprocess,
            lengths=lengths,
        )
        return self.embed_prepared_utterances(prepared_utterances, batch_size=batch_size)

    def embed_utterance(self, wav, return_partials=False, rate=1.3, min_coverage=0.75,
                        source_sr=None, preprocess=None, batch_size=None):
        if not np.isclose(rate, self.rate) or not np.isclose(min_coverage, self.min_coverage):
            raise ValueError(
                "This encoder was initialized with fixed rate=%s and min_coverage=%s" %
                (self.rate, self.min_coverage)
            )

        prepared = self.prepare_utterance(wav, source_sr=source_sr, preprocess=preprocess)
        return self.embed_prepared_utterance(prepared, return_partials=return_partials,
                                             batch_size=batch_size)

    def embed_speaker(self, wavs, **kwargs):
        embeds = np.asarray(self.embed_utterances(wavs, **kwargs), dtype=np.float32)
        raw_embed = embeds.mean(axis=0)
        return raw_embed / np.linalg.norm(raw_embed, 2)

    def make_embedding(self, audio, factors=None, batch_size=None, rate=1.3, min_coverage=0.75):
        sources = _coerce_embedding_sources(audio)
        weights = _normalize_factors(factors, len(sources))

        embeds = []
        for source in sources:
            if _is_audio_sr_pair(source) and isinstance(source, (tuple, list)):
                wav, source_sr = cast(Tuple[np.ndarray, int], tuple(source))
                embed = self.embed_utterance(
                    wav,
                    source_sr=source_sr,
                    preprocess=True,
                    batch_size=batch_size,
                    rate=rate,
                    min_coverage=min_coverage,
                )
            else:
                embed = self.embed_utterance(
                    source,
                    batch_size=batch_size,
                    rate=rate,
                    min_coverage=min_coverage,
                )
            embeds.append(np.asarray(embed, dtype=np.float32))

        embeds = np.asarray(embeds, dtype=np.float32)
        mixed = np.sum(embeds * weights[:, None], axis=0, dtype=np.float32)
        norm = float(np.linalg.norm(mixed, ord=2))
        if norm <= 0.0 or not np.isfinite(norm):
            raise ValueError("Interpolated speaker embedding has invalid norm")
        return mixed / norm
