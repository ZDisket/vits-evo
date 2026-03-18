from pathlib import Path
from typing import List, NamedTuple, Optional, Sequence, Union

import numpy as np

from .audio import preprocess_wav
from .hparams import mel_window_length, mel_window_step, partials_n_frames, sampling_rate


_N_FFT = int(sampling_rate * mel_window_length / 1000)
_HOP_LENGTH = int(sampling_rate * mel_window_step / 1000)
_PARTIAL_WAV_SAMPLES = partials_n_frames * _HOP_LENGTH
_PARTIAL_LEFT_CONTEXT = _HOP_LENGTH
_PARTIAL_RIGHT_CONTEXT = (_N_FFT // 2) % _HOP_LENGTH
_MODEL_INPUT_WAV_SAMPLES = _PARTIAL_LEFT_CONTEXT + _PARTIAL_WAV_SAMPLES + _PARTIAL_RIGHT_CONTEXT


class OnnxPreparedUtterance(NamedTuple):
    partials: np.ndarray
    wav_slices: List[slice]


class OnnxVoiceEncoderPreprocessor:
    def __init__(self, rate=1.3, min_coverage=0.75):
        self.rate = float(rate)
        self.min_coverage = float(min_coverage)
        self.model_input_num_samples = _MODEL_INPUT_WAV_SAMPLES
        self.partial_num_samples = _PARTIAL_WAV_SAMPLES

    @staticmethod
    def preprocess_wav(fpath_or_wav, source_sr=None):
        return preprocess_wav(fpath_or_wav, source_sr=source_sr)

    @staticmethod
    def compute_partial_slices(n_samples: int, rate, min_coverage):
        assert 0 < min_coverage <= 1

        n_frames = int(np.ceil((n_samples + 1) / _HOP_LENGTH))
        frame_step = int(np.round((sampling_rate / rate) / _HOP_LENGTH))
        assert 0 < frame_step, "The rate is too high"
        assert frame_step <= partials_n_frames, "The rate is too low, it should be %f at least" % \
            (sampling_rate / (_HOP_LENGTH * partials_n_frames))

        wav_slices = []
        steps = max(1, n_frames - partials_n_frames + frame_step + 1)
        for i in range(0, steps, frame_step):
            wav_range = np.array([i, i + partials_n_frames]) * _HOP_LENGTH
            wav_slices.append(slice(*wav_range))

        last_wav_range = wav_slices[-1]
        coverage = (n_samples - last_wav_range.start) / (last_wav_range.stop - last_wav_range.start)
        if coverage < min_coverage and len(wav_slices) > 1:
            wav_slices = wav_slices[:-1]

        return wav_slices

    def _prepare_wav(self, fpath_or_wav, source_sr=None, preprocess=None):
        if preprocess is None:
            preprocess = source_sr is not None or isinstance(fpath_or_wav, (str, Path))

        if preprocess:
            wav = self.preprocess_wav(fpath_or_wav, source_sr=source_sr)
        else:
            wav = fpath_or_wav

        return np.asarray(wav, dtype=np.float32).reshape(-1)

    def _coerce_wavs(self, wavs, source_srs=None, preprocess=None, lengths=None):
        if isinstance(wavs, np.ndarray):
            wavs = np.asarray(wavs, dtype=np.float32)
            if wavs.ndim == 1:
                prepared = [wavs.reshape(-1)]
            elif wavs.ndim == 2:
                if lengths is None:
                    lengths = np.full(wavs.shape[0], wavs.shape[1], dtype=np.int64)
                else:
                    lengths = np.asarray(lengths, dtype=np.int64).reshape(-1)
                    if lengths.shape[0] != wavs.shape[0]:
                        raise ValueError("Lengths must have one entry per waveform")
                prepared = [wavs[index, :int(length)] for index, length in enumerate(lengths)]
            else:
                raise ValueError("Expected a 1D or 2D numpy array of waveforms")

            if source_srs is not None or preprocess:
                raise ValueError("Batch numpy arrays only support preprocess=False and no source_srs")
            return [np.asarray(wav, dtype=np.float32).reshape(-1) for wav in prepared]

        wavs = list(wavs)
        if not wavs:
            raise ValueError("Expected at least one waveform")

        if source_srs is None:
            source_srs = [None] * len(wavs)
        elif np.isscalar(source_srs):
            source_srs = [source_srs] * len(wavs)
        else:
            source_srs = list(source_srs)
            if len(source_srs) != len(wavs):
                raise ValueError("source_srs must have one entry per waveform")

        return [
            self._prepare_wav(wav, source_sr=source_sr, preprocess=preprocess)
            for wav, source_sr in zip(wavs, source_srs)
        ]

    def _split_utterance(self, wav: np.ndarray):
        wav_slices = self.compute_partial_slices(len(wav), self.rate, self.min_coverage)
        max_wave_length = wav_slices[-1].stop
        if max_wave_length >= len(wav):
            wav = np.pad(wav, (0, max_wave_length - len(wav)), "constant")

        partials = []
        for wav_slice in wav_slices:
            start = wav_slice.start - _PARTIAL_LEFT_CONTEXT
            stop = wav_slice.stop + _PARTIAL_RIGHT_CONTEXT
            left_pad = max(0, -start)
            right_pad = max(0, stop - len(wav))
            part = wav[max(0, start):min(len(wav), stop)]
            if left_pad or right_pad:
                part = np.pad(part, (left_pad, right_pad), "constant")
            partials.append(part)

        partials = np.stack(partials, axis=0).astype(np.float32, copy=False)
        if partials.shape[1] != _MODEL_INPUT_WAV_SAMPLES:
            raise RuntimeError("Unexpected partial size %s" % (partials.shape[1],))
        return OnnxPreparedUtterance(partials=partials, wav_slices=wav_slices)

    def prepare_utterance(self, wav, source_sr=None, preprocess=None):
        prepared_wav = self._prepare_wav(wav, source_sr=source_sr, preprocess=preprocess)
        return self._split_utterance(prepared_wav)

    def prepare_utterances(self, wavs, source_srs=None, preprocess=None, lengths=None):
        prepared_wavs = self._coerce_wavs(
            wavs,
            source_srs=source_srs,
            preprocess=preprocess,
            lengths=lengths,
        )
        return [self._split_utterance(wav) for wav in prepared_wavs]

    @staticmethod
    def collate_prepared_utterances(prepared_utterances: Sequence[OnnxPreparedUtterance]):
        prepared_utterances = list(prepared_utterances)
        if not prepared_utterances:
            raise ValueError("Expected at least one prepared utterance")

        counts = np.array([prepared.partials.shape[0] for prepared in prepared_utterances], dtype=np.int64)
        flat_partials = np.concatenate([prepared.partials for prepared in prepared_utterances], axis=0)
        return flat_partials.astype(np.float32, copy=False), counts

    @staticmethod
    def split_partial_embeddings(partial_embeddings: np.ndarray, counts):
        counts = np.asarray(counts, dtype=np.int64).reshape(-1)
        groups = []
        offset = 0
        for count in counts:
            groups.append(partial_embeddings[offset:offset + count])
            offset += count
        return groups

    @staticmethod
    def aggregate_partial_embeddings(partial_embeddings: np.ndarray, counts):
        utterance_embeddings = []
        for partial_group in OnnxVoiceEncoderPreprocessor.split_partial_embeddings(partial_embeddings, counts):
            raw_embed = partial_group.mean(axis=0)
            utterance_embeddings.append(raw_embed / np.linalg.norm(raw_embed, 2))
        return np.asarray(utterance_embeddings, dtype=np.float32)
