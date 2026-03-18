# Resemblyzer ONNX


## Install

```bash
pip install -r requirements.txt
```

## Standalone usage

```python
from pathlib import Path

from resemblyzer_onnx import OnnxVoiceEncoder

encoder = OnnxVoiceEncoder()
embed = encoder.embed_utterance(Path("path/to/audio.wav"))
print(embed.shape)
```

For explicit pipeline control, split preprocessing and inference:

```python
from pathlib import Path

from resemblyzer_onnx import OnnxVoiceEncoderInference, OnnxVoiceEncoderPreprocessor

preprocessor = OnnxVoiceEncoderPreprocessor()
inference = OnnxVoiceEncoderInference(device="cuda")

prepared = preprocessor.prepare_utterance(Path("path/to/audio.wav"))
partial_embeds = inference.embed_partials(prepared.partials)
embed = preprocessor.aggregate_partial_embeddings(partial_embeds, [prepared.partials.shape[0]])[0]
```

For batched inference:

```python
embeds = encoder.embed_utterances([
    Path("audio_1.wav"),
    Path("audio_2.wav"),
])
print(embeds.shape)
```

For inference-time speaker blending, use `make_embedding` with one source or a weighted mix:

```python
embed = encoder.make_embedding(Path("speaker.wav"))

mixed = encoder.make_embedding(
    [
        Path("speaker_a.wav"),
        (wav_b, sr_b),
    ],
    factors=[0.7, 0.3],
)
```

If you already have a preprocessed waveform at 16 kHz, skip preprocessing:

```python
embed = encoder.embed_utterance(wav, preprocess=False)
```

If you already have a padded batch tensor, pass lengths explicitly:

```python
embeds = encoder.embed_utterances(batch_wavs, lengths=batch_lengths, preprocess=False)
```

The wrapper keeps silence trimming outside the ONNX graph but applies it automatically for file paths and for waveforms passed with `source_sr=...`.

Device selection accepts `device="cpu"`, `device="cuda"`, or `device="rocm"`. You can also pass an explicit ONNX Runtime `providers=[...]` list.

The exported ONNX checkpoint itself is a per-partial batch encoder. The Python wrapper preserves the original Resemblyzer utterance behavior by splitting long utterances into 1.6 s partials, batching those partials through ONNX Runtime, averaging the partial embeddings, and normalizing the result.

For dataset-scale processing with many CPU preprocess workers feeding one GPU worker, use the example module:

```bash
python -m resemblyzer_onnx.example_dataset_processing /path/to/audio dataset_embeds.npz --glob "*.flac"
```
