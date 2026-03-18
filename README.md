# VITS EVOlution

https://private-user-images.githubusercontent.com/30500847/565807180-12530801-518e-4879-b28a-00048867d189.mp4?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NzM4NjIzODQsIm5iZiI6MTc3Mzg2MjA4NCwicGF0aCI6Ii8zMDUwMDg0Ny81NjU4MDcxODAtMTI1MzA4MDEtNTE4ZS00ODc5LWIyOGEtMDAwNDg4NjdkMTg5Lm1wND9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNjAzMTglMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjYwMzE4VDE5MjgwNFomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTIyYzUwODQ4Y2RlZmIyYmI3YmNmNjViNzViM2JlMDkyZDhlYTJiNDE1YmJhMGI0NjdhZWY1ZmQ2OTdmODgxZTImWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.BvM_SKed4rGrLIaJZ0F-LGxQvs1OlREgXPg2bfxcUg4

See more samples on my [Xitter post](https://x.com/ZDi____/status/2034366015652667596)

VITS EVOlution is an open source text to speech stack model around zero-shot voice cloning and low latency. It includes an ONNX speaker encoder, TTS inference, uses DeepPhonemizer for phonemes (MIT), and a voice blending option that averages multiple speaker embeddings to create new voices.

## Features

- Zero-shot voice cloning from a reference clip
- Voice blending with two or more reference embeddings
- ONNX release format for both the speaker encoder and the TTS model
- Permissive licenses for all the stack
- CPU inference at about `0.18` real-time factor on an `Intel(R) Xeon(R) Platinum 8470`, or about `5.6x` faster than real time


## Released models

| Model | Type | Checkpoint | Hugging Face demo | Colab demo |
| --- | --- | --- | --- | --- |
| `vits-evo-zero-shot-v1` | Zero-shot TTS (ONNX) | [Google Drive](https://drive.google.com/file/d/1pzLZCm2rr9kdS6GSBlOckFdE2pdQTL3W/view?usp=sharing) | [HF Space](https://huggingface.co/spaces/ZDisket/vits-evo-zs1) | [Google Colab](https://drive.google.com/file/d/1Dx19cyRUQ3eTnsWEbchg8pNc81uW9GkK/view?usp=sharing) |

## Minimal run

Clone DeepPhonemizer and install it:

```bash
git clone https://github.com/ZDisket/DeepPhonemizer.git
pip install ./DeepPhonemizer
```

After downloading the TTS model (duh!), download the English phonemizer checkpoint:

```bash
wget https://public-asai-dl-models.s3.eu-central-1.amazonaws.com/DeepPhonemizer/en_us_cmudict_ipa_forward.pt
```


```python
from onnx_tts import VitsEvo
from resemblyzer_onnx import OnnxVoiceEncoder

tts = VitsEvo(
    "path/to/model.onnx",
    config_path="configs/zero_shot_pretrain_nosdp_1gpu.inference.json",
    phonemizer_checkpoint="en_us_cmudict_ipa_forward.pt",
)

speaker_encoder = OnnxVoiceEncoder(device="cpu")
embedding = speaker_encoder.make_embedding("reference.wav")

audio = tts.synthesize(
    "Hello from VITS EVOlution.",
    speaker_embedding=embedding,
)

tts.save_wav("output.wav", audio)
```

## Voice blending

Voice blending averages two or more speaker embeddings to create a new voice.

```python
embedding = speaker_encoder.make_embedding(
    ["speaker_a.wav", "speaker_b.wav", "speaker_c.wav"],
    factors=[1.0, 0.7, 0.4],
)

audio = tts.synthesize(
    "This voice is blended from multiple references.",
    speaker_embedding=embedding,
)
```

## Gradio app

Run the included demo app:

```bash
python gradio_zero_shot.py --model path/to/model.onnx --config configs/zero_shot_pretrain_nosdp_1gpu.inference.json --phonemizer-checkpoint en_us_cmudict_ipa_forward.pt
```

The app lets you:

- upload a primary reference clip
- mix extra reference clips with custom weights
- inspect phoneme output
- test zero-shot cloning and blended voices from the browser

## Contact
Hey you? Like [my](https://zdisket.github.io/page.html) stuff? [email me](mailto:nika109021@gmail.com) or [DM me on Xitter](https://x.com/ZDi____) for any inquiries you may have.
