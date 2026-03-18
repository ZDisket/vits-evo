import argparse
import time
from typing import List

from onnx_tts import VitsEvo
from resemblyzer_onnx import OnnxVoiceEncoder


def _collect_voice_blend_sources(primary_audio, blend_enabled, blend_audio_2, blend_weight_2, blend_audio_3, blend_weight_3):
    if primary_audio is None:
        raise ValueError("Please provide a reference audio file.")

    sources: List[str] = []
    factors: List[float] = []

    sources.append(primary_audio)
    factors.append(1.0)

    if blend_enabled:
        if blend_audio_2 is not None:
            sources.append(blend_audio_2)
            factors.append(float(blend_weight_2))
        if blend_audio_3 is not None:
            sources.append(blend_audio_3)
            factors.append(float(blend_weight_3))

    return sources, factors


def _normalize_mix_factors(factors: List[float]) -> List[float]:
    total = float(sum(factors))
    if total <= 0.0:
        raise ValueError("Voice blending weights must sum to a positive value.")
    return [float(factor) / total for factor in factors]


def _voice_blending_summary(primary_audio, blend_enabled, blend_audio_2, blend_weight_2, blend_audio_3, blend_weight_3):
    if primary_audio is None:
        return "Voice Blending: waiting for primary reference audio."

    labels = ["Primary"]
    factors = [1.0]

    if blend_enabled:
        if blend_audio_2 is not None:
            labels.append("Blend 2")
            factors.append(float(blend_weight_2))
        if blend_audio_3 is not None:
            labels.append("Blend 3")
            factors.append(float(blend_weight_3))

    normalized = _normalize_mix_factors(factors)
    parts = [f"{label}: {weight:.1%}" for label, weight in zip(labels, normalized)]
    if len(parts) == 1:
        return f"Voice Blending: single reference active ({parts[0]})."
    return "Voice Blending: " + ", ".join(parts)


def build_app(args):
    import gradio as gr

    tts = VitsEvo(
        args.model,
        config_path=args.config,
        metadata_path=args.metadata,
        frontend=args.frontend,
        phonemizer_checkpoint=args.phonemizer_checkpoint,
        phonemizer_lang=args.phonemizer_lang,
        providers=args.tts_provider,
    )
    speaker_encoder = OnnxVoiceEncoder(
        device=args.embedder_device,
        providers=args.embedder_provider,
        weights_fpath=args.embedder_weights,
        verbose=True,
    )

    def synthesize(text: str, reference_audio, voice_blending: bool, blend_audio_2, blend_weight_2: float, blend_audio_3, blend_weight_3: float):
        if not text or not text.strip():
            raise gr.Error("Please enter text to synthesize.")

        try:
            start_time = time.perf_counter()
            sources, factors = _collect_voice_blend_sources(
                reference_audio,
                voice_blending,
                blend_audio_2,
                blend_weight_2,
                blend_audio_3,
                blend_weight_3,
            )
            embedding = speaker_encoder.make_embedding(sources, factors=factors)
            audio, phonemes = tts.synthesize(
                text.strip(),
                speaker_embedding=embedding,
                return_phonemes=True,
            )
            elapsed = max(time.perf_counter() - start_time, 1e-8)
        except Exception as exc:
            raise gr.Error(str(exc)) from exc

        audio_seconds = float(len(audio) / float(tts.sampling_rate)) if len(audio) else 0.0
        rtf = elapsed / max(audio_seconds, 1e-8)
        provider = tts.providers[0] if tts.providers else "unknown"
        status = (
            f"Made {audio_seconds:.2f} seconds of audio in {elapsed:.2f} seconds on "
            f"{provider}, RTF: {rtf:.4f}"
        )

        return (tts.sampling_rate, audio), phonemes, tts.describe(), status

    with gr.Blocks(title="Zero-Shot ONNX TTS") as demo:
        gr.Markdown("# Zero-Shot ONNX TTS")
        gr.Markdown(
            "Upload a reference voice sample, type text, and synthesize speech "
            "using the ONNX zero-shot model. Enable Voice Blending to mix multiple references."
        )

        with gr.Row():
            with gr.Column(scale=2):
                text = gr.Textbox(label="Text", lines=5, placeholder="Enter text to synthesize")
                reference_audio = gr.Audio(label="Primary Reference Audio", type="filepath", sources=["upload", "microphone"])
                voice_blending = gr.Checkbox(label="Voice Blending", value=False)
                voice_blending_summary = gr.Markdown("Voice Blending: waiting for primary reference audio.")
                with gr.Accordion("Voice Blending", open=False):
                    gr.Markdown("Optional extra references. Their weights are mixed with the primary reference.")
                    blend_audio_2 = gr.Audio(
                        label="Blend Reference 2",
                        type="filepath",
                        sources=["upload", "microphone"],
                    )
                    blend_weight_2 = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=1.0,
                        step=0.05,
                        label="Blend Weight 2",
                    )
                    blend_audio_3 = gr.Audio(
                        label="Blend Reference 3",
                        type="filepath",
                        sources=["upload", "microphone"],
                    )
                    blend_weight_3 = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=1.0,
                        step=0.05,
                        label="Blend Weight 3",
                    )
                synth_button = gr.Button("Synthesize", variant="primary")

            with gr.Column(scale=2):
                output_audio = gr.Audio(label="Synthesized Audio")
                phonemes = gr.Textbox(label="Phonemes", lines=4)
                status = gr.Textbox(label="Status", value="Ready", interactive=False)
                model_info = gr.JSON(label="Runtime Info")

        summary_inputs = [
            reference_audio,
            voice_blending,
            blend_audio_2,
            blend_weight_2,
            blend_audio_3,
            blend_weight_3,
        ]

        for component in summary_inputs:
            component.change(
                fn=_voice_blending_summary,
                inputs=summary_inputs,
                outputs=voice_blending_summary,
            )

        synth_button.click(
            fn=synthesize,
            inputs=[
                text,
                reference_audio,
                voice_blending,
                blend_audio_2,
                blend_weight_2,
                blend_audio_3,
                blend_weight_3,
            ],
            outputs=[output_audio, phonemes, model_info, status],
        )

    return demo


def parse_args():
    parser = argparse.ArgumentParser(description="Gradio app for zero-shot ONNX TTS")
    parser.add_argument("--model", required=True, help="Path to the zero-shot ONNX TTS model")
    parser.add_argument("--config", help="Path to the inference config JSON")
    parser.add_argument("--metadata", help="Optional inference metadata JSON")
    parser.add_argument("--frontend", choices=["deepphonemizer", "cleaners", "prephonemized"])
    parser.add_argument("--phonemizer-checkpoint", help="DeepPhonemizer checkpoint path")
    parser.add_argument("--phonemizer-lang", help="DeepPhonemizer language code")
    parser.add_argument(
        "--tts-provider",
        action="append",
        default=None,
        help="ONNX Runtime provider for TTS. Can be passed multiple times.",
    )
    parser.add_argument("--embedder-device", default="cuda", help="Speaker embedder device: cpu, cuda, or rocm")
    parser.add_argument(
        "--embedder-provider",
        action="append",
        default=None,
        help="Explicit ONNX Runtime provider for the speaker embedder. Can be passed multiple times.",
    )
    parser.add_argument("--embedder-weights", help="Optional custom speaker embedder ONNX weights")
    parser.add_argument("--host", default="127.0.0.1", help="Gradio bind host")
    parser.add_argument("--port", type=int, default=7860, help="Gradio bind port")
    parser.add_argument("--share", action="store_true", help="Enable Gradio share mode")
    return parser.parse_args()


def main():
    args = parse_args()
    app = build_app(args)
    app.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
