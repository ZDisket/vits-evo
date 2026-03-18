import argparse

import numpy as np

from onnx_tts import VitsEvo


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal ONNX TTS inference example")
    parser.add_argument("--model", required=True, help="Path to the ONNX model")
    parser.add_argument("--config", help="Path to the training config JSON")
    parser.add_argument("--metadata", help="Optional inference metadata JSON")
    parser.add_argument("--output", default="output.wav", help="Output WAV path")
    parser.add_argument("--frontend", choices=["deepphonemizer", "cleaners", "prephonemized"])
    parser.add_argument("--phonemizer-checkpoint", help="DeepPhonemizer checkpoint path")
    parser.add_argument("--phonemizer-lang", help="DeepPhonemizer language code")

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--text", help="Raw input text")
    input_group.add_argument("--phonemes", help="Already-cleaned phoneme string")

    speaker_group = parser.add_mutually_exclusive_group()
    speaker_group.add_argument("--speaker-id", type=int, help="Integer speaker ID for multi-speaker exports")
    speaker_group.add_argument(
        "--speaker-embedding",
        help="Path to a .npy file with shape [channels] or [1, channels]",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    speaker_embedding = None
    if args.speaker_embedding:
        speaker_embedding = np.load(args.speaker_embedding)

    tts = VitsEvo(
        args.model,
        config_path=args.config,
        metadata_path=args.metadata,
        frontend=args.frontend,
        phonemizer_checkpoint=args.phonemizer_checkpoint,
        phonemizer_lang=args.phonemizer_lang,
    )

    if args.phonemes is not None:
        audio = tts.synthesize_phonemes(
            args.phonemes,
            speaker_id=args.speaker_id,
            speaker_embedding=speaker_embedding,
        )
    else:
        audio = tts.synthesize(
            args.text,
            speaker_id=args.speaker_id,
            speaker_embedding=speaker_embedding,
        )

    output_path = tts.save_wav(args.output, audio)
    print(f"Saved audio to {output_path}")
    print(tts.describe())


if __name__ == "__main__":
    main()
