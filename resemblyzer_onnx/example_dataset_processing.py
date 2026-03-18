import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from resemblyzer_onnx.inference import OnnxVoiceEncoderInference
from resemblyzer_onnx.preprocessing import OnnxVoiceEncoderPreprocessor


_WORKER_PREPROCESSOR = None


def _init_preprocessor_worker(rate, min_coverage):
    global _WORKER_PREPROCESSOR
    _WORKER_PREPROCESSOR = OnnxVoiceEncoderPreprocessor(rate=rate, min_coverage=min_coverage)


def _prepare_file(task):
    index, file_path = task
    prepared = _WORKER_PREPROCESSOR.prepare_utterance(Path(file_path))
    return index, file_path, prepared


def _flush_prepared(buffered_items, inference, onnx_batch_size):
    indices = [index for index, _, _ in buffered_items]
    prepared_utterances = [prepared for _, _, prepared in buffered_items]
    flat_partials, counts = OnnxVoiceEncoderPreprocessor.collate_prepared_utterances(prepared_utterances)
    partial_embeddings = inference.embed_partials(flat_partials, batch_size=onnx_batch_size)
    utterance_embeddings = OnnxVoiceEncoderPreprocessor.aggregate_partial_embeddings(
        partial_embeddings,
        counts,
    )
    return indices, utterance_embeddings


def _collect_files(input_dir: Path, pattern: str):
    files = sorted(path for path in input_dir.rglob(pattern) if path.is_file())
    if not files:
        raise FileNotFoundError("No files matched %r under %s" % (pattern, input_dir))
    return files


def process_dataset(file_paths, output_fpath: Path, preprocess_workers=4, partial_buffer_size=512,
                    onnx_batch_size=256, device="cuda", weights_fpath=None,
                    rate=1.3, min_coverage=0.75):
    inference = OnnxVoiceEncoderInference(device=device, weights_fpath=weights_fpath)
    embeddings = np.zeros((len(file_paths), 256), dtype=np.float32)
    buffered_items = []
    buffered_partials = 0
    processed = 0

    with ProcessPoolExecutor(
        max_workers=preprocess_workers,
        initializer=_init_preprocessor_worker,
        initargs=(rate, min_coverage),
    ) as executor:
        futures = [executor.submit(_prepare_file, (index, str(path))) for index, path in enumerate(file_paths)]
        for future in as_completed(futures):
            index, _, prepared = future.result()
            buffered_items.append((index, file_paths[index], prepared))
            buffered_partials += prepared.partials.shape[0]

            if buffered_partials >= partial_buffer_size:
                indices, batch_embeddings = _flush_prepared(buffered_items, inference, onnx_batch_size)
                for item_index, embedding in zip(indices, batch_embeddings):
                    embeddings[item_index] = embedding
                processed += len(indices)
                print("Processed %d/%d files" % (processed, len(file_paths)))
                buffered_items = []
                buffered_partials = 0

    if buffered_items:
        indices, batch_embeddings = _flush_prepared(buffered_items, inference, onnx_batch_size)
        for item_index, embedding in zip(indices, batch_embeddings):
            embeddings[item_index] = embedding
        processed += len(indices)
        print("Processed %d/%d files" % (processed, len(file_paths)))

    output_fpath.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_fpath,
        paths=np.asarray([str(path) for path in file_paths]),
        embeddings=embeddings,
    )
    print("Saved embeddings to %s" % output_fpath)


def main():
    parser = argparse.ArgumentParser(
        description="Example multi-worker dataset embedding pipeline for the standalone ONNX encoder.",
    )
    parser.add_argument("input_dir", type=Path, help="Directory containing audio files")
    parser.add_argument("output", type=Path, help="Output .npz file")
    parser.add_argument("--glob", default="*.wav", help="Recursive glob pattern to match audio files")
    parser.add_argument("--preprocess-workers", type=int, default=4,
                        help="Number of CPU workers for preprocessing")
    parser.add_argument("--partial-buffer-size", type=int, default=512,
                        help="Flush to ONNX when this many partials are ready")
    parser.add_argument("--onnx-batch-size", type=int, default=256,
                        help="Maximum number of partials per ONNX Runtime call")
    parser.add_argument("--device", default="cuda", help="ONNX Runtime device preference: cpu, cuda, or rocm")
    parser.add_argument("--weights", type=Path, default=None, help="Optional path to an ONNX checkpoint")
    parser.add_argument("--rate", type=float, default=1.3, help="Partial rate parameter")
    parser.add_argument("--min-coverage", type=float, default=0.75,
                        help="Minimum final partial coverage")
    args = parser.parse_args()

    file_paths = _collect_files(args.input_dir, args.glob)
    process_dataset(
        file_paths=file_paths,
        output_fpath=args.output,
        preprocess_workers=args.preprocess_workers,
        partial_buffer_size=args.partial_buffer_size,
        onnx_batch_size=args.onnx_batch_size,
        device=args.device,
        weights_fpath=args.weights,
        rate=args.rate,
        min_coverage=args.min_coverage,
    )


if __name__ == "__main__":
    main()
