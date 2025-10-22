"""
Real-time audio inference optimized for low latency.

This script processes audio in small blocks (mini-batches) to simulate
real-time streaming with minimal latency while maintaining good throughput.
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset import (
    compute_context_length,
    denormalize_from_unit_range,
    normalize_to_unit_range,
)
from main import ModelMetadata, coerce_dilations  # noqa: F401 - ensure layer registration
from tcn import TCN  # noqa: F401 - required to deserialize custom layer

LOGGER = logging.getLogger(__name__)

TASK_ROOT = Path(__file__).resolve().parent.parent
SAVED_MODELS_DIR = TASK_ROOT / "savedModels"
CONFIGS_DIR = TASK_ROOT / "configs"
DATA_INPUT_DIR = TASK_ROOT / "data" / "input"
DATA_RESULTS_DIR = TASK_ROOT / "data" / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time optimized inference with configurable latency/throughput tradeoff."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Name of the .keras file under savedModels/ (e.g. test2.keras).",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Filename of the input waveform located in data/input/ (e.g. rawAudio.wav).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Filename for the output waveform (saved in data/results/<model-name>/).",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=64,
        help="Number of samples to process per block (default: 64). Smaller = lower latency but slower.",
    )
    return parser.parse_args()


def load_config_for_model(model_path: Path) -> Tuple[int, int]:
    """Load seq_len and context from the config file matching the model name."""
    config_path = CONFIGS_DIR / f"{model_path.stem}.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file `{config_path.name}` was not found in `{CONFIGS_DIR}`.\n"
            "Ensure the config shares the same base name as the saved model."
        )
    with config_path.open("r", encoding="utf-8-sig") as handle:
        config = json.load(handle)
    
    data_cfg = config.get("data", {})
    try:
        seq_len = int(data_cfg["seq_len"])
    except KeyError as exc:
        raise KeyError(f"Missing `{exc.args[0]}` in the `data` section of {config_path}.") from exc
    if seq_len <= 0:
        raise ValueError("`seq_len` must be a positive integer in the config.")
    
    model_cfg = config.get("model", {})
    kernel_size = int(model_cfg.get("kernel_size", 3))
    nb_stacks = int(model_cfg.get("nb_stacks", 1))
    dilations_cfg = model_cfg.get("dilations")
    if dilations_cfg is None:
        raise ValueError("Model config must include `dilations` to derive context.")
    dilations = coerce_dilations(dilations_cfg)
    context = compute_context_length(kernel_size, dilations, nb_stacks)

    return seq_len, context


def load_waveform(path: Path) -> Tuple[np.ndarray, int]:
    """Load a mono WAV file and return the waveform and sample rate."""
    audio_bytes = tf.io.read_file(str(path))
    waveform, sample_rate = tf.audio.decode_wav(audio_bytes)
    sr = int(sample_rate.numpy())
    if waveform.shape[-1] != 1:
        raise ValueError(f"WAV file `{path}` must be mono. Found {waveform.shape[-1]} channels.")
    waveform = tf.squeeze(tf.cast(waveform, tf.float32), axis=-1).numpy()
    return waveform, sr


def streaming_inference_optimized(
    model: tf.keras.Model,
    normalized_waveform: np.ndarray,
    seq_len: int,
    context: int,
    block_size: int = 64,
) -> Tuple[np.ndarray, float, dict]:
    """
    Process audio in streaming fashion with mini-batches for efficiency.
    
    Args:
        model: Trained TCN model
        normalized_waveform: Input audio normalized to [0, 1]
        seq_len: Target sequence length (per-block output window)
        context: Additional causal samples required by the receptive field
        block_size: Number of samples to process per model call
    
    Returns:
        outputs: Processed audio samples
        elapsed_time: Total computation time in seconds
        stats: Dictionary with performance metrics
    """
    num_samples = len(normalized_waveform)
    input_seq_len = seq_len + context
    buffer = np.zeros(input_seq_len, dtype=np.float32)
    outputs = np.zeros(num_samples, dtype=np.float32)
    
    LOGGER.info("Starting streaming inference on %d samples (block_size=%d)...", 
                num_samples, block_size)
    
    # Statistics
    num_blocks = 0
    total_model_calls = 0
    log_interval = max(1, (num_samples // block_size) // 10)
    
    inference_start = time.perf_counter()
    
    idx = 0
    while idx < num_samples:
        # Determine block size (might be smaller at the end)
        current_block_size = min(block_size, num_samples - idx)
        
        # Prepare mini-batch of windows
        windows = []
        for offset in range(current_block_size):
            # Update buffer with new sample
            buffer[:-1] = buffer[1:]
            buffer[-1] = normalized_waveform[idx + offset]
            windows.append(buffer.copy())

        # Stack windows into batch: (block_size, input_seq_len)
        batch = np.stack(windows, axis=0)
        batch = batch[..., np.newaxis]  # (block_size, input_seq_len, 1)

        # Single model call for the entire block
        preds = model(batch, training=False)
        
        # Extract predictions
        if isinstance(preds, tf.Tensor):
            preds = preds.numpy()
        preds = np.asarray(preds)
        
        # Take the last output from each prediction
        if preds.ndim == 3:
            block_outputs = preds[:, -1, 0]
        elif preds.ndim == 2:
            block_outputs = preds[:, -1]
        else:
            raise ValueError(f"Unexpected prediction shape: {preds.shape}")
        
        # Store outputs
        outputs[idx:idx + current_block_size] = block_outputs
        
        idx += current_block_size
        num_blocks += 1
        total_model_calls += 1
        
        # Progress logging
        if num_blocks % log_interval == 0 or idx >= num_samples:
            elapsed = time.perf_counter() - inference_start
            progress = idx / num_samples * 100
            samples_per_sec = idx / elapsed if elapsed > 0 else 0
            LOGGER.info(
                "Progress: %d/%d (%.1f%%) | %.2f samples/sec | %d model calls",
                idx,
                num_samples,
                progress,
                samples_per_sec,
                total_model_calls,
            )
    
    total_elapsed = time.perf_counter() - inference_start
    
    stats = {
        "num_samples": num_samples,
        "block_size": block_size,
        "num_blocks": num_blocks,
        "total_model_calls": total_model_calls,
        "avg_samples_per_call": num_samples / total_model_calls if total_model_calls > 0 else 0,
    }
    
    return outputs, total_elapsed, stats


def ensure_results_dir(model_name: str) -> Path:
    """Ensure the results directory exists for the given model."""
    dest = DATA_RESULTS_DIR / model_name
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def run_inference(args: argparse.Namespace):
    """Main real-time inference pipeline."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    # Resolve paths
    model_path = (SAVED_MODELS_DIR / args.model).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Saved model `{model_path}` not found.")

    input_path = (DATA_INPUT_DIR / args.input).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input WAV `{input_path}` not found.")

    results_dir = ensure_results_dir(model_path.stem)
    output_path = (results_dir / args.output).resolve()

    # Load config and model
    seq_len, context = load_config_for_model(model_path)
    input_seq_len = seq_len + context
    LOGGER.info(
        "Using seq_len=%d, context=%d (input_seq_len=%d) from config.",
        seq_len,
        context,
        input_seq_len,
    )

    LOGGER.info("Loading model from `%s`...", model_path)
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={"TCN": TCN, "ModelMetadata": ModelMetadata},
        compile=False,
    )

    LOGGER.info("Model architecture:")
    model.summary()

    # Load and preprocess input audio
    LOGGER.info("Loading input waveform `%s`...", input_path)
    waveform, sample_rate = load_waveform(input_path)
    metadata_layer = None
    try:
        metadata_layer = model.get_layer("model_metadata")
    except ValueError:
        LOGGER.warning(
            "Model `%s` is missing training metadata; skipping compatibility checks.",
            model_path.name,
        )

    metadata_config = {}
    model_sample_rate = None
    if metadata_layer is not None:
        model_sample_rate = int(getattr(metadata_layer, "sample_rate", None))
        try:
            metadata_config = metadata_layer.config_dict
        except Exception:  # pragma: no cover
            metadata_config = {}

    if model_sample_rate is None:
        LOGGER.warning("Sample rate metadata unavailable; skipping sample rate check.")
    elif model_sample_rate != sample_rate:
        raise ValueError(
            f"Sample rate mismatch: model expects {model_sample_rate} Hz but input `{input_path.name}` is {sample_rate} Hz."
        )
    else:
        LOGGER.info(
            "Sample rate check passed (model=%d Hz, input=%d Hz).", model_sample_rate, sample_rate
        )
    if metadata_config:
        config_path = metadata_config.get("config_path")
        if config_path:
            LOGGER.info("Model metadata references config `%s`.", config_path)
    original_length = waveform.shape[0]
    audio_duration = original_length / sample_rate
    
    LOGGER.info(
        "Audio info: %d samples, %.3f seconds @ %d Hz",
        original_length,
        audio_duration,
        sample_rate,
    )
    
    normalized_waveform = normalize_to_unit_range(waveform)

    # Run streaming inference with mini-batches
    predictions, inference_time, stats = streaming_inference_optimized(
        model,
        normalized_waveform,
        seq_len,
        context,
        block_size=args.block_size,
    )

    # Denormalize output
    reconstructed = np.clip(predictions, 0.0, 1.0)
    reconstructed = denormalize_from_unit_range(reconstructed)

    # Write output
    LOGGER.info("Writing result to `%s`...", output_path)
    audio_tensor = tf.convert_to_tensor(reconstructed[:, np.newaxis], dtype=tf.float32)
    wav_bytes = tf.audio.encode_wav(audio_tensor, sample_rate)
    tf.io.write_file(str(output_path), wav_bytes)

    # Real-time performance analysis
    realtime_factor = audio_duration / inference_time if inference_time > 0 else float("inf")
    samples_per_sec = original_length / inference_time if inference_time > 0 else 0
    latency_ms = (args.block_size / sample_rate) * 1000
    
    LOGGER.info("")
    LOGGER.info("=" * 70)
    LOGGER.info("REAL-TIME STREAMING PERFORMANCE REPORT")
    LOGGER.info("=" * 70)
    LOGGER.info("Audio duration:         %.3f seconds", audio_duration)
    LOGGER.info("Computation time:       %.3f seconds", inference_time)
    LOGGER.info("Real-time factor:       %.2f x", realtime_factor)
    LOGGER.info("Throughput:             %.2f samples/sec", samples_per_sec)
    LOGGER.info("Required for real-time: %d samples/sec", sample_rate)
    LOGGER.info("")
    LOGGER.info("Streaming configuration:")
    LOGGER.info("  Block size:           %d samples", args.block_size)
    LOGGER.info("  Block latency:        %.2f ms", latency_ms)
    LOGGER.info("  Total blocks:         %d", stats["num_blocks"])
    LOGGER.info("  Model calls:          %d (vs %d for sample-by-sample)", 
                stats["total_model_calls"], original_length)
    LOGGER.info("  Speedup factor:       %.1f x fewer model calls",
                original_length / stats["total_model_calls"])
    LOGGER.info("")
    
    if realtime_factor >= 1.0:
        speedup_pct = (realtime_factor - 1.0) * 100
        LOGGER.info("SUCCESS: Model CAN process audio in real-time!")
        LOGGER.info("  Processing is %.1f%% faster than real-time.", speedup_pct)
        LOGGER.info("  Effective latency: %.2f ms per block", latency_ms)
        LOGGER.info("  Can be used for: Live processing, plugins, hardware")
    else:
        slowdown_pct = (1.0 - realtime_factor) * 100
        LOGGER.info("FAILURE: Model CANNOT process audio in real-time.")
        LOGGER.info("  Processing is %.1f%% slower than real-time.", slowdown_pct)
        LOGGER.info("  Need to speed up by %.2f x to achieve real-time.", 1.0 / realtime_factor)
        LOGGER.info("")
        LOGGER.info("Suggestions to improve performance:")
        LOGGER.info("  1. Increase --block-size (e.g. 128, 256) for better throughput")
        LOGGER.info("  2. Reduce model complexity (fewer filters, stacks, or dilations)")
        LOGGER.info("  3. Use GPU acceleration")
        LOGGER.info("  4. Convert model to TensorFlow Lite or ONNX")
    
    LOGGER.info("=" * 70)
    LOGGER.info("")
    LOGGER.info("Output saved to: %s", output_path)


if __name__ == "__main__":
    run_inference(parse_args())
