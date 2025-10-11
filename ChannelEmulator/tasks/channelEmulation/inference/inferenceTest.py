import argparse
import json
import logging
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf

from tcn import TCN  # noqa: F401 - required to deserialize custom layer

LOGGER = logging.getLogger(__name__)

TASK_ROOT = Path(__file__).resolve().parent.parent
SAVED_MODELS_DIR = TASK_ROOT / "savedModels"
CONFIGS_DIR = TASK_ROOT / "configs"
DATA_INPUT_DIR = TASK_ROOT / "data" / "input"
DATA_RESULTS_DIR = TASK_ROOT / "data" / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference with a trained TCN model and export the processed audio."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Name of the .keras file under savedModels/ (e.g. wavenet3.keras).",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Filename of the input waveform located in data/input/ (e.g. rawAudio.wav).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help=(
            "Unused placeholder to keep compatibility. "
            "Outputs are stored as <input>_inf.wav under data/results/<model-name>/."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size to use during inference (defaults to 8).",
    )
    return parser.parse_args()


def load_config_for_model(model_path: Path) -> Tuple[int, int]:
    config_path = CONFIGS_DIR / f"{model_path.stem}.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file `{config_path.name}` was not found in `{CONFIGS_DIR}`.\n"
            "Ensure the config shares the same base name as the saved model."
        )
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    data_cfg = config.get("data", {})
    try:
        seq_len = int(data_cfg["seq_len"])
        hop_len = int(data_cfg["hop_len"])
    except KeyError as exc:
        raise KeyError(
            f"Missing `{exc.args[0]}` in the `data` section of {config_path}."
        ) from exc
    if seq_len <= 0 or hop_len <= 0:
        raise ValueError("`seq_len` and `hop_len` must be positive integers in the config.")
    return seq_len, hop_len


def load_waveform(path: Path) -> Tuple[np.ndarray, int]:
    audio_bytes = tf.io.read_file(str(path))
    waveform, sample_rate = tf.audio.decode_wav(audio_bytes)
    sr = int(sample_rate.numpy())
    if waveform.shape[-1] != 1:
        raise ValueError(f"WAV file `{path}` must be mono. Found {waveform.shape[-1]} channels.")
    waveform = tf.squeeze(tf.cast(waveform, tf.float32), axis=-1).numpy()
    return waveform, sr


def normalize_waveform(waveform: np.ndarray) -> Tuple[np.ndarray, float, float]:
    min_val = float(np.min(waveform))
    max_val = float(np.max(waveform))
    if np.isclose(max_val, min_val):
        return np.zeros_like(waveform, dtype=np.float32), min_val, max_val
    normalized = (waveform - min_val) / (max_val - min_val)
    return normalized.astype(np.float32), min_val, max_val


def denormalize_waveform(waveform: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    if np.isclose(max_val, min_val):
        return np.full_like(waveform, fill_value=min_val, dtype=np.float32)
    denormalized = waveform * (max_val - min_val) + min_val
    return denormalized.astype(np.float32)


def frame_audio(waveform: np.ndarray, frame_length: int, frame_step: int) -> np.ndarray:
    tensor = tf.convert_to_tensor(waveform, dtype=tf.float32)
    frames = tf.signal.frame(tensor, frame_length, frame_step, pad_end=True, pad_value=0.0)
    return frames.numpy()


def overlap_add(frames: np.ndarray, frame_step: int, original_length: int) -> np.ndarray:
    frame_length = frames.shape[1]
    total_length = frame_step * (frames.shape[0] - 1) + frame_length
    aggregate = np.zeros(total_length, dtype=np.float32)
    weight = np.zeros(total_length, dtype=np.float32)

    for idx, frame in enumerate(frames):
        start = idx * frame_step
        end = start + frame_length
        aggregate[start:end] += frame
        weight[start:end] += 1.0

    weight[weight == 0] = 1.0
    reconstructed = aggregate / weight
    return reconstructed[:original_length]


def ensure_results_dir():
    DATA_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def run_inference(args: argparse.Namespace):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    model_path = (SAVED_MODELS_DIR / args.model).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Saved model `{model_path}` not found.")

    input_path = (DATA_INPUT_DIR / args.input).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input WAV `{input_path}` not found.")

    ensure_results_dir()
    model_results_dir = DATA_RESULTS_DIR / model_path.stem
    model_results_dir.mkdir(parents=True, exist_ok=True)
    input_stem = Path(args.input).stem
    output_path = (model_results_dir / f"{input_stem}_inf.wav").resolve()

    seq_len, hop_len = load_config_for_model(model_path)
    LOGGER.info("Using seq_len=%d, hop_len=%d inferred from config.", seq_len, hop_len)

    LOGGER.info("Loading model from `%s`...", model_path)
    model = tf.keras.models.load_model(model_path, custom_objects={"TCN": TCN}, compile=False)

    LOGGER.info("Loading input waveform `%s`...", input_path)
    waveform, sample_rate = load_waveform(input_path)
    original_length = waveform.shape[0]
    normalized_waveform, min_val, max_val = normalize_waveform(waveform)

    LOGGER.info("Framing audio into windows...")
    frames = frame_audio(normalized_waveform, seq_len, hop_len)
    if frames.size == 0:
        raise ValueError("No frames produced from the input audio. Check seq_len and hop_len.")

    frames = frames[..., np.newaxis]  # (num_frames, seq_len, 1)

    LOGGER.info("Running inference on %d frame(s)...", frames.shape[0])
    inference_start = time.perf_counter()
    predictions = model.predict(frames, batch_size=args.batch_size, verbose=0)
    predictions = np.squeeze(np.asarray(predictions), axis=-1)

    LOGGER.info("Reconstructing waveform from overlapping windows...")
    reconstructed = overlap_add(predictions, hop_len, original_length)
    reconstructed = np.clip(reconstructed, 0.0, 1.0)
    reconstructed = denormalize_waveform(reconstructed, min_val, max_val)

    LOGGER.info("Writing result to `%s`...", output_path)
    audio_tensor = tf.convert_to_tensor(reconstructed[:, np.newaxis], dtype=tf.float32)
    wav_bytes = tf.audio.encode_wav(audio_tensor, sample_rate)
    tf.io.write_file(str(output_path), wav_bytes)
    elapsed = time.perf_counter() - inference_start

    audio_duration = original_length / sample_rate
    realtime_factor = audio_duration / elapsed if elapsed > 0 else float("inf")

    LOGGER.info(
        "Inference complete in %.3f s (audio duration %.3f s, real-time factor %.2f). Saved to `%s`.",
        elapsed,
        audio_duration,
        realtime_factor,
        output_path,
    )


if __name__ == "__main__":
    run_inference(parse_args())
