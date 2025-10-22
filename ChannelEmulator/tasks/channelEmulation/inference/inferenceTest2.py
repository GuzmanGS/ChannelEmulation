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
from tcn import TCN  # noqa: F401 - required for deserialization

LOGGER = logging.getLogger(__name__)

TASK_ROOT = Path(__file__).resolve().parent.parent
SAVED_MODELS_DIR = TASK_ROOT / "savedModels"
CONFIGS_DIR = TASK_ROOT / "configs"
DATA_INPUT_DIR = TASK_ROOT / "data" / "input"
DATA_RESULTS_DIR = TASK_ROOT / "data" / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run sample-by-sample inference with a trained TCN model and export the processed audio."
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
            "Unused placeholder to keep compatibility. Outputs are stored as <input>_inf.wav "
            "under data/results/<model-name>/"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Preserved for compatibility; ignored because inference runs strictly sample by sample.",
    )
    return parser.parse_args()


def load_config(model_path: Path) -> Tuple[int, int, int]:
    config_path = CONFIGS_DIR / f"{model_path.stem}.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file {config_path.name} was not found in {CONFIGS_DIR}.\n"
            "Ensure the config shares the same base name as the saved model."
        )
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    data_cfg = config.get("data", {})

    try:
        seq_len = int(data_cfg["seq_len"])
    except KeyError as exc:
        raise KeyError(f"Missing {exc.args[0]} in the data section of {config_path}.") from exc
    if seq_len <= 0:
        raise ValueError("seq_len must be a positive integer in the config.")

    hop_len_cfg = data_cfg.get("hop_len")
    overlap = data_cfg.get("overlap")
    if overlap is not None:
        overlap = float(overlap)
        if not 0 <= overlap < 1:
            raise ValueError("overlap must be in the range [0, 1).")
        hop_len = max(1, int(round(seq_len * (1.0 - overlap))))
    elif hop_len_cfg is not None:
        hop_len = int(hop_len_cfg)
        if hop_len <= 0:
            raise ValueError("hop_len must be a positive integer in the config.")
    else:
        raise ValueError("Config must define either hop_len or overlap.")

    model_cfg = config.get("model", {})
    kernel_size = int(model_cfg.get("kernel_size", 3))
    nb_stacks = int(model_cfg.get("nb_stacks", 1))
    dilations_cfg = model_cfg.get("dilations")
    if dilations_cfg is None:
        raise ValueError("Model config must include dilations to derive context.")
    dilations = coerce_dilations(dilations_cfg)
    context = compute_context_length(kernel_size, dilations, nb_stacks)

    return seq_len, hop_len, context


def load_waveform(path: Path) -> Tuple[np.ndarray, int]:
    audio_bytes = tf.io.read_file(str(path))
    waveform, sample_rate = tf.audio.decode_wav(audio_bytes)
    sr = int(sample_rate.numpy())
    if waveform.shape[-1] != 1:
        raise ValueError(f"WAV file {path} must be mono. Found {waveform.shape[-1]} channels.")
    waveform = tf.squeeze(tf.cast(waveform, tf.float32), axis=-1).numpy()
    return waveform, sr


def ensure_results_dir(model_name: str) -> Path:
    dest = DATA_RESULTS_DIR / model_name
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def sample_by_sample_inference(
    model: tf.keras.Model,
    normalized_waveform: np.ndarray,
    seq_len: int,
    context: int,
) -> np.ndarray:
    num_samples = len(normalized_waveform)
    input_seq_len = seq_len + context
    buffer = np.zeros(input_seq_len, dtype=np.float32)
    outputs = np.zeros(num_samples, dtype=np.float32)

    for idx, sample in enumerate(normalized_waveform):
        buffer[:-1] = buffer[1:]
        buffer[-1] = sample
        window = buffer.reshape(1, input_seq_len, 1)
        pred = model(window, training=False)
        if isinstance(pred, tf.Tensor):
            pred = pred.numpy()
        pred = np.asarray(pred)
        if pred.ndim == 3:
            current = pred[0, -1, 0]
        elif pred.ndim == 2:
            current = pred[0, -1]
        elif pred.ndim == 1:
            current = pred[-1]
        else:
            raise ValueError(f"Unexpected prediction shape: {pred.shape}")
        outputs[idx] = float(current)

        if idx % 100000 == 0 and idx > 0:
            LOGGER.info("Processed %d / %d samples...", idx, num_samples)

    return outputs


def run_inference(args: argparse.Namespace):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    model_path = (SAVED_MODELS_DIR / args.model).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Saved model {model_path} not found.")

    input_path = (DATA_INPUT_DIR / args.input).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input WAV {input_path} not found.")

    seq_len, hop_len, context = load_config(model_path)
    input_seq_len = seq_len + context
    LOGGER.info(
        "Model %s: seq_len=%d, hop_len=%d (implied overlap %.2f), context=%d (input_seq_len=%d).",
        model_path.name,
        seq_len,
        hop_len,
        1.0 - hop_len / seq_len,
        context,
        input_seq_len,
    )

    LOGGER.info("Loading model from %s...", model_path)
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={"TCN": TCN, "ModelMetadata": ModelMetadata},
        compile=False,
    )

    LOGGER.info("Model architecture:")
    model.summary()

    LOGGER.info("Reading input waveform %s...", input_path)
    waveform, sample_rate = load_waveform(input_path)
    metadata_layer = None
    try:
        metadata_layer = model.get_layer("model_metadata")
    except ValueError:
        LOGGER.warning("Model `%s` is missing metadata; skipping compatibility check.", model_path.name)

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
        LOGGER.info("Sample rate check passed (model=%d Hz, input=%d Hz).", model_sample_rate, sample_rate)
    if metadata_config:
        config_path = metadata_config.get("config_path")
        if config_path:
            LOGGER.info("Model metadata references config `%s`.", config_path)
    normalized_waveform = normalize_to_unit_range(waveform)

    LOGGER.info("Running strictly sequential inference over %d samples...", len(normalized_waveform))
    inference_start = time.perf_counter()
    predictions = sample_by_sample_inference(
        model,
        normalized_waveform,
        seq_len,
        context,
    )
    elapsed = time.perf_counter() - inference_start

    reconstructed = np.clip(predictions, 0.0, 1.0)
    reconstructed = denormalize_from_unit_range(reconstructed)

    results_dir = ensure_results_dir(model_path.stem)
    output_path = results_dir / f"{Path(args.input).stem}_inf.wav"

    LOGGER.info("Writing result to %s...", output_path)
    audio_tensor = tf.convert_to_tensor(reconstructed[:, np.newaxis], dtype=tf.float32)
    wav_bytes = tf.audio.encode_wav(audio_tensor, sample_rate)
    tf.io.write_file(str(output_path), wav_bytes)

    audio_duration = len(waveform) / sample_rate
    realtime_factor = audio_duration / elapsed if elapsed > 0 else float("inf")
    LOGGER.info(
        "Inference complete in %.3f s (audio duration %.3f s, real-time factor %.2f).",
        elapsed,
        audio_duration,
        realtime_factor,
    )


if __name__ == "__main__":
    run_inference(parse_args())
