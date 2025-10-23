from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

TASK_ROOT = Path(__file__).resolve().parent
DATA_INPUT_DIR = TASK_ROOT / "data" / "input"
DATA_OUTPUT_DIR = TASK_ROOT / "data" / "output"
DATA_RESULTS_DIR = TASK_ROOT / "data" / "results"


def load_waveform(path: Path) -> tuple[np.ndarray, int]:
    """Return waveform (mono) and sample rate for the provided WAV file."""
    audio_bytes = tf.io.read_file(str(path))
    waveform, sample_rate = tf.audio.decode_wav(audio_bytes)

    if waveform.shape[-1] != 1:
        raise ValueError(f"WAV file `{path}` must be mono. Found {waveform.shape[-1]} channels.")

    return tf.squeeze(waveform, axis=-1).numpy(), int(sample_rate.numpy())


def display_waveforms(
    target: Path,
    result: Path,
    input_waveform: Path | None = None,
) -> None:
    """Plot waveforms on the same axes (Matlab-style overlay)."""
    waveform_t, sr_t = load_waveform(target)
    waveform_r, sr_r = load_waveform(result)

    if sr_t != sr_r:
        raise ValueError(
            f"Sample rate mismatch: `{target}` is {sr_t} Hz while `{result}` is {sr_r} Hz."
        )

    plt.figure(figsize=(12, 6))

    if input_waveform is not None:
        waveform_i, sr_i = load_waveform(input_waveform)
        if sr_i != sr_t:
            raise ValueError(
                f"Sample rate mismatch: `{input_waveform}` is {sr_i} Hz while `{target}` is {sr_t} Hz."
            )
        time_i = np.arange(len(waveform_i), dtype=np.float32) / sr_i
        plt.plot(
            time_i,
            waveform_i,
            label=f"Input: {input_waveform.name}",
            linewidth=1.0,
            color="tab:green",
        )

    time_t = np.arange(len(waveform_t), dtype=np.float32) / sr_t
    time_r = np.arange(len(waveform_r), dtype=np.float32) / sr_r

    plt.plot(time_t, waveform_t, label=f"Target: {target.name}", linewidth=1.0)
    plt.plot(time_r, waveform_r, label=f"Result: {result.name}", linewidth=1.0, alpha=0.8)
    plt.title("Waveform Comparison")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Display two WAV files overlaid in a single plot."
    )
    parser.add_argument(
        "--target",
        required=True,
        help="Name or relative path of the WAV file located under data/output/.",
    )
    parser.add_argument(
        "--result",
        required=True,
        help="Name or relative path of the WAV file located under data/results/.",
    )
    parser.add_argument(
        "--input",
        help="Optional WAV filename located under data/input/ to plot first in green.",
    )
    return parser.parse_args()


def resolve_under(base: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate.resolve()
    return (base / candidate).resolve()


def main() -> None:
    args = parse_args()
    target = resolve_under(DATA_OUTPUT_DIR, args.target)
    result = resolve_under(DATA_RESULTS_DIR, args.result)
    input_path = resolve_under(DATA_INPUT_DIR, args.input) if args.input else None

    if not target.exists():
        raise FileNotFoundError(f"File `{target}` does not exist.")
    if not result.exists():
        raise FileNotFoundError(f"File `{result}` does not exist.")
    if input_path is not None and not input_path.exists():
        raise FileNotFoundError(f"File `{input_path}` does not exist.")

    display_waveforms(target, result, input_path)


if __name__ == "__main__":
    main()
