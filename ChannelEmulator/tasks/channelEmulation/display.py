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


def display_waveforms(waveforms: list[tuple[str, Path, str]]) -> None:
    """Plot up to three waveforms on the same axes with predefined colours."""
    if not waveforms:
        raise ValueError("At least one waveform must be provided for display.")

    plt.figure(figsize=(12, 6))

    reference_sr: int | None = None
    for idx, (label, path, color) in enumerate(waveforms):
        waveform, sr = load_waveform(path)
        if reference_sr is None:
            reference_sr = sr
        elif sr != reference_sr:
            raise ValueError(
                f"Sample rate mismatch when plotting `{path.name}`. "
                f"Expected {reference_sr} Hz but got {sr} Hz."
            )
        time_axis = np.arange(len(waveform), dtype=np.float32) / sr
        plt.plot(
            time_axis,
            waveform,
            label=label,
            linewidth=1.0,
            color=color,
            alpha=0.5,
        )

    plt.title("Waveform Comparison")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Display up to three WAV files overlaid in a single plot."
    )
    parser.add_argument(
        "--input",
        help="WAV filename located under data/input/ to plot in green.",
    )
    parser.add_argument(
        "--output",
        help="WAV filename located under data/output/ to plot in blue.",
    )
    parser.add_argument(
        "--result",
        help="WAV filename located under data/results/ to plot in orange.",
    )
    return parser.parse_args()


def resolve_under(base: Path, value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate.resolve()
    return (base / candidate).resolve()


def main() -> None:
    args = parse_args()
    selections: list[tuple[str, Path, str]] = []

    if args.input:
        input_path = resolve_under(DATA_INPUT_DIR, args.input)
        if not input_path.exists():
            raise FileNotFoundError(f"File `{input_path}` does not exist.")
        selections.append((f"Input: {input_path.name}", input_path, "tab:green"))

    if args.output:
        output_path = resolve_under(DATA_OUTPUT_DIR, args.output)
        if not output_path.exists():
            raise FileNotFoundError(f"File `{output_path}` does not exist.")
        selections.append((f"Output: {output_path.name}", output_path, "tab:blue"))

    if args.result:
        result_path = resolve_under(DATA_RESULTS_DIR, args.result)
        if not result_path.exists():
            raise FileNotFoundError(f"File `{result_path}` does not exist.")
        selections.append((f"Result: {result_path.name}", result_path, "tab:orange"))

    if not selections:
        raise SystemExit("No waveforms specified. Provide at least one of --input, --output, or --result.")

    display_waveforms(selections)


if __name__ == "__main__":
    main()
