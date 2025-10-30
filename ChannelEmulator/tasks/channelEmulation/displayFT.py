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
    """Return mono waveform and sample rate for the provided WAV file."""
    audio_bytes = tf.io.read_file(str(path))
    waveform, sample_rate = tf.audio.decode_wav(audio_bytes)
    if waveform.shape[-1] != 1:
        raise ValueError(f"WAV file `{path}` must be mono. Found {waveform.shape[-1]} channels.")
    return tf.squeeze(waveform, axis=-1).numpy(), int(sample_rate.numpy())


def compute_spectrum(waveform: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute single-sided magnitude spectrum."""
    # Apply Hann window to reduce spectral leakage
    window = np.hanning(len(waveform))
    windowed = waveform * window
    spectrum = np.fft.rfft(windowed)
    freqs = np.fft.rfftfreq(len(windowed), d=1.0 / sample_rate)
    magnitude = np.abs(spectrum) / np.sum(window)
    magnitude_db = 20 * np.log10(np.maximum(magnitude, 1e-12))
    return freqs, magnitude_db


def display_spectra(
    spectra: list[tuple[str, np.ndarray, np.ndarray, str]],
    *,
    log_frequency: bool = False,
) -> None:
    if not spectra:
        raise ValueError("At least one waveform must be provided for display.")

    plt.figure(figsize=(12, 6))
    for label, freqs, mags_db, color in spectra:
        plt.plot(freqs, mags_db, label=label, color=color, linewidth=1.0, alpha=0.5)

    plt.title("Magnitude Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    if log_frequency:
        plt.xscale("log")
    plt.tight_layout()
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Display Fourier magnitude spectra for up to three WAV files."
    )
    parser.add_argument("--input", help="WAV filename under data/input/ to plot in green.")
    parser.add_argument("--output", help="WAV filename under data/output/ to plot in blue.")
    parser.add_argument("--result", help="WAV filename under data/results/ to plot in orange.")
    parser.add_argument(
        "--logfreq",
        action="store_true",
        help="Use logarithmic frequency axis for the spectrum.",
    )
    return parser.parse_args()


def resolve_under(base: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve()


def main() -> None:
    args = parse_args()
    spectra: list[tuple[str, np.ndarray, np.ndarray, str]] = []

    if args.input:
        path = resolve_under(DATA_INPUT_DIR, args.input)
        if not path.exists():
            raise FileNotFoundError(f"File `{path}` does not exist.")
        waveform, sr = load_waveform(path)
        freqs, mags = compute_spectrum(waveform, sr)
        spectra.append((f"Input: {path.name}", freqs, mags, "tab:green"))

    if args.output:
        path = resolve_under(DATA_OUTPUT_DIR, args.output)
        if not path.exists():
            raise FileNotFoundError(f"File `{path}` does not exist.")
        waveform, sr = load_waveform(path)
        freqs, mags = compute_spectrum(waveform, sr)
        spectra.append((f"Output: {path.name}", freqs, mags, "tab:blue"))

    if args.result:
        path = resolve_under(DATA_RESULTS_DIR, args.result)
        if not path.exists():
            raise FileNotFoundError(f"File `{path}` does not exist.")
        waveform, sr = load_waveform(path)
        freqs, mags = compute_spectrum(waveform, sr)
        spectra.append((f"Result: {path.name}", freqs, mags, "tab:orange"))

    if not spectra:
        raise SystemExit("No waveforms specified. Provide at least one of --input, --output, or --result.")

    display_spectra(spectra, log_frequency=args.logfreq)


if __name__ == "__main__":
    main()
