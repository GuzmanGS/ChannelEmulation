from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf

TASK_ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_DIR = TASK_ROOT / "data" / "input"


def validate_audio(file_path: Path) -> Tuple[int, int]:
    if not file_path.exists():
        raise FileNotFoundError(f"Input file `{file_path}` does not exist.")

    info = sf.info(str(file_path))

    if info.channels != 1:
        raise ValueError(f"File `{file_path}` must be mono; found {info.channels} channels.")

    if info.subtype not in {"PCM_16", "PCM_24", "PCM_32", "FLOAT", "DOUBLE"}:
        raise ValueError(
            f"Unsupported subtype `{info.subtype}` in `{file_path}`. Expected standard PCM/float WAV."
        )

    return info.samplerate, info.subtype


def concatenate_wavs(first: Path, second: Path, output: Path) -> None:
    sr_first, subtype_first = validate_audio(first)
    sr_second, subtype_second = validate_audio(second)

    if sr_first != sr_second:
        raise ValueError(
            f"Sample rate mismatch: `{first.name}`={sr_first} Hz vs `{second.name}`={sr_second} Hz."
        )
    if subtype_first != subtype_second:
        raise ValueError(
            f"Bit depth/subtype mismatch: `{first.name}`={subtype_first} vs `{second.name}`={subtype_second}."
        )

    data_first, _ = sf.read(str(first), dtype="float32")
    data_second, _ = sf.read(str(second), dtype="float32")

    concatenated = np.concatenate([data_first, data_second], axis=0)

    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output), concatenated, sr_first, subtype=subtype_first)

    print(f"Saved concatenated file to `{output}` ({len(concatenated)} samples @ {sr_first} Hz).")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Concatenate two mono WAV files from data/input/ ensuring matching format."
    )
    parser.add_argument("first", help="First WAV file name (located in data/input/).")
    parser.add_argument("second", help="Second WAV file name (located in data/input/).")
    parser.add_argument("output", help="Output WAV file name to create in data/input/.")
    return parser.parse_args()


def resolve_input(name: str) -> Path:
    path = Path(name)
    return path if path.is_absolute() else (INPUT_DIR / path).resolve()


def main() -> None:
    args = parse_args()
    first = resolve_input(args.first)
    second = resolve_input(args.second)
    output = resolve_input(args.output)

    concatenate_wavs(first, second, output)


if __name__ == "__main__":
    main()
