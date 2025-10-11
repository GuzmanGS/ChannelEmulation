from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import tensorflow as tf

DEFAULT_SAMPLE_RATE = 16_000

LOGGER = logging.getLogger(__name__)


@dataclass
class SequenceDataset:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    sample_rate: int
    seq_len: int


def prepare_sequence_dataset(
    data_root: Path,
    seq_len: int,
    hop_len: int,
    sample_rate: Optional[int] = None,
    val_split: float = 0.1,
    limit_total_segments: Optional[int] = None,
    seed: int = 42,
    input_filename: str = "rawAudio.wav",
    output_filename: str = "fxAudioVFuzz.wav",
) -> SequenceDataset:
    """Load WAV pairs and convert them into framed sequences suited for seq2seq training."""
    if seq_len <= 0:
        raise ValueError("`seq_len` must be a positive integer.")
    if hop_len <= 0:
        raise ValueError("`hop_len` must be a positive integer.")
    if not 0 < val_split < 1:
        raise ValueError("`val_split` must be between 0 and 1.")

    data_root = Path(data_root)
    input_dir, target_dir = _ensure_structure(data_root)
    rng = np.random.default_rng(seed)

    wav_pairs = _collect_wav_pairs(
        input_dir,
        target_dir,
        input_filename=input_filename,
        output_filename=output_filename,
    )
    resolved_sample_rate: Optional[int] = sample_rate

    if not wav_pairs:
        LOGGER.warning(
            "No WAV pairs found in `%s`. Falling back to a synthetic dataset.", data_root
        )
        return _synthetic_dataset(seq_len, val_split, rng, sample_rate or DEFAULT_SAMPLE_RATE)

    segments_x: List[np.ndarray] = []
    segments_y: List[np.ndarray] = []
    total_segments = 0

    for input_path, target_path in wav_pairs:
        input_waveform, sr_in = _load_waveform(input_path, resolved_sample_rate)
        target_waveform, sr_out = _load_waveform(target_path, resolved_sample_rate or sr_in)

        input_waveform = _normalize_waveform(input_waveform)
        target_waveform = _normalize_waveform(target_waveform)

        if sr_in != sr_out:
            raise ValueError(
                f"Sample rate mismatch between `{input_path.name}` ({sr_in} Hz) and "
                f"`{target_path.name}` ({sr_out} Hz)."
            )

        if resolved_sample_rate is None:
            resolved_sample_rate = sr_in

        if len(input_waveform) != len(target_waveform):
            raise ValueError(
                f"Input `{input_path.name}` and target `{target_path.name}` must have identical length."
            )

        trimmed_len = len(input_waveform)
        if trimmed_len < seq_len:
            LOGGER.warning(
                "Skipping pair `%s` because trimmed length (%d) is shorter than seq_len (%d).",
                input_path.name,
                trimmed_len,
                seq_len,
            )
            continue

        input_frames = _frame_waveform(input_waveform, seq_len, hop_len)
        target_frames = _frame_waveform(target_waveform, seq_len, hop_len)

        frame_count = min(len(input_frames), len(target_frames))
        if frame_count == 0:
            continue

        if limit_total_segments is not None and total_segments >= limit_total_segments:
            break

        if (
            limit_total_segments is not None
            and total_segments + frame_count > limit_total_segments
        ):
            frame_count = limit_total_segments - total_segments
            input_frames = input_frames[:frame_count]
            target_frames = target_frames[:frame_count]

        segments_x.append(input_frames[:frame_count])
        segments_y.append(target_frames[:frame_count])
        total_segments += frame_count

        if limit_total_segments is not None and total_segments >= limit_total_segments:
            break

    if not segments_x:
        LOGGER.warning(
            "No usable windows extracted from `%s`. Falling back to a synthetic dataset.",
            data_root,
        )
        return _synthetic_dataset(seq_len, val_split, rng, resolved_sample_rate or DEFAULT_SAMPLE_RATE)

    x = _stack_segments(segments_x)
    y = _stack_segments(segments_y)

    if x.shape[0] < 2:
        LOGGER.warning(
            "Only %d window(s) extracted. Need at least 2 for train/validation split; using synthetic data.",
            x.shape[0],
        )
        return _synthetic_dataset(seq_len, val_split, rng, resolved_sample_rate or DEFAULT_SAMPLE_RATE)

    x = x[..., np.newaxis].astype(np.float32)
    y = y[..., np.newaxis].astype(np.float32)

    dataset_size = x.shape[0]
    if limit_total_segments is not None:
        dataset_size = min(dataset_size, limit_total_segments)

    if dataset_size != x.shape[0]:
        x = x[:dataset_size]
        y = y[:dataset_size]

    indices = rng.permutation(dataset_size)
    x = x[indices]
    y = y[indices]

    val_count = max(1, int(round(dataset_size * val_split)))
    val_count = min(val_count, dataset_size - 1)

    return SequenceDataset(
        x_train=x[:-val_count],
        y_train=y[:-val_count],
        x_val=x[-val_count:],
        y_val=y[-val_count:],
        sample_rate=resolved_sample_rate or sample_rate or DEFAULT_SAMPLE_RATE,
        seq_len=seq_len,
    )


def _ensure_structure(data_root: Path) -> Tuple[Path, Path]:
    input_dir = data_root / "input"
    target_dir = data_root / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    return input_dir, target_dir


def _collect_wav_pairs(
    input_dir: Path,
    target_dir: Path,
    *,
    input_filename: Optional[str],
    output_filename: Optional[str],
) -> List[Tuple[Path, Path]]:
    if input_filename and output_filename:
        input_path = input_dir / input_filename
        target_path = target_dir / output_filename
        missing = [str(path) for path in (input_path, target_path) if not path.exists()]
        if missing:
            LOGGER.warning("Missing file(s): %s", ", ".join(missing))
            return []
        LOGGER.info(
            "Loaded explicit pair: `%s` -> `%s`.",
            input_path.name,
            target_path.name,
        )
        return [(input_path, target_path)]

    LOGGER.warning(
        "Input/output filenames not provided explicitly; attempting to match by identical names."
    )

    pairs: List[Tuple[Path, Path]] = []
    for input_path in sorted(input_dir.glob("*.wav")):
        candidate = target_dir / input_path.name
        if candidate.exists():
            pairs.append((input_path, candidate))

    if not pairs:
        LOGGER.warning(
            "No matching filenames between `%s` and `%s`.", input_dir, target_dir
        )
    else:
        LOGGER.info("Matched %d pair(s) by identical filenames.", len(pairs))
    return pairs


def _load_waveform(path: Path, enforce_sample_rate: Optional[int]) -> Tuple[np.ndarray, int]:
    audio_bytes = tf.io.read_file(str(path))
    waveform, sample_rate = tf.audio.decode_wav(audio_bytes)
    sr = int(sample_rate.numpy())

    if waveform.shape[-1] != 1:
        raise ValueError(f"WAV file `{path}` must be mono. Found {waveform.shape[-1]} channels.")

    if enforce_sample_rate is not None and sr != enforce_sample_rate:
        raise ValueError(
            f"Expected sample rate {enforce_sample_rate} Hz but got {sr} Hz when reading `{path}`."
        )

    waveform = tf.squeeze(tf.cast(waveform, tf.float32), axis=-1)
    return waveform.numpy(), sr


def _frame_waveform(waveform: np.ndarray, seq_len: int, hop_len: int) -> np.ndarray:
    tensor = tf.convert_to_tensor(waveform, dtype=tf.float32)
    frames = tf.signal.frame(tensor, seq_len, hop_len, pad_end=True, pad_value=0.0)
    return frames.numpy()


def _normalize_waveform(waveform: np.ndarray) -> np.ndarray:
    waveform = waveform.astype(np.float32)
    min_val = float(np.min(waveform))
    max_val = float(np.max(waveform))
    if np.isclose(max_val, min_val):
        return np.zeros_like(waveform, dtype=np.float32)
    normalized = (waveform - min_val) / (max_val - min_val)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def _stack_segments(segments: List[np.ndarray]) -> np.ndarray:
    if len(segments) == 1:
        return segments[0]
    return np.concatenate(segments, axis=0)


def _synthetic_dataset(
    seq_len: int,
    val_split: float,
    rng: np.random.Generator,
    sample_rate: int,
) -> SequenceDataset:
    num_samples = max(128, seq_len // 2)
    inputs = rng.normal(scale=0.5, size=(num_samples, seq_len)).astype(np.float32)

    # Simple FIR channel with small added noise.
    kernel = np.array([0.85, -0.3, 0.12], dtype=np.float32)
    targets = np.array(
        [np.convolve(row, kernel, mode="same") for row in inputs], dtype=np.float32
    )
    targets += 0.01 * rng.normal(size=targets.shape).astype(np.float32)

    inputs = _normalize_waveform(inputs)[..., np.newaxis]
    targets = _normalize_waveform(targets)[..., np.newaxis]

    indices = rng.permutation(num_samples)
    inputs = inputs[indices]
    targets = targets[indices]

    val_count = max(1, int(round(num_samples * val_split)))
    val_count = min(val_count, num_samples - 1)

    return SequenceDataset(
        x_train=inputs[:-val_count],
        y_train=targets[:-val_count],
        x_val=inputs[-val_count:],
        y_val=targets[-val_count:],
        sample_rate=sample_rate,
        seq_len=seq_len,
    )
