from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf

DEFAULT_SAMPLE_RATE = 16_000

LOGGER = logging.getLogger(__name__)


def normalize_to_unit_range(waveform: np.ndarray) -> np.ndarray:
    """Map waveform values from [-1, 1] to [0, 1] without altering dynamics."""
    if waveform.ndim != 1:
        raise ValueError("Expected a 1-D waveform for normalization.")
    waveform = waveform.astype(np.float32, copy=False)
    normalized = 0.5 * (waveform + 1.0)
    return np.clip(normalized, 0.0, 1.0, out=normalized)


def denormalize_from_unit_range(waveform: np.ndarray) -> np.ndarray:
    """Map waveform values from [0, 1] back to [-1, 1]."""
    if waveform.ndim != 1:
        raise ValueError("Expected a 1-D waveform for denormalization.")
    waveform = waveform.astype(np.float32, copy=False)
    restored = 2.0 * waveform - 1.0
    return np.clip(restored, -1.0, 1.0, out=restored)


def compute_receptive_field(
    kernel_size: int,
    dilations: Sequence[int],
    nb_stacks: int,
) -> int:
    """Return the total receptive field (in samples) of the TCN."""
    if kernel_size < 1:
        raise ValueError("`kernel_size` must be a positive integer.")
    if nb_stacks < 1:
        raise ValueError("`nb_stacks` must be a positive integer.")
    if not dilations:
        raise ValueError("`dilations` must contain at least one value.")

    kernel_extent = max(0, kernel_size - 1)
    dilation_sum = sum(int(d) for d in dilations)
    if dilation_sum <= 0:
        raise ValueError("`dilations` must contain positive integers.")

    per_block = 2 * kernel_extent  # each residual block has two causal convolutions
    receptive_field = 1 + per_block * nb_stacks * dilation_sum
    return receptive_field


def compute_context_length(
    kernel_size: int,
    dilations: Sequence[int],
    nb_stacks: int,
) -> int:
    """Return the number of past samples required as causal context."""
    receptive_field = compute_receptive_field(kernel_size, dilations, nb_stacks)
    return max(0, receptive_field - 1)


def frame_with_context(
    waveform: np.ndarray,
    frame_length: int,
    frame_step: int,
    *,
    context: int = 0,
    pad_value: float = 0.0,
) -> np.ndarray:
    """Slice a 1-D waveform into contextual frames with optional zero causal padding."""
    if frame_length <= 0:
        raise ValueError("`frame_length` must be a positive integer.")
    if frame_step <= 0:
        raise ValueError("`frame_step` must be a positive integer.")
    if context < 0:
        raise ValueError("`context` must be non-negative.")

    waveform = waveform.astype(np.float32, copy=False)
    total_length = int(waveform.shape[0])
    if total_length == 0:
        return np.zeros((0, frame_length + context), dtype=np.float32)

    frame_count = int(np.ceil(total_length / frame_step))
    frames = np.full(
        (frame_count, frame_length + context),
        pad_value,
        dtype=np.float32,
    )

    for index in range(frame_count):
        start = index * frame_step
        end = start + frame_length
        indices = np.arange(start - context, end, dtype=np.int64)
        valid = (indices >= 0) & (indices < total_length)
        frames[index, valid] = waveform[indices[valid]]

    return frames


@dataclass
class SequenceDataset:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    sample_rate: int
    input_seq_len: int
    target_seq_len: int
    context: int


def prepare_sequence_dataset(
    data_root: Path,
    seq_len: int,
    hop_len: int,
    context: int,
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
    if context < 0:
        raise ValueError("`context` must be non-negative.")

    data_root = Path(data_root)
    input_dir, target_dir = _ensure_structure(data_root)
    rng = np.random.default_rng(seed)

    wav_pairs = _collect_wav_pairs(
        input_dir,
        target_dir,
        input_filename=input_filename,
        output_filename=output_filename,
    )
    resolved_sample_rate: Optional[int] = None

    if not wav_pairs:
        LOGGER.warning(
            "No WAV pairs found in `%s`. Falling back to a synthetic dataset.", data_root
        )
        return _synthetic_dataset(
            seq_len=seq_len,
            val_split=val_split,
            rng=rng,
            sample_rate=DEFAULT_SAMPLE_RATE,
            context=context,
        )

    segments_x: List[np.ndarray] = []
    segments_y: List[np.ndarray] = []
    total_segments = 0

    for input_path, target_path in wav_pairs:
        input_waveform, sr_in = _load_waveform(input_path, resolved_sample_rate)
        target_waveform, sr_out = _load_waveform(target_path, resolved_sample_rate or sr_in)

        input_waveform = normalize_to_unit_range(input_waveform)
        target_waveform = normalize_to_unit_range(target_waveform)

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

        input_frames = frame_with_context(
            input_waveform, seq_len, hop_len, context=context, pad_value=0.0
        )
        target_frames = frame_with_context(
            target_waveform, seq_len, hop_len, context=0, pad_value=0.0
        )

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
        return _synthetic_dataset(
            seq_len=seq_len,
            val_split=val_split,
            rng=rng,
            sample_rate=resolved_sample_rate or DEFAULT_SAMPLE_RATE,
            context=context,
        )

    x = _stack_segments(segments_x)
    y = _stack_segments(segments_y)

    if x.shape[0] < 2:
        LOGGER.warning(
            "Only %d window(s) extracted. Need at least 2 for train/validation split; using synthetic data.",
            x.shape[0],
        )
        return _synthetic_dataset(
            seq_len=seq_len,
            val_split=val_split,
            rng=rng,
            sample_rate=resolved_sample_rate or DEFAULT_SAMPLE_RATE,
            context=context,
        )

    x = x[..., np.newaxis].astype(np.float32)
    y = y[..., np.newaxis].astype(np.float32)

    dataset_size = x.shape[0]
    if limit_total_segments is not None:
        dataset_size = min(dataset_size, limit_total_segments)

    if dataset_size != x.shape[0]:
        x = x[:dataset_size]
        y = y[:dataset_size]

    indices = rng.permutation(dataset_size)
    LOGGER.info("Shuffling %d window(s) with RNG seed %d.", dataset_size, seed)
    x = x[indices]
    y = y[indices]

    val_count = max(1, int(round(dataset_size * val_split)))
    val_count = min(val_count, dataset_size - 1)

    input_seq_len = x.shape[1]
    target_seq_len = y.shape[1]

    return SequenceDataset(
        x_train=x[:-val_count],
        y_train=y[:-val_count],
        x_val=x[-val_count:],
        y_val=y[-val_count:],
        sample_rate=resolved_sample_rate or DEFAULT_SAMPLE_RATE,
        input_seq_len=int(input_seq_len),
        target_seq_len=int(target_seq_len),
        context=int(context),
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


def _stack_segments(segments: List[np.ndarray]) -> np.ndarray:
    if len(segments) == 1:
        return segments[0]
    return np.concatenate(segments, axis=0)


def _synthetic_dataset(
    seq_len: int,
    val_split: float,
    rng: np.random.Generator,
    sample_rate: int,
    context: int,
) -> SequenceDataset:
    num_samples = max(128, seq_len // 2)
    input_seq_len = seq_len + max(0, context)
    raw_inputs = rng.normal(scale=0.5, size=(num_samples, input_seq_len)).astype(np.float32)
    raw_inputs = np.clip(raw_inputs, -1.0, 1.0, out=raw_inputs)

    # Simple FIR channel with small added noise.
    kernel = np.array([0.85, -0.3, 0.12], dtype=np.float32)
    raw_targets = np.array(
        [np.convolve(row[:seq_len], kernel, mode="same") for row in raw_inputs],
        dtype=np.float32,
    )
    raw_targets += 0.01 * rng.normal(size=raw_targets.shape).astype(np.float32)
    raw_targets = np.clip(raw_targets, -1.0, 1.0, out=raw_targets)

    inputs = np.stack([normalize_to_unit_range(row) for row in raw_inputs], axis=0)[..., np.newaxis]
    targets = np.stack([normalize_to_unit_range(row) for row in raw_targets], axis=0)[..., np.newaxis]

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
        input_seq_len=input_seq_len,
        target_seq_len=seq_len,
        context=max(0, context),
    )
