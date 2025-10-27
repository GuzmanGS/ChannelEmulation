from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from dataset import (
    SequenceDataset,
    compute_context_length,
    denormalize_from_unit_range,
    prepare_sequence_dataset,
)
from main import (
    DEFAULT_CONFIG_RELATIVE,
    DEFAULT_DATA_ROOT,
    coerce_dilations,
    load_config,
)


def _resolve_hop_length(seq_len: int, data_cfg: dict) -> int:
    overlap = data_cfg.get("overlap")
    hop_len_cfg = data_cfg.get("hop_len")

    if overlap is not None:
        overlap = float(overlap)
        if not 0 <= overlap < 1:
            raise ValueError("`overlap` must be in the range [0, 1).")
        return max(1, int(round(seq_len * (1.0 - overlap))))

    if hop_len_cfg is None:
        raise ValueError("Config must define either `hop_len` or `overlap`.")

    hop_len = int(hop_len_cfg)
    if hop_len <= 0:
        raise ValueError("`hop_len` must be a positive integer.")
    return hop_len


def _prepare_dataset(
    config_path: str,
    input_name: str,
    output_name: str,
) -> Tuple[SequenceDataset, dict, Path]:
    config, resolved_config = load_config(config_path)
    data_cfg = dict(config.get("data", {}))
    model_cfg = dict(config.get("model", {}))

    seq_len = int(data_cfg.get("seq_len", 0))
    if seq_len <= 0:
        raise ValueError("`seq_len` must be a positive integer.")

    hop_len = _resolve_hop_length(seq_len, data_cfg)
    val_split = float(data_cfg.get("val_split", 0.1))
    if not 0 < val_split < 1:
        raise ValueError("`val_split` must be between 0 and 1.")

    limit = data_cfg.get("limit")
    limit = None if limit is None else int(limit)

    dilations_cfg = model_cfg.get("dilations")
    dilations = coerce_dilations(dilations_cfg) if dilations_cfg is not None else [1, 2, 4, 8]
    kernel_size = int(model_cfg.get("kernel_size", 3))
    nb_stacks = int(model_cfg.get("nb_stacks", 1))
    context = compute_context_length(kernel_size, dilations, nb_stacks)

    dataset = prepare_sequence_dataset(
        data_root=DEFAULT_DATA_ROOT,
        seq_len=seq_len,
        hop_len=hop_len,
        context=context,
        val_split=val_split,
        limit_total_segments=limit,
        input_filename=input_name,
        output_filename=output_name,
    )
    return dataset, config, resolved_config


def choose_window(
    dataset: SequenceDataset,
    split: str,
    index: int | None,
    random_pick: bool,
) -> Tuple[np.ndarray, np.ndarray, int]:
    if split == "train":
        inputs, targets = dataset.x_train, dataset.y_train
    else:
        inputs, targets = dataset.x_val, dataset.y_val

    if len(inputs) == 0:
        raise ValueError(f"No windows available in the `{split}` split.")

    if random_pick:
        idx = random.randrange(len(inputs))
    else:
        idx = 0 if index is None else index
        if not 0 <= idx < len(inputs):
            raise IndexError(f"Requested index {idx} out of range for split `{split}` (size={len(inputs)}).")

    window_input = inputs[idx, :, 0]
    window_target = targets[idx, :, 0]
    return window_input, window_target, idx


def plot_window(
    window_input: np.ndarray,
    window_target: np.ndarray,
    *,
    sample_rate: int,
    context: int,
    denormalize: bool,
    title: str,
) -> None:
    if denormalize:
        display_input = denormalize_from_unit_range(window_input)
        display_target = denormalize_from_unit_range(window_target)
    else:
        display_input = window_input
        display_target = window_target

    time_input = np.arange(len(display_input), dtype=np.float32) / sample_rate
    context_time = context / sample_rate
    time_target = np.arange(len(display_target), dtype=np.float32) / sample_rate + context_time

    plt.figure(figsize=(12, 6))
    plt.plot(time_input, display_input, label="Input (context + frame)", color="tab:green", linewidth=1.0)
    plt.plot(time_target, display_target, label="Target frame", color="tab:orange", linewidth=1.0)

    if context > 0:
        plt.axvspan(
            0.0,
            context_time,
            color="tab:gray",
            alpha=0.1,
            label="Causal context",
        )

    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualise individual framed windows from the training dataset."
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_RELATIVE,
        help="JSON configuration used to prepare the dataset (default: %(default)s).",
    )
    parser.add_argument(
        "--input",
        default="rawAudio.wav",
        help="Input WAV filename located under data/input/ (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        default="fxAudioVFuzz.wav",
        help="Target WAV filename located under data/output/ (default: %(default)s).",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val"),
        default="train",
        help="Which split to draw the window from (default: %(default)s).",
    )
    parser.add_argument(
        "--index",
        type=int,
        help="Index of the window to display (0-based). Ignored when --random is set.",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Pick a random window from the selected split instead of using --index.",
    )
    parser.add_argument(
        "--keep-normalized",
        dest="denormalize",
        action="store_false",
        help="Display waveforms in their normalised [0, 1] range instead of converting back to [-1, 1].",
    )
    parser.set_defaults(denormalize=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset, config, resolved_config = _prepare_dataset(args.config, args.input, args.output)

    window_input, window_target, actual_index = choose_window(
        dataset,
        split=args.split,
        index=args.index,
        random_pick=args.random,
    )

    data_cfg = config.get("data", {})
    metadata = (
        f"{args.split} window #{actual_index} | "
        f"seq_len={data_cfg.get('seq_len')} | "
        f"context={dataset.context} | "
        f"sample_rate={dataset.sample_rate} Hz | "
        f"config={resolved_config.name}"
    )

    plot_window(
        window_input,
        window_target,
        sample_rate=dataset.sample_rate,
        context=dataset.context,
        denormalize=args.denormalize,
        title=metadata,
    )


if __name__ == "__main__":
    main()
