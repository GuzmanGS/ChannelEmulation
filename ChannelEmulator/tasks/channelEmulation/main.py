import argparse
import json
import logging
from pathlib import Path
from typing import List, Sequence, Union

import tensorflow as tf

from dataset import SequenceDataset, prepare_sequence_dataset
from tcn import TCN, tcn_full_summary

LOGGER = logging.getLogger(__name__)

TASK_ROOT = Path(__file__).parent.resolve()
DEFAULT_CONFIG_RELATIVE = "configs/wavenet3.json"
DEFAULT_CONFIG_PATH = TASK_ROOT / DEFAULT_CONFIG_RELATIVE
DEFAULT_DATA_ROOT = TASK_ROOT / "data"
DEFAULT_INPUT_DIR = DEFAULT_DATA_ROOT / "input"
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_ROOT / "output"
DEFAULT_SAVE_DIR = TASK_ROOT / "savedModels"
DEFAULT_INPUT_FILE = "rawAudio.wav"
DEFAULT_OUTPUT_FILE = "fxAudioVFuzz.wav"


def parse_dilations(values: str) -> List[int]:
    parts = [item.strip() for item in values.split(",") if item.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("Provide at least one dilation value.")
    try:
        dilations = [int(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Dilations must be integers separated by commas."
        ) from exc
    if any(d <= 0 for d in dilations):
        raise argparse.ArgumentTypeError("Dilations must be positive integers.")
    return dilations


def load_config(config_path: Union[str, Path]) -> tuple[dict, Path]:
    raw_path = Path(config_path).expanduser()
    candidates = []

    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.append(Path.cwd() / raw_path)
        candidates.append(TASK_ROOT / raw_path)
        candidates.append(TASK_ROOT / "configs" / raw_path.name)
        candidates.append(DEFAULT_CONFIG_PATH)

    for candidate in candidates:
        if candidate.exists():
            resolved = candidate.resolve()
            with resolved.open("r", encoding="utf-8") as handle:
                return json.load(handle), resolved

    raise FileNotFoundError(f"Unable to locate configuration file: {config_path}")


def coerce_dilations(value: Union[str, Sequence[int]]) -> List[int]:
    if value is None:
        raise ValueError("Dilation configuration is missing.")
    if isinstance(value, str):
        return parse_dilations(value)
    if isinstance(value, Sequence):
        return [int(item) for item in value]
    raise TypeError(f"Unsupported dilation configuration type: {type(value)!r}")


def build_model(
    seq_len: int,
    input_dim: int,
    output_dim: int,
    *,
    nb_filters: int,
    kernel_size: int,
    dilations: List[int],
    nb_stacks: int,
    dropout: float,
    padding: str,
    learning_rate: float,
    use_skip_connections: bool,
):
    inputs = tf.keras.Input(shape=(seq_len, input_dim), name="input_waveform")
    tcn_layer = TCN(
        nb_filters=nb_filters,
        kernel_size=kernel_size,
        nb_stacks=nb_stacks,
        dilations=tuple(dilations),
        padding=padding,
        use_skip_connections=use_skip_connections,
        dropout_rate=dropout,
        return_sequences=True,
        name="tcn_backbone",
    )
    x = tcn_layer(inputs)
    outputs = tf.keras.layers.Dense(
        output_dim, activation="linear", name="channel_output"
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="channel_emulation_tcn",
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss="mean_squared_error")
    return model


def run(args: argparse.Namespace):
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s"
    )
    config, resolved_config_path = load_config(args.config)
    if not isinstance(config, dict):
        raise ValueError("Configuration file must contain a JSON object.")

    data_cfg = dict(config.get("data", {}))
    model_cfg = dict(config.get("model", {}))
    training_cfg = dict(config.get("training", {}))

    if args.input is not None:
        data_cfg["input_file"] = args.input
    if args.output is not None:
        data_cfg["output_file"] = args.output

    seq_len = int(data_cfg.get("seq_len", 0))
    hop_len = int(data_cfg.get("hop_len", 0))
    if seq_len <= 0 or hop_len <= 0:
        raise ValueError("`seq_len` and `hop_len` must be positive integers.")

    sample_rate = data_cfg.get("sample_rate")
    if sample_rate is not None:
        sample_rate = int(sample_rate)
        if sample_rate <= 0:
            raise ValueError("`sample_rate` must be a positive integer when provided.")

    val_split = float(data_cfg.get("val_split", 0.1))
    if not 0 < val_split < 1:
        raise ValueError("`val_split` must be between 0 and 1.")

    limit = data_cfg.get("limit")
    limit = None if limit is None else int(limit)
    input_file = data_cfg.get("input_file", DEFAULT_INPUT_FILE)
    output_file = data_cfg.get("output_file", DEFAULT_OUTPUT_FILE)

    if not input_file or not output_file:
        raise ValueError("Both `input_file` and `output_file` must be specified.")

    dilations_cfg = model_cfg.get("dilations")
    dilations = coerce_dilations(dilations_cfg) if dilations_cfg is not None else [1, 2, 4, 8]
    nb_filters = int(model_cfg.get("nb_filters", 32))
    kernel_size = int(model_cfg.get("kernel_size", 3))
    nb_stacks = int(model_cfg.get("nb_stacks", 1))
    dropout = float(model_cfg.get("dropout", 0.0))
    padding = model_cfg.get("padding", "causal")
    use_skip_connections = bool(model_cfg.get("use_skip_connections", True))

    learning_rate = float(training_cfg.get("learning_rate", 0.001))
    batch_size = int(training_cfg.get("batch_size", 32))
    epochs = int(training_cfg.get("epochs", 50))
    patience = int(training_cfg.get("patience", 10))
    seed = int(training_cfg.get("seed", 42))

    save_model = (DEFAULT_SAVE_DIR / f"{resolved_config_path.stem}.keras").resolve()

    LOGGER.info(
        "Preparing dataset from `%s` using config `%s`...",
        DEFAULT_DATA_ROOT,
        resolved_config_path,
    )

    tf.random.set_seed(seed)
    dataset: SequenceDataset = prepare_sequence_dataset(
        data_root=DEFAULT_DATA_ROOT,
        seq_len=seq_len,
        hop_len=hop_len,
        sample_rate=sample_rate,
        val_split=val_split,
        limit_total_segments=limit,
        seed=seed,
        input_filename=input_file,
        output_filename=output_file,
    )

    LOGGER.info(
        "Dataset ready | sample_rate=%d Hz | train=%d | val=%d",
        dataset.sample_rate,
        len(dataset.x_train),
        len(dataset.x_val),
    )

    model = build_model(
        seq_len=dataset.seq_len,
        input_dim=dataset.x_train.shape[-1],
        output_dim=dataset.y_train.shape[-1],
        nb_filters=nb_filters,
        kernel_size=kernel_size,
        dilations=dilations,
        nb_stacks=nb_stacks,
        dropout=dropout,
        padding=padding,
        learning_rate=learning_rate,
        use_skip_connections=use_skip_connections,
    )

    try:
        tcn_full_summary(model)
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.warning("Falling back to `model.summary()` due to: %s", exc)
        model.summary()

    callbacks = []
    if len(dataset.x_val):
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=patience,
                restore_best_weights=True,
            )
        )

    fit_kwargs = dict(
        x=dataset.x_train,
        y=dataset.y_train,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        shuffle=True,
    )

    if len(dataset.x_val):
        fit_kwargs["validation_data"] = (dataset.x_val, dataset.y_val)

    history = model.fit(**fit_kwargs)

    if save_model:
        save_path = Path(save_model)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(save_path))
        LOGGER.info("Saved model to `%s`.", save_path)

    return history


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a seq2seq TCN to emulate a channel using WAV pairs."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_RELATIVE,
        help="JSON configuration file path (relative paths searched in the configs folder).",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input WAV filename located under the data/input directory.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Target WAV filename located under the data/output directory.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
