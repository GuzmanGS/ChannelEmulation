import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import List, Optional, Sequence, Union

import tensorflow as tf

from dataset import (
    SequenceDataset,
    compute_context_length,
    prepare_sequence_dataset,
)
from tcn import TCN, tcn_full_summary

LOGGER = logging.getLogger(__name__)

TASK_ROOT = Path(__file__).parent.resolve()
DEFAULT_CONFIG_RELATIVE = "configs/wavenet3.json"
DEFAULT_CONFIG_PATH = TASK_ROOT / DEFAULT_CONFIG_RELATIVE
DEFAULT_DATA_ROOT = TASK_ROOT / "data"
DEFAULT_SAVE_DIR = TASK_ROOT / "savedModels"

warnings.filterwarnings(
    "ignore",
    message=".*tf\\.placeholder is deprecated.*",
)


def _validate_pre_emphasis(coefficient: float) -> float:
    if not isinstance(coefficient, (float, int)):
        raise TypeError("Pre-emphasis coefficient must be a real number.")
    coeff = float(coefficient)
    if not 0.0 <= coeff < 1.0:
        raise ValueError("Pre-emphasis coefficient must be in the range [0.0, 1.0).")
    return coeff


def apply_pre_emphasis(signal: tf.Tensor, coefficient: float) -> tf.Tensor:
    """Apply H(z) = 1 - coefficient * z^-1 along the time axis."""
    coeff = _validate_pre_emphasis(coefficient)
    tensor = tf.convert_to_tensor(signal, dtype=tf.float32)

    rank = tensor.shape.rank
    if rank is None:
        raise ValueError("Pre-emphasis expects a tensor with known rank.")
    if rank < 2 or rank > 3:
        raise ValueError(
            "Pre-emphasis expects shape (batch, time) or (batch, time, channels); "
            f"received a rank-{rank} tensor."
        )

    added_channel = False
    if rank == 2:
        tensor = tensor[..., tf.newaxis]
        added_channel = True

    first = tensor[:, :1, :]
    remaining = tensor[:, 1:, :] - coeff * tensor[:, :-1, :]
    emphasized = tf.concat([first, remaining], axis=1)

    if added_channel:
        emphasized = tf.squeeze(emphasized, axis=-1)

    return emphasized


class PreEmphasisESRLoss(tf.keras.losses.Loss):
    def __init__(self, *, coefficient: float, epsilon: float = 1e-8, name: str = "pre_emphasis_esr"):
        coeff = _validate_pre_emphasis(coefficient)
        if epsilon <= 0.0:
            raise ValueError("Epsilon must be positive to stabilise the ESR denominator.")
        super().__init__(name=name)
        self.coefficient = coeff
        self.epsilon = float(epsilon)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        target = apply_pre_emphasis(y_true, self.coefficient)
        predicted = apply_pre_emphasis(y_pred, self.coefficient)
        error = target - predicted
        axes = tf.range(1, tf.rank(error))
        numerator = tf.reduce_sum(tf.square(error), axis=axes)
        denominator = tf.reduce_sum(tf.square(target), axis=axes)
        ratio = numerator / (denominator + self.epsilon)
        return tf.reduce_mean(ratio)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"coefficient": self.coefficient, "epsilon": self.epsilon})
        return config


@tf.keras.utils.register_keras_serializable(package="channel_emulation")
class ESRMSELoss(tf.keras.losses.Loss):
    """Combination of ESR (with pre-emphasis) and time-domain MSE."""

    def __init__(
        self,
        *,
        coefficient: float,
        epsilon: float = 1e-8,
        esr_weight: float = 0.5,
        mse_weight: float = 0.5,
        name: str = "esr_mse_loss",
    ):
        super().__init__(name=name)
        self.coefficient = _validate_pre_emphasis(coefficient)
        if epsilon <= 0.0:
            raise ValueError("Epsilon must be positive to stabilise the ESR denominator.")
        self.epsilon = float(epsilon)
        self.esr_weight = float(esr_weight)
        self.mse_weight = float(mse_weight)
        self._esr = PreEmphasisESRLoss(coefficient=self.coefficient, epsilon=self.epsilon)
        self._mse = tf.keras.losses.MeanSquaredError()

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        esr_value = self._esr(y_true, y_pred)
        mse_value = self._mse(y_true, y_pred)
        return self.esr_weight * esr_value + self.mse_weight * mse_value

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "coefficient": self.coefficient,
                "epsilon": self.epsilon,
                "esr_weight": self.esr_weight,
                "mse_weight": self.mse_weight,
            }
        )
        return config


@tf.keras.utils.register_keras_serializable(package="channel_emulation")
class ModelMetadata(tf.keras.layers.Layer):
    """Identity layer that stores training metadata (sample rate, config)."""

    def __init__(
        self,
        *,
        sample_rate: int,
        config: Optional[dict] = None,
        config_json: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if sample_rate is None:
            raise ValueError("`sample_rate` must be provided for ModelMetadata.")
        self.sample_rate = int(sample_rate)
        if config is not None:
            self.config_json = json.dumps(config, sort_keys=True)
        elif config_json is not None:
            self.config_json = str(config_json)
        else:
            raise ValueError("Either `config` or `config_json` must be provided for ModelMetadata.")

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return tf.identity(inputs)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                "sample_rate": self.sample_rate,
                "config_json": self.config_json,
            }
        )
        return config

    @property
    def config_dict(self) -> dict:
        return json.loads(self.config_json)


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
            # Use 'utf-8-sig' to be tolerant of BOM-prefixed JSON files (common on Windows editors)
            with resolved.open("r", encoding="utf-8-sig") as handle:
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
    input_seq_len: int,
    target_seq_len: int,
    input_dim: int,
    output_dim: int,
    *,
    context: int,
    nb_filters: int,
    kernel_size: int,
    dilations: List[int],
    nb_stacks: int,
    dropout: float,
    padding: str,
    learning_rate: float,
    use_skip_connections: bool,
    pre_emphasis: float,
    loss_epsilon: float = 1e-8,
    sample_rate: int,
    config_metadata: dict,
):
    inputs = tf.keras.Input(shape=(input_seq_len, input_dim), name="input_waveform")

    # Linear projection of the mono input (no bias, no activation).
    x = tf.keras.layers.Conv1D(
        filters=input_dim,
        kernel_size=1,
        activation=None,
        use_bias=True,
        name="input_linear_projection",
    )(inputs)

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
    x = tcn_layer(x)

    # Collapse filters back to a single channel via linear projection with bias.
    x = tf.keras.layers.Conv1D(
        filters=output_dim,
        kernel_size=1,
        activation=None,
        use_bias=True,
        name="output_linear_projection",
    )(x)

    if context > 0:
        x = tf.keras.layers.Cropping1D(cropping=(context, 0), name="context_crop")(x)
    outputs = ModelMetadata(
        sample_rate=sample_rate,
        config=config_metadata,
        name="model_metadata",
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="channel_emulation_tcn",
    )
    model.sample_rate = int(sample_rate)
    model.config_metadata = config_metadata
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
    loss = ESRMSELoss(
        coefficient=pre_emphasis,
        epsilon=loss_epsilon,
        esr_weight=0.5,
        mse_weight=0.5,
    )
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[
            PreEmphasisESRLoss(coefficient=pre_emphasis, epsilon=loss_epsilon, name="esr"),
            tf.keras.metrics.MeanSquaredError(name="mse"),
        ],
    )
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

    seq_len = int(data_cfg.get("seq_len", 0))
    if seq_len <= 0:
        raise ValueError("`seq_len` must be a positive integer.")

    hop_len_cfg = data_cfg.get("hop_len")
    overlap = data_cfg.get("overlap")
    if overlap is not None:
        overlap = float(overlap)
        if not 0 <= overlap < 1:
            raise ValueError("`overlap` must be in the range [0, 1).")
        hop_len = max(1, int(round(seq_len * (1.0 - overlap))))
    elif hop_len_cfg is not None:
        hop_len = int(hop_len_cfg)
        if hop_len <= 0:
            raise ValueError("`hop_len` must be a positive integer.")
    else:
        raise ValueError("Provide either `hop_len` or `overlap` in the config.")

    val_split = float(data_cfg.get("val_split", 0.1))
    if not 0 < val_split < 1:
        raise ValueError("`val_split` must be between 0 and 1.")

    limit = data_cfg.get("limit")
    limit = None if limit is None else int(limit)
    input_file = args.input
    output_file = args.output

    if not input_file or not output_file:
        raise ValueError("Both `--input` and `--output` must be provided.")

    dilations_cfg = model_cfg.get("dilations")
    dilations = coerce_dilations(dilations_cfg) if dilations_cfg is not None else [1, 2, 4, 8]
    nb_filters = int(model_cfg.get("nb_filters", 32))
    kernel_size = int(model_cfg.get("kernel_size", 3))
    nb_stacks = int(model_cfg.get("nb_stacks", 1))
    dropout = float(model_cfg.get("dropout", 0.0))
    padding = model_cfg.get("padding", "causal")
    use_skip_connections = bool(model_cfg.get("use_skip_connections", True))
    context = compute_context_length(kernel_size, dilations, nb_stacks)

    data_metadata = {
        "seq_len": seq_len,
        "input_seq_len": seq_len + context,
        "context": context,
        "hop_len": hop_len,
        "overlap": overlap,
        "val_split": val_split,
        "limit": limit,
    }

    learning_rate = float(training_cfg.get("learning_rate", 0.001))
    batch_size = int(training_cfg.get("batch_size", 32))
    epochs = int(training_cfg.get("epochs", 50))
    patience = int(training_cfg.get("patience", 10))
    pre_emphasis = float(training_cfg.get("pre_emphasis", 0.95))
    loss_epsilon = float(training_cfg.get("loss_epsilon", 1e-8))
    seed = int(training_cfg.get("seed", 42))

    save_model = (DEFAULT_SAVE_DIR / f"{resolved_config_path.stem}.keras").resolve()

    LOGGER.info(
        "Preparing dataset from `%s` using config `%s` (hop_len=%d, overlap=%.2f)...",
        DEFAULT_DATA_ROOT,
        resolved_config_path,
        hop_len,
        1.0 - (hop_len / seq_len),
    )

    tf.random.set_seed(seed)
    dataset: SequenceDataset = prepare_sequence_dataset(
        data_root=DEFAULT_DATA_ROOT,
        seq_len=seq_len,
        hop_len=hop_len,
        context=context,
        val_split=val_split,
        limit_total_segments=limit,
        seed=seed,
        input_filename=input_file,
        output_filename=output_file,
    )

    data_metadata["sample_rate"] = dataset.sample_rate
    data_metadata["input_seq_len"] = dataset.input_seq_len
    data_metadata["target_seq_len"] = dataset.target_seq_len
    data_metadata["context"] = dataset.context

    model_metadata_cfg = {
        "nb_filters": nb_filters,
        "kernel_size": kernel_size,
        "dilations": list(dilations),
        "nb_stacks": nb_stacks,
        "dropout": dropout,
        "padding": padding,
        "use_skip_connections": use_skip_connections,
    }

    training_metadata_cfg = {
        "batch_size": batch_size,
        "epochs": epochs,
        "patience": patience,
        "learning_rate": learning_rate,
        "pre_emphasis": pre_emphasis,
        "loss_epsilon": loss_epsilon,
        "seed": seed,
        "optimizer": "Adam",
    }

    metadata_payload = {
        "config_path": str(resolved_config_path),
        "data": data_metadata,
        "model": model_metadata_cfg,
        "training": training_metadata_cfg,
        "io": {"input": input_file, "output": output_file},
    }

    LOGGER.info(
        "Dataset ready | sample_rate=%d Hz | train=%d | val=%d | input_seq_len=%d | target_seq_len=%d | context=%d",
        dataset.sample_rate,
        len(dataset.x_train),
        len(dataset.x_val),
        dataset.input_seq_len,
        dataset.target_seq_len,
        dataset.context,
    )

    model = build_model(
        input_seq_len=dataset.input_seq_len,
        target_seq_len=dataset.target_seq_len,
        input_dim=dataset.x_train.shape[-1],
        output_dim=dataset.y_train.shape[-1],
        context=dataset.context,
        nb_filters=nb_filters,
        kernel_size=kernel_size,
        dilations=dilations,
        nb_stacks=nb_stacks,
        dropout=dropout,
        padding=padding,
        learning_rate=learning_rate,
        use_skip_connections=use_skip_connections,
        pre_emphasis=pre_emphasis,
        loss_epsilon=loss_epsilon,
        sample_rate=dataset.sample_rate,
        config_metadata=metadata_payload,
    )

    tf_version = tf.version.VERSION.split("-")[0]
    version_parts = [int(part) for part in tf_version.split(".")[:2]]
    major, minor = (version_parts + [0])[:2]
    if major < 2 or (major == 2 and minor <= 5):
        try:
            tcn_full_summary(model)
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.warning("Falling back to `model.summary()` due to: %s", exc)
            model.summary()
    else:
        LOGGER.info(
            "Skipping `tcn_full_summary` for TensorFlow %s; using `model.summary()` instead.",
            tf.version.VERSION,
        )
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

    # Solo añadir validation_data si hay datos de validación
    if len(dataset.x_val) > 0:
        fit_kwargs["validation_data"] = (dataset.x_val, dataset.y_val)
    else:
        LOGGER.info("Training without validation data (val_split=0)")

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
        required=True,
        help="Input WAV filename located under the data/input directory.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output WAV filename located under the data/output directory.",
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
