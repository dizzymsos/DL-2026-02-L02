"""
Punto de entrada del Laboratorio 02: Conv1D para clasificacion multiclase
categorica (conv1d_softmax) y ordinal (conv1d_coral).

Ejemplo:
    python main.py --data-path "dataset/15 atributos R0-R5.sav" --epochs 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.config import (
    ALGORITHMS,
    DEFAULT_ALGORITHM,
    DEFAULT_BATCH_SIZE,
    DEFAULT_BETA,
    DEFAULT_CONV_CHANNELS,
    DEFAULT_DROPOUT,
    DEFAULT_EPOCHS,
    DEFAULT_GDS_INNER_FOLDS,
    DEFAULT_GDS_OUTER_FOLDS,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_INNER_FOLDS,
    DEFAULT_KERNEL_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_OUTER_FOLDS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RANDOM_SEED,
    DEFAULT_RANK_METRIC,
    DEFAULT_RANK_MODE,
    DEFAULT_TARGET,
    DEFAULT_WEIGHT_DECAY,
    TARGET_COLUMNS,
    ExperimentConfig,
    ModelConfig,
    TrainingConfig,
    ValidationConfig,
)
from src.evaluation.metrics import METRIC_KEYS
from src.experiment import ExperimentRunner


def build_argument_parser() -> argparse.ArgumentParser:
    """Define los argumentos de linea de comandos."""

    parser = argparse.ArgumentParser(
        description=(
            "Laboratorio 02: Conv1D + MLP con validacion anidada y busqueda "
            "de hiperparametros. conv1d_softmax esta implementado; "
            "conv1d_coral queda como trabajo del alumno."
        )
    )

    data = parser.add_argument_group("Datos y experimentos")
    data.add_argument("--data-path", required=True, help="Archivo .csv o .sav dentro de dataset/.")
    data.add_argument(
        "--target-name",
        default=DEFAULT_TARGET,
        choices=TARGET_COLUMNS,
        help=f"Objetivo de un experimento suelto (defecto {DEFAULT_TARGET}).",
    )
    data.add_argument(
        "--all-targets",
        action="store_true",
        help="Ejecuta GDS y GDS_R1 ... GDS_R5 (GDS usa --gds-*-folds).",
    )
    data.add_argument(
        "--algorithm",
        default=DEFAULT_ALGORITHM,
        choices=[*ALGORITHMS, "all"],
        help="conv1d_softmax, conv1d_coral o all (ambos).",
    )

    model = parser.add_argument_group("Modelo (configuracion base)")
    model.add_argument(
        "--conv-channels",
        type=int,
        nargs="+",
        default=list(DEFAULT_CONV_CHANNELS),
        help="Canales de cada bloque Conv1D. Ej: --conv-channels 16 32.",
    )
    model.add_argument("--kernel-size", type=int, default=DEFAULT_KERNEL_SIZE, help="Kernel impar.")
    model.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM, help="Neuronas del MLP.")
    model.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT, help="Dropout del MLP.")

    training = parser.add_argument_group("Entrenamiento")
    training.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    training.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    training.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    training.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Epocas por fold.")
    training.add_argument(
        "--use-weights",
        action="store_true",
        help="Pesos por numero efectivo de muestras (solo conv1d_coral).",
    )
    training.add_argument("--beta", type=float, default=DEFAULT_BETA, help="Beta del numero efectivo.")

    validation = parser.add_argument_group("Validacion")
    validation.add_argument("--outer-folds", type=int, default=DEFAULT_OUTER_FOLDS)
    validation.add_argument("--inner-folds", type=int, default=DEFAULT_INNER_FOLDS)
    validation.add_argument("--gds-outer-folds", type=int, default=DEFAULT_GDS_OUTER_FOLDS)
    validation.add_argument("--gds-inner-folds", type=int, default=DEFAULT_GDS_INNER_FOLDS)
    validation.add_argument(
        "--no-grid",
        action="store_true",
        help="No recorre HYPERPARAMETER_GRID: usa solo la configuracion base.",
    )

    output = parser.add_argument_group("Ejecucion y reportes")
    output.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    output.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    output.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    output.add_argument("--rank-metric", default=DEFAULT_RANK_METRIC, choices=METRIC_KEYS)
    output.add_argument("--rank-mode", default=DEFAULT_RANK_MODE, choices=["auto", "max", "min"])
    return parser


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    """Traduce los argumentos de la CLI a un ExperimentConfig."""

    algorithms = ALGORITHMS if args.algorithm == "all" else (args.algorithm,)
    targets = tuple(TARGET_COLUMNS) if args.all_targets else (args.target_name,)
    return ExperimentConfig(
        data_path=Path(args.data_path),
        targets=targets,
        algorithms=tuple(algorithms),
        all_targets=args.all_targets,
        model=ModelConfig(
            conv_channels=tuple(args.conv_channels),
            kernel_size=args.kernel_size,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
        ),
        training=TrainingConfig(
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            epochs=args.epochs,
            use_class_weights=args.use_weights,
            beta=args.beta,
        ),
        validation=ValidationConfig(
            outer_folds=args.outer_folds,
            inner_folds=args.inner_folds,
            gds_outer_folds=args.gds_outer_folds,
            gds_inner_folds=args.gds_inner_folds,
            use_grid=not args.no_grid,
        ),
        seed=args.seed,
        device=args.device,
        output_dir=Path(args.output_dir),
        rank_metric=args.rank_metric,
        rank_mode=args.rank_mode,
    )


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()
    try:
        config = config_from_args(args)
    except ValueError as error:
        parser.error(str(error))
    results = ExperimentRunner(config).run()
    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())
