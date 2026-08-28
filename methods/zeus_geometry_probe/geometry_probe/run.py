"""CLI for the controlled ZEUS target-only comparison."""

from __future__ import annotations

import argparse
import sys
from dataclasses import fields
from pathlib import Path

ZEUS_ROOT = Path(__file__).resolve().parents[1]
if str(ZEUS_ROOT) not in sys.path:
    sys.path.insert(0, str(ZEUS_ROOT))

from geometry_probe.core import ProbeConfig, run_probe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare ZEUS and geometry objectives with matched ZEUS tasks.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=ProbeConfig.seed)
    parser.add_argument("--train-steps", type=int, default=ProbeConfig.train_steps)
    parser.add_argument("--eval-tasks", type=int, default=ProbeConfig.eval_tasks)
    parser.add_argument("--train-mode", default=ProbeConfig.train_mode)
    parser.add_argument("--num-gaussians", type=int, default=ProbeConfig.num_gaussians)
    parser.add_argument("--min-points", type=int, default=ProbeConfig.min_points)
    parser.add_argument("--max-points", type=int, default=ProbeConfig.max_points)
    parser.add_argument("--dim", type=int, default=ProbeConfig.dim)
    parser.add_argument("--min-distance", type=float, default=ProbeConfig.min_distance)
    parser.add_argument("--eigenvalue-p1", type=float, default=ProbeConfig.eigenvalue_p1)
    parser.add_argument("--eigenvalue-p2", type=float, default=ProbeConfig.eigenvalue_p2)
    parser.add_argument("--start-distance", type=float, default=ProbeConfig.start_distance)
    parser.add_argument("--max-blocks", type=int, default=ProbeConfig.max_blocks)
    parser.add_argument("--num-categorical", type=int, default=ProbeConfig.num_categorical)
    parser.add_argument("--max-categories", type=int, default=ProbeConfig.max_categories)
    parser.add_argument("--categorical-chance", type=float, default=ProbeConfig.categorical_chance)
    parser.add_argument("--embed-dim", type=int, default=ProbeConfig.embed_dim)
    parser.add_argument("--n-head", type=int, default=ProbeConfig.n_head)
    parser.add_argument("--hid-dim", type=int, default=ProbeConfig.hid_dim)
    parser.add_argument("--n-layers", type=int, default=ProbeConfig.n_layers)
    parser.add_argument("--dropout", type=float, default=ProbeConfig.dropout)
    parser.add_argument("--learning-rate", type=float, default=ProbeConfig.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=ProbeConfig.weight_decay)
    parser.add_argument("--knn-k", type=int, default=ProbeConfig.knn_k)
    parser.add_argument("--device", default=ProbeConfig.device)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_fields = {field.name for field in fields(ProbeConfig)}
    values = {name: getattr(args, name) for name in config_fields}
    result = run_probe(ProbeConfig(**values), args.output)
    for arm in ("zeus", "geometry"):
        for mode in ("gaussian", "gaussian_transformed"):
            summary = result["arms"][arm]["summary"][mode]
            print(
                f"{arm} {mode}: ARI={summary['ari_representation']['mean']:.4f}, "
                f"CKA={summary['cka_to_x_ref']['mean']:.4f}, "
                f"kNN={summary['knn_overlap_to_x_ref']['mean']:.4f}"
            )


if __name__ == "__main__":
    main()
