"""Run the full LDM pipeline: train autoencoder → train diffusion → sample → evaluate."""

from __future__ import annotations

import argparse
import sys
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the complete LDM pipeline end-to-end.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["train-ae", "eval-ae", "train-diff", "sample", "eval-diff"],
        choices=["train-ae", "eval-ae", "train-diff", "sample", "eval-diff"],
        help="Stages to run (default: all, in order).",
    )
    return parser.parse_args()


def _run_stage(name: str, func, config_path: str) -> None:
    print(f"\n{'='*60}")
    print(f"  STAGE: {name}")
    print(f"{'='*60}\n")
    start = time.time()
    # Patch sys.argv so each sub-main sees --config
    original_argv = sys.argv
    sys.argv = ["ldm", "--config", config_path]
    try:
        func()
    finally:
        sys.argv = original_argv
    elapsed = time.time() - start
    minutes, seconds = divmod(elapsed, 60)
    print(f"\n  ✓ {name} completed in {int(minutes)}m {seconds:.1f}s\n")


def main() -> None:
    args = parse_args()

    stage_map = {
        "train-ae": ("Train Autoencoder", lambda: __import__("ldm.train_autoencoder", fromlist=["main"]).main),
        "eval-ae": ("Evaluate Autoencoder", lambda: __import__("ldm.evaluate_autoencoder", fromlist=["main"]).main),
        "train-diff": ("Train Diffusion", lambda: __import__("ldm.train_diffusion", fromlist=["main"]).main),
        "sample": ("Sample", lambda: __import__("ldm.sample", fromlist=["main"]).main),
        "eval-diff": ("Evaluate Diffusion", lambda: __import__("ldm.evaluate_diffusion", fromlist=["main"]).main),
    }

    selected = args.stages
    print(f"LDM Pipeline — running {len(selected)} stage(s): {', '.join(selected)}")
    pipeline_start = time.time()

    for stage_key in selected:
        display_name, get_func = stage_map[stage_key]
        _run_stage(display_name, get_func(), args.config)

    total = time.time() - pipeline_start
    minutes, seconds = divmod(total, 60)
    hours, minutes = divmod(int(minutes), 60)
    print(f"{'='*60}")
    print(f"  Pipeline complete — total time: {hours}h {minutes}m {seconds:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
