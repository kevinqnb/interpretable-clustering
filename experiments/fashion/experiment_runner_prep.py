#!/usr/bin/env python3
####################################################################################################
# Prep-only runner for the fashion experiment pipeline: runs mine_rules.py ->
# [alphas.py || ids_lambda_search.py] -> select_alphas.py at the single default
# confidence threshold defined in config.py, then stops -- it does NOT run
# max_rules.py, lambda.py, or confidence.py (or their companion _exkmc/_combine
# scripts). Use this when you want to run those three as their own individual
# experiments (e.g. to time/debug them separately, or split them across
# machines) rather than as part of one end-to-end experiment_runner.py invocation.
#
# After this script finishes, run max_rules.py/max_rules_exkmc.py/max_rules_combine.py,
# lambda.py/lambda_exkmc.py/lambda_combine.py, and confidence.py directly, e.g.:
#   uv run python experiments/fashion/max_rules.py
#   uv run python experiments/fashion/max_rules_exkmc.py
#   uv run python experiments/fashion/max_rules_combine.py
#   uv run python experiments/fashion/confidence.py
#
# Each stage is a standalone module-level script (not an importable function), so
# stages are driven via subprocess rather than direct function calls.

import argparse
import os
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
PROJECT_ROOT = next((p for p in _HERE.parents if (p / "data").is_dir()), None)
if PROJECT_ROOT is None:
    raise ModuleNotFoundError("Could not locate repository root.")
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.fashion.config import TOTAL_CPU_COUNT

FASHION_DIR = PROJECT_ROOT / "experiments" / "fashion"


def _spawn(script_name: str, env: dict) -> subprocess.Popen:
    print(f"=== launching {script_name} ===")
    # cwd=PROJECT_ROOT: every stage script does file I/O against repo-root-relative
    # paths (e.g. 'data/experiments/fashion/rules/') and never chdir's itself.
    return subprocess.Popen(
        [sys.executable, str(FASHION_DIR / script_name)],
        cwd=str(PROJECT_ROOT),
        env=env,
    )


def _wait(procs_with_names):
    failures = []
    for proc, name in procs_with_names:
        ret = proc.wait()
        if ret != 0:
            failures.append((name, ret))
    if failures:
        raise RuntimeError(f"Stage(s) failed: {failures}")


def _run_stage(scripts: list[str], total_cpu_count: int) -> None:
    """
    Runs `scripts` concurrently, each getting an equal share of `total_cpu_count`
    (via the FASHION_CPU_COUNT env var, which config.py's CPU_COUNT reads).
    """
    per_script = max(1, total_cpu_count // len(scripts))
    env = dict(os.environ) | {"FASHION_CPU_COUNT": str(per_script)}
    print(f"--- stage {scripts}: {per_script} cores/script (budget {total_cpu_count}) ---")
    _wait([(_spawn(s, env), s) for s in scripts])


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run the fashion pipeline's prep stages only (mine_rules -> "
            "[alphas || ids_lambda_search] -> select_alphas), stopping before "
            "max_rules.py/lambda.py/confidence.py."
        )
    )
    parser.add_argument(
        "--total-cpu-count", type=int, default=None,
        help=(
            "Total CPU budget for the prep stages. At each stage below, this is "
            "divided evenly across however many scripts run concurrently at "
            "that point so no stage oversubscribes the machine. Defaults to "
            "experiments/fashion/config.py's TOTAL_CPU_COUNT, which itself "
            "defaults to the machine's detected core count."
        ),
    )
    args = parser.parse_args()
    total_cpu_count = args.total_cpu_count if args.total_cpu_count is not None else TOTAL_CPU_COUNT

    _run_stage(["mine_rules.py"], total_cpu_count)
    _run_stage(["alphas.py", "ids_lambda_search.py"], total_cpu_count)
    _run_stage(["select_alphas.py"], total_cpu_count)

    print("=== fashion prep stages complete -- run max_rules.py/lambda.py/confidence.py yourself ===")


if __name__ == "__main__":
    main()

####################################################################################################
