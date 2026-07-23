#!/usr/bin/env python3
####################################################################################################
# End-to-end runner for the climate experiment pipeline, at the single default confidence
# threshold defined in config.py. Replaces the old run_confidence_sweep.sh -- there is no
# confidence loop and no --emit-grid barrier stage, since there is now only one confidence
# threshold to fit against.
#
# Dependency graph:
#   mine_rules.py
#      -> [ alphas.py || ids_lambda_search.py ]              (parallel, independent)
#      -> select_alphas.py                                    (needs alphas.py's output)
#      -> [ max_rules.py || lambda.py || confidence.py || input_sensitivity.py ]
#                                                              (parallel; all four depend only
#                                                                on select_alphas.py's and
#                                                                ids_lambda_search.py's output,
#                                                                not on each other)
#
# Each stage is a standalone module-level script (not an importable function), so stages are
# driven via subprocess rather than direct function calls.

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

from experiments.climate.config import TOTAL_CPU_COUNT

CLIMATE_DIR = PROJECT_ROOT / "experiments" / "climate"


def _spawn(script_name: str, env: dict) -> subprocess.Popen:
    print(f"=== launching {script_name} ===")
    # cwd=PROJECT_ROOT: every stage script does file I/O against repo-root-relative
    # paths (e.g. 'data/experiments/climate/rules/') and never chdir's itself.
    return subprocess.Popen(
        [sys.executable, str(CLIMATE_DIR / script_name)],
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
    (via the CLIMATE_CPU_COUNT env var, which config.py's CPU_COUNT reads) -- so a
    stage with 3 concurrent scripts gives each 1/3 of the budget, rather than
    each independently requesting the full machine the way a single CPU_COUNT
    constant did before. Scripts that don't actually use joblib (mine_rules.py,
    ids_lambda_search.py, select_alphas.py) just ignore the env var, so their
    share of the split is unused overhead, not lost capacity -- see each
    script's own CPU_COUNT usage (or lack of it).
    """
    per_script = max(1, total_cpu_count // len(scripts))
    env = dict(os.environ) | {"CLIMATE_CPU_COUNT": str(per_script)}
    print(f"--- stage {scripts}: {per_script} cores/script (budget {total_cpu_count}) ---")
    _wait([(_spawn(s, env), s) for s in scripts])


def main():
    parser = argparse.ArgumentParser(
        description="Run the climate experiment pipeline end-to-end."
    )
    parser.add_argument(
        "--total-cpu-count", type=int, default=None,
        help=(
            "Total CPU budget for the whole run. At each stage below, this is "
            "divided evenly across however many scripts run concurrently at "
            "that point in the pipeline (e.g. 4-way in the final stage, where "
            "max_rules.py/lambda.py/confidence.py/input_sensitivity.py run at once) so no stage "
            "oversubscribes the machine. Defaults to "
            "experiments/climate/config.py's TOTAL_CPU_COUNT, which itself "
            "defaults to the machine's detected core count."
        ),
    )
    args = parser.parse_args()
    total_cpu_count = args.total_cpu_count if args.total_cpu_count is not None else TOTAL_CPU_COUNT

    _run_stage(["mine_rules.py"], total_cpu_count)
    _run_stage(["alphas.py", "ids_lambda_search.py"], total_cpu_count)
    _run_stage(["select_alphas.py"], total_cpu_count)
    _run_stage(["max_rules.py", "lambda.py", "confidence.py"], total_cpu_count)

    print("=== climate pipeline complete ===")


if __name__ == "__main__":
    main()

####################################################################################################
