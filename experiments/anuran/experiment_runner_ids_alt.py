#!/usr/bin/env python3
####################################################################################################
# End-to-end runner for the anuran pipeline's IDS-alt variant: ids_lambda_search_alt.py, then
# max_rules_ids_alt.py. Both are single-script stages that reuse rule mining / alphas outputs
# from a prior experiment_runner.py run rather than regenerating them.
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

from experiments.anuran.config import TOTAL_CPU_COUNT

ANURAN_DIR = PROJECT_ROOT / "experiments" / "anuran"


def _spawn(script_name: str, env: dict) -> subprocess.Popen:
    print(f"=== launching {script_name} ===")
    # cwd=PROJECT_ROOT: every stage script does file I/O against repo-root-relative
    # paths (e.g. 'data/experiments/anuran/rules/') and never chdir's itself.
    return subprocess.Popen(
        [sys.executable, str(ANURAN_DIR / script_name)],
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
    (via the ANURAN_CPU_COUNT env var, which config.py's CPU_COUNT reads). Both
    stages here run a single script, so each simply gets the full budget.
    """
    per_script = max(1, total_cpu_count // len(scripts))
    env = dict(os.environ) | {"ANURAN_CPU_COUNT": str(per_script)}
    print(f"--- stage {scripts}: {per_script} cores/script (budget {total_cpu_count}) ---")
    _wait([(_spawn(s, env), s) for s in scripts])


def main():
    parser = argparse.ArgumentParser(
        description="Run the anuran pipeline's IDS-alt variant end-to-end."
    )
    parser.add_argument(
        "--total-cpu-count", type=int, default=None,
        help=(
            "Total CPU budget for the whole run. Each stage here is a single script, so it "
            "gets the full budget rather than sharing it with concurrent stage-mates. Defaults "
            "to experiments/anuran/config.py's TOTAL_CPU_COUNT, which itself defaults to the "
            "machine's detected core count."
        ),
    )
    args = parser.parse_args()
    total_cpu_count = args.total_cpu_count if args.total_cpu_count is not None else TOTAL_CPU_COUNT

    _run_stage(["ids_lambda_search_alt.py"], total_cpu_count)
    _run_stage(["max_rules_ids_alt.py"], total_cpu_count)

    print("=== anuran pipeline complete ===")


if __name__ == "__main__":
    main()

####################################################################################################
