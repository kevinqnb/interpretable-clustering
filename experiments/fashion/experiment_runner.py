#!/usr/bin/env python3
####################################################################################################
# End-to-end runner for the fashion experiment pipeline, at the single default confidence
# threshold defined in config.py. Replaces the old run_confidence_sweep.sh -- there is no
# confidence loop, since there is now only one confidence threshold to fit against.
#
# Dependency graph:
#   mine_rules.py
#      -> [ alphas.py || ids_lambda_search.py ]     (parallel, independent)
#      -> select_alphas.py                           (needs alphas.py's output)
#      -> two families running concurrently, each internally sequential
#         (fashion splits max_rules.py/lambda.py across companion scripts --
#         some algorithms take much longer to fit here, see
#         experiments/README.md's step-3 note):
#           max_rules.py -> max_rules_exkmc.py -> max_rules_combine.py
#           lambda.py    -> lambda_exkmc.py    -> lambda_combine.py
#      -> confidence.py (once, after the families finish -- it does its own internal
#                         0.0-0.95 confidence sweep over the pre-filter pool, and only
#                         needs select_alphas.py's/ids_lambda_search.py's output, not
#                         max_rules.py's/lambda.py's)
#
# Each stage is a standalone module-level script (not an importable function), so stages are
# driven via subprocess rather than direct function calls.

import argparse
import os
import subprocess
import sys
import threading
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


def _run_family(scripts: list[str], cpu_count: int) -> None:
    """
    Runs `scripts` one after another (e.g. max_rules.py -> max_rules_exkmc.py ->
    max_rules_combine.py), all sharing `cpu_count`. The companion _exkmc/_combine
    scripts are already cheap (their own fixed cpu_count of 1) and don't read
    FASHION_CPU_COUNT, so passing it along is harmless overhead, not lost capacity.
    """
    env = dict(os.environ) | {"FASHION_CPU_COUNT": str(cpu_count)}
    for script in scripts:
        print(f"=== launching {script} ({cpu_count} cores) ===")
        ret = _spawn(script, env).wait()
        if ret != 0:
            raise RuntimeError(f"{script} failed with exit code {ret}")


def _run_concurrent_families(families: list[list[str]], total_cpu_count: int) -> None:
    """
    Runs each family in `families` concurrently (one thread per family; each
    thread's subprocesses still run in their own OS process), splitting
    `total_cpu_count` evenly across families.
    """
    per_family = max(1, total_cpu_count // len(families))
    print(f"--- {len(families)} concurrent families: {per_family} cores/family (budget {total_cpu_count}) ---")
    errors = []

    def _worker(seq):
        try:
            _run_family(seq, per_family)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=_worker, args=(seq,)) for seq in families]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[0]


def main():
    parser = argparse.ArgumentParser(
        description="Run the fashion experiment pipeline end-to-end."
    )
    parser.add_argument(
        "--total-cpu-count", type=int, default=None,
        help=(
            "Total CPU budget for the whole run. At each stage below, this is "
            "divided evenly across however many scripts/families run "
            "concurrently at that point in the pipeline so no stage "
            "oversubscribes the machine. Defaults to "
            "experiments/fashion/config.py's TOTAL_CPU_COUNT, which itself "
            "defaults to the machine's detected core count."
        ),
    )
    args = parser.parse_args()
    total_cpu_count = args.total_cpu_count if args.total_cpu_count is not None else TOTAL_CPU_COUNT

    _run_stage(["mine_rules.py"], total_cpu_count)
    _run_stage(["alphas.py", "ids_lambda_search.py"], total_cpu_count)
    _run_stage(["select_alphas.py"], total_cpu_count)
    _run_concurrent_families(
        [
            ["max_rules.py", "max_rules_exkmc.py", "max_rules_combine.py"],
            ["lambda.py", "lambda_exkmc.py", "lambda_combine.py"],
        ],
        total_cpu_count,
    )
    _run_stage(["confidence.py"], total_cpu_count)

    print("=== fashion pipeline complete ===")


if __name__ == "__main__":
    main()

####################################################################################################
