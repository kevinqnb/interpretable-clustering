#!/usr/bin/env python3
####################################################################################################
# Runs, concurrently, every max_rules/lambda per-model script whose fitting work is fundamentally
# single-process -- CN2/Decision-Tree/IDS refit sequentially outside `Experiment.run()`'s joblib
# dispatch (see those scripts' own comments: reproducibility requires each trial to reseed
# single-process), and lambda.py's CBA/CN2 are each a single fit broadcast across the whole
# lambda grid (one joblib task). All seven get zero benefit from more than 1 core each, so instead
# of giving one of them a big core allocation, this launches all seven at once (up to
# `--total-cpu-count` at a time), each pinned to 1 core via the *_CPU_COUNT env var --
# e.g. on an 8-core machine, all 7 start immediately and one core sits idle; on a smaller machine,
# the rest queue and start as slots free up.
#
# Deliberately NOT included: max_rules_cba.py (7 useful cores), max_rules_pec.py (24),
# lambda_pec.py (51), and the exkmc scripts (hardcode 1 core themselves already, for unrelated
# reasons) -- those scale with more cores per job and are better run as their own
# larger-allocation jobs, not bundled into a fixed 1-core-per-job runner like this one.
#
# Prerequisite: mine_rules.py -> [alphas.py || ids_lambda_search.py] -> select_alphas.py must
# already have produced this OUTFILE_REF's selected_alphas/ensemble_rules/ids_lambdas files (see
# experiment_runner.py's dependency graph) -- this script checks for them up front and fails fast
# with one clear message, rather than launching seven processes that each crash on the same
# missing file.

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_HERE = Path(__file__).resolve()
PROJECT_ROOT = next((p for p in _HERE.parents if (p / "data").is_dir()), None)
if PROJECT_ROOT is None:
    raise ModuleNotFoundError("Could not locate repository root.")
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.fashion.config import TOTAL_CPU_COUNT, OUTFILE_REF, RULES_DIR, ALPHAS_DIR

FASHION_DIR = PROJECT_ROOT / "experiments" / "fashion"

SCRIPTS = [
    "max_rules_cn2.py",
    "max_rules_decision_tree.py",
    "max_rules_ids.py",
    "lambda_cba.py",
    "lambda_cn2.py",
    "lambda_decision_tree.py",
    "lambda_ids.py",
]

# Not exhaustive per-script (e.g. max_rules_cn2.py/max_rules_decision_tree.py don't actually
# touch ensemble_rules), but every file here is needed by at least one script in SCRIPTS, and
# checking the union up front means one clear error instead of a different crash per script.
REQUIRED_FILES = [
    RULES_DIR + f"ensemble_rules{OUTFILE_REF}.pkl",
    RULES_DIR + f"ensemble_labels{OUTFILE_REF}.pkl",
    RULES_DIR + f"ids_lambdas{OUTFILE_REF}.json",
    ALPHAS_DIR + f"selected_alphas{OUTFILE_REF}.json",
]


def _check_prereqs():
    missing = [f for f in REQUIRED_FILES if not (PROJECT_ROOT / f).exists()]
    if missing:
        raise FileNotFoundError(
            "Missing upstream artifacts needed by this runner's scripts -- run "
            "mine_rules.py -> [alphas.py || ids_lambda_search.py] -> select_alphas.py first. "
            f"Missing: {missing}"
        )

    ids_cache = PROJECT_ROOT / (RULES_DIR + f"ids_coverage_cache_ensemble{OUTFILE_REF}.pkl")
    if not ids_cache.exists():
        print(
            f"NOTE: {ids_cache} doesn't exist yet. max_rules_ids.py and lambda_ids.py will each "
            "try to build it if it's still missing when they start, and this runner starts both "
            "at once -- so they could race to build and write it simultaneously. Run "
            "ids_lambda_search.py first (the canonical builder) to avoid that."
        )


def _run_script(script_name, env):
    print(f"=== launching {script_name} (1 core) ===")
    ret = subprocess.Popen(
        [sys.executable, str(FASHION_DIR / script_name)],
        cwd=str(PROJECT_ROOT),
        env=env,
    ).wait()
    status = "OK" if ret == 0 else f"FAILED (exit {ret})"
    print(f"=== {script_name}: {status} ===")
    return script_name, ret


def main():
    parser = argparse.ArgumentParser(
        description="Run every 1-core-useful max_rules/lambda per-model script concurrently."
    )
    parser.add_argument(
        "--total-cpu-count", type=int, default=None,
        help=(
            "Max scripts to run at once (each pinned to exactly 1 core). Defaults to "
            "experiments/fashion/config.py's TOTAL_CPU_COUNT, which itself defaults to the "
            "machine's detected core count."
        ),
    )
    args = parser.parse_args()
    total_cpu_count = args.total_cpu_count if args.total_cpu_count is not None else TOTAL_CPU_COUNT

    _check_prereqs()

    env = dict(os.environ) | {"FASHION_CPU_COUNT": "1"}
    max_workers = max(1, min(total_cpu_count, len(SCRIPTS)))
    print(f"Running {len(SCRIPTS)} single-core scripts, {max_workers} at a time...")

    failures = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_run_script, script, env) for script in SCRIPTS]
        for future in as_completed(futures):
            script_name, ret = future.result()
            if ret != 0:
                failures.append((script_name, ret))

    if failures:
        raise RuntimeError(f"Script(s) failed: {failures}")

    print("=== all single-core scripts complete ===")


if __name__ == "__main__":
    main()

####################################################################################################
