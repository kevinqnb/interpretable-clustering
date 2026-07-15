"""
Shared harness for timing 'distorted-greedy' vs. 'heap-distorted-greedy' PEC fits across a
growing, randomly-sampled subset of a mined rule ensemble. Used by each dataset's `timer.py`.

Growing-subset procedure (per trial): draw one uniform random permutation of the full rule
ensemble, then fit PEC on successive prefixes of size n_step, 2*n_step, 3*n_step, ..., stopping
once the prefix reaches the full ensemble (the last step may be shorter than n_step). This is
equivalent to "start with a random size-n_step subset, then repeatedly append a random
min(n_step, n_remaining)-sized subset of whatever rules haven't been included yet, until none
remain" -- a prefix of a uniform random permutation is a uniform random subset, and each
successive slice of that same permutation is a uniform random subset of what's left -- but it
avoids re-sampling without replacement from a shrinking pool at every step.

Timing is measured around PEC(...).fit(X, y) (set_labels + select + trim), i.e. exactly what a
real caller pays to go from a rule pool to a fitted decision set. To isolate the two selection
algorithms rather than compare-plus-recompute-lambda* noise, callers should pass a fixed
`lambda_val` in `pec_kwargs` (probed once from the full ensemble, e.g. via
`PEC.compute_lambda_star`) rather than leaving it as None -- see each dataset's `timer.py`.
"""

import time
import numpy as np
from typing import Any, Dict, List, Set
from numpy.typing import NDArray

from intercluster import Rule
from intercluster.decision_sets import PEC

from experiments.modules import aggregate_trials

####################################################################################################

ALGORITHMS = ['distorted-greedy', 'heap-distorted-greedy']


def _growing_pool_sizes(n_total: int, n_step: int) -> List[int]:
    """Ascending prefix sizes n_step, 2*n_step, ..., ending with n_total exactly once."""
    sizes = list(range(n_step, n_total, n_step))
    if not sizes or sizes[-1] != n_total:
        sizes.append(n_total)
    return sizes


def time_pec_fit(
    rules: List[Rule],
    X: NDArray,
    y: List[Set[int]],
    selection_algorithm: str,
    pec_kwargs: Dict[str, Any],
) -> float:
    """Times a single `PEC(rules=rules, selection_algorithm=selection_algorithm, ...).fit(X, y)`
    call. Returns elapsed wall-clock seconds; the fitted PEC instance itself is discarded."""
    pec = PEC(rules=rules, selection_algorithm=selection_algorithm, **pec_kwargs)
    start = time.perf_counter()
    pec.fit(X, y)
    return time.perf_counter() - start


def run_timing_sweep(
    rules: List[Rule],
    X: NDArray,
    y: List[Set[int]],
    pec_kwargs: Dict[str, Any],
    n_step: int = 200,
    n_trials: int = 10,
    seed: int = 342,
) -> Dict[str, Any]:
    """
    Bootstraps a timing comparison between 'distorted-greedy' and 'heap-distorted-greedy'
    PEC fits over a growing random subset of `rules` (see module docstring for the sampling
    procedure).

    Args:
        rules (List[Rule]): Full mined rule ensemble to sample from.
        X (NDArray): Input dataset.
        y (List[Set[int]]): Target labels (e.g. baseline cluster assignment).
        pec_kwargs (Dict[str, Any]): Keyword arguments forwarded to every `PEC(...)`
            construction. Must NOT include `rules` or `selection_algorithm` -- this function
            supplies both. Should include a fixed `lambda_val` (see module docstring).
        n_step (int, optional): Number of rules added to the pool at each step. Defaults to 200.
        n_trials (int, optional): Number of independent random orderings of `rules` to average
            over. Defaults to 10.
        seed (int, optional): Base seed; trial i draws its permutation from
            `np.random.default_rng(seed + i)`. Defaults to 342.

    Returns:
        results (Dict[str, Any]): {
            'n_step': int, 'n_trials': int, 'seed': int, 'n_total_rules': int,
            'pool_sizes': List[int],
            'distorted-greedy': List[Dict[str, Any]],       # one {'mean','std','values'} per pool size
            'heap-distorted-greedy': List[Dict[str, Any]],  # one {'mean','std','values'} per pool size
        }
    """
    n_total = len(rules)
    pool_sizes = _growing_pool_sizes(n_total, n_step)

    # trial_times[alg][step_idx] collects one elapsed time per trial.
    trial_times = {alg: [[] for _ in pool_sizes] for alg in ALGORITHMS}

    for trial in range(n_trials):
        rng = np.random.default_rng(seed + trial)
        order = rng.permutation(n_total)
        for step_idx, size in enumerate(pool_sizes):
            subset = [rules[i] for i in order[:size]]
            for alg in ALGORITHMS:
                elapsed = time_pec_fit(subset, X, y, alg, pec_kwargs)
                trial_times[alg][step_idx].append(elapsed)

    results: Dict[str, Any] = {
        'n_step': n_step,
        'n_trials': n_trials,
        'seed': seed,
        'n_total_rules': n_total,
        'pool_sizes': pool_sizes,
    }
    for alg in ALGORITHMS:
        # aggregate_trials expects a list of per-trial dicts sharing the same keys; here each
        # "trial" dict has one key per pool size, so this aggregates across trials independently
        # for each pool size in a single call.
        per_trial_dicts = [
            {size: trial_times[alg][step_idx][trial] for step_idx, size in enumerate(pool_sizes)}
            for trial in range(n_trials)
        ]
        aggregated = aggregate_trials(per_trial_dicts)
        results[alg] = [aggregated[size] for size in pool_sizes]

    return results
