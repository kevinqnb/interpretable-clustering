####################################################################################################
import sys
from pathlib import Path

# Ensure the repository root (the folder that contains `data/`) is on sys.path.
# This makes `from data.preprocessing import ...` work when running this file directly.
_HERE = Path(__file__).resolve()
PROJECT_ROOT = next((p for p in _HERE.parents if (p / "data").is_dir()), None)
if PROJECT_ROOT is None:
    raise ModuleNotFoundError(
        "Could not locate repository root."
    )
sys.path.insert(0, str(PROJECT_ROOT))

from data.preprocessing import *
from experiments.experiment import Experiment
from experiments.modules import *
from experiments.fashion.config import (
    SEED, N_CLUSTERS, N_SELECT_DEFAULT, MAX_RULES, SHALLOW_TREE_DEPTH_FACTOR,
    N_FOREST, FOREST_MAX_DEPTH, CAR_MIN_SUPPORT, CAR_MIN_CONFIDENCE,
    CAR_MAX_RULE_LENGTH, CONFIDENCE_DEFAULT, N_TRIALS, TRIAL_SEEDS, CPU_COUNT,
    OUTFILE_REF, RULES_DIR, LAMBDA_DIR,
)

####################################################################################################

import os
import json
import math
import pickle
from collections import defaultdict
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from intercluster import *
from intercluster.decision_trees import *
from intercluster.decision_sets import *
from intercluster.decision_sets.ids import IDSCoverageCache
from intercluster.decision_sets.objectives import *
from intercluster.decision_sets.mining import *
from intercluster.measurements import *


# Prevents memory leakage for KMeans:
os.environ["OMP_NUM_THREADS"] = "1"

experiment_cpu_count = CPU_COUNT

# REMINDER: Initialize the seed only here, not inside any sub-function or
# class (except select baseline experiments like KMeans) -- passing a seed
# there resets it on every call.
# Classes with their own internal randomness (IDS, DecisionTree) accept an
# explicit random_state instead of relying on this global seed -- see
# `trial_seeds` below, which derives one seed per trial so those modules can be
# refit across multiple trials and their results reported as mean/std rather
# than a single, arbitrarily-seeded point estimate.
# Note: ExKMC is fit in its own separate script (`lambda_exkmc.py`; see
# `lambda_combine.py`), mirroring max_rules.py's split (see
# experiments/README.md).
seed = SEED

# Number of independent random-seed trials used to evaluate stochastic modules
# (IDS, Decision-Tree). Deterministic modules (PEC, ExKMC, CN2, CBA, WRA) are fit
# once, since repeating them would just reproduce the same result. `trial_seeds`
# is derived deterministically from `seed` so that re-running this script
# reproduces the exact same set of trials.
n_trials = N_TRIALS
trial_seeds = TRIAL_SEEDS

def _memoryview_safe(x):
    """
    Make array safe to run in a Cython memoryview-based kernel.
    As far as I can tell, this sometimes is an issue when data is pickled in
    multiprocessing environments.
    """
    if not x.flags.writeable:
        if not x.flags.owndata:
            x = x.copy(order='C')
        x.setflags(write=True)
    return x

####################################################################################################
# Read and process data:
data, data_labels, feature_labels, scaler = load_preprocessed_fashion()
data = _memoryview_safe(data)
n,d = data.shape

fixed_parameters = {
    'n': n,
    'd': d,
    'n_clusters': N_CLUSTERS,
    'n_select': N_SELECT_DEFAULT,
    'max_rules': MAX_RULES,
    'shallow_tree_depth_factor': SHALLOW_TREE_DEPTH_FACTOR,
    'n_forest': N_FOREST,
    'forest_max_depth': FOREST_MAX_DEPTH,
    'car_min_support': CAR_MIN_SUPPORT,
    'car_min_confidence': CAR_MIN_CONFIDENCE,
    'car_max_rule_length': CAR_MAX_RULE_LENGTH, # (really means 4 by pyfim convention)
    'filter_confidence': CONFIDENCE_DEFAULT,
    'seed': seed,
    'n_trials': n_trials,
    'trial_seeds': trial_seeds,
}

# Unlike max_rules.py (which sweeps the rule budget n_select over n_rules_list),
# this experiment fixes the rule budget at fixed_parameters['n_select'] -- the same
# budget alphas.py used to tune alpha -- and instead sweeps PEC's lambda hyperparameter.
n_select = fixed_parameters['n_select']

np.random.seed(fixed_parameters['seed'])

kmeans_base = KMeansBase(n_clusters = fixed_parameters['n_clusters'], random_seed = fixed_parameters['seed'])
kmeans_assignment = kmeans_base.assign(data)
kmeans_labels = kmeans_base.labels

# Weights for uncertainty objectives
weights = distance_ratio_score(data, kmeans_base.centers)
fixed_parameters['weights'] = weights.tolist()

# Alpha values for objectives:
# Identical to lambda.py, except alpha is fixed to 0 for every module instead of being loaded
# from select_alphas.py's elbow-selected 'selected_alphas*.json' -- so this doesn't depend on
# select_alphas.py having been run at all. `defaultdict(float)` stands in for the per-module dict
# lambda.py loads from disk: every module_name lookup below (`fixed_parameters['alpha'][...]`)
# resolves to 0.0 regardless of key.
fixed_parameters['alpha'] = defaultdict(float)

decision_info_dict_directory = RULES_DIR

outfile = LAMBDA_DIR
# This output ref differs from lambda.py's so this script's (alpha=0) results don't clobber
# lambda.py's own (elbow-selected alpha) output.
outfile_ref = '_dscluster' + OUTFILE_REF + '_alpha_zero'

####################################################################################################
# Load pre-mined rules:

ensemble_rules = load_rules(RULES_DIR + f'ensemble_rules{OUTFILE_REF}.pkl')

with open(RULES_DIR + f'ensemble_labels{OUTFILE_REF}.pkl', 'rb') as f:
    ensemble_labels = pickle.load(f)

rule_miner_dict = {
    'ensemble': (None, ensemble_rules, None),
}

####################################################################################################
# Comparison Modules:
#
# NOTE on reproducibility: Decision-Tree and IDS both have inherent randomness in
# their fitted solution (sklearn tree tie-breaking and randomized-greedy/SLS
# selection respectively). Rather than fit each once under the single global
# `seed`, these two are refit across `trial_seeds` further below (see
# "Stochastic module trials") and their results are recorded as mean/std/values
# instead of a single point estimate. PEC, ExKMC, WRA, CBA, and CN2 have no
# internal randomness given fixed inputs. ExKMC is handled in the separate
# `lambda_exkmc.py` script (see `lambda_combine.py`).
#
# NOTE on this experiment vs. max_rules.py: none of the comparison models below take
# a lambda parameter, and their selected rules don't change as PEC's lambda changes.
# So -- unlike max_rules.py, which refits each comparison model once per rule budget r
# -- every comparison model here is fit exactly ONCE, at the fixed `n_select` budget,
# and its result is simply broadcast across every lambda value in the sweep.

decision_tree_shared_params = {'max_leaf_nodes': n_select}
decision_tree_mod = DecisionTreeMod(
    model = DecisionTree,
    name = 'Decision-Tree'
)


cba_shared_params = {'n_select': n_select}
cba_mod = DecisionSetMod(
    model=CBA,
    rules=ensemble_rules,
    rule_labels=ensemble_labels,
    name='CBA'
)


cn2_shared_params = {'n_select': n_select}
cn2_mod = DecisionSetMod(
    model=CN2,
    rules=None,
    name='CN2'
)


# IDS:
with open(RULES_DIR + f'ids_lambdas{OUTFILE_REF}.json') as f:
    ids_lambdas = json.load(f)
if isinstance(ids_lambdas, dict):
    ids_lambdas = list(ids_lambdas.values())

_ids_cache_path = RULES_DIR + f'ids_coverage_cache_ensemble{OUTFILE_REF}.pkl'
if os.path.exists(_ids_cache_path):
    print("Loading pre-built IDS cache...")
    with open(_ids_cache_path, 'rb') as f:
        ids_cache = pickle.load(f)
    print(f"IDS cache loaded ({len(ids_cache.decisions)} decisions).")
else:
    print("Pre-computing IDS cache...")
    # Built exactly the way ids_lambda_search.py builds it -- IDSCoverageCache.from_rules over the
    # ensemble rules and their labels -- so this fallback reproduces the cached file rather than a
    # different one. Two things make that worth being careful about: from_rules keys decisions to
    # rule order and keeps one per rule, whereas routing through IDS.fit()/set_labels builds a set
    # (hash order, and duplicate decisions silently collapse), and the IDS optimizer indexes into
    # that ordering. from_rules also runs no optimizer, unlike fit(), which would run a full
    # selection pass over the whole pool purely as a side effect and then discard it.
    ids_cache = IDSCoverageCache.from_rules(
        ensemble_rules, ensemble_labels, data, kmeans_labels
    )
    with open(_ids_cache_path, 'wb') as f:
        pickle.dump(ids_cache, f)
    print(f"IDS cache ready: {len(ids_cache.decisions)} decisions.")

ids_shared_params = {
    'n_select': n_select,
    'lambdas': ids_lambdas,
    'cache': ids_cache,
    'optimizer': 'random_greedy',
}
ids_mod = DecisionSetMod(
    model=IDS,
    rules=ensemble_rules,
    rule_labels=ensemble_labels,
    name='IDS'
)

####################################################################################################
# Objectives for Decision Set Clustering:

objective_dict = {
    'coverage-mistake': {
        'objective_type': 'coverage-mistake',
        'selection_algorithm': 'distorted-greedy',
        'precomputed_path': os.path.join(
            decision_info_dict_directory, f'mistake_info_dict{OUTFILE_REF}.pkl.gz'
        )
    },
    'coverage-cost': {
        'objective_type': 'coverage-cost',
        'cluster_centers': kmeans_base.centers,
        'cluster_cost_method': 'kmeans',
        'selection_algorithm': 'distorted-greedy',
        'precomputed_path': os.path.join(
            decision_info_dict_directory, f'cost_info_dict{OUTFILE_REF}.pkl.gz'
        )
    },
    'coverage-pairwise-distance': {
        'objective_type': 'coverage-pairwise-distance',
        'selection_algorithm': 'distorted-greedy',
        'precomputed_path': os.path.join(
            decision_info_dict_directory, f'pairwise_distance_info_dict{OUTFILE_REF}.pkl.gz'
        )
    },
    # 'coverage-mistake-weighted': {
    #     'objective_type': 'coverage-mistake',
    #     'weights': weights,
    #     'selection_algorithm': 'distorted-greedy',
    #     'precomputed_path': os.path.join(
    #         decision_info_dict_directory, f'mistake_info_dict{OUTFILE_REF}.pkl.gz'
    #     )
    # },
    # 'coverage-cost-weighted': {
    #     'objective_type': 'coverage-cost',
    #     'cluster_centers': kmeans_base.centers,
    #     'weights': weights,
    #     'cluster_cost_method': 'kmeans',
    #     'selection_algorithm': 'distorted-greedy',
    #     'precomputed_path': os.path.join(
    #         decision_info_dict_directory, f'cost_info_dict{OUTFILE_REF}.pkl.gz'
    #     )
    # },
    # 'coverage-pairwise-distance-weighted': {
    #     'objective_type': 'coverage-pairwise-distance',
    #     'weights': weights,
    #     'selection_algorithm': 'distorted-greedy',
    #     'precomputed_path': os.path.join(
    #         decision_info_dict_directory, f'pairwise_distance_info_dict{OUTFILE_REF}.pkl.gz'
    #     )
    # },
}

####################################################################################################
# Lambda grids:
#
# For each objective, PEC's automatic lambda selection (lambda_val=None) gives the
# minimum lambda* for which the distorted-greedy approximation guarantee holds (see
# `Objective.compute_lambdas`/`set_lambda` in
# intercluster/decision_sets/objectives/objectives.py). We probe this once per
# objective with a throwaway PEC fit (cheap: it reuses the precomputed coverage/cost
# caches, exactly like the many alpha_val fits in alphas.py), then sweep lambda over
# [0, 2 * lambda*], with lambda* itself guaranteed to be an exact grid point (built as
# two linspaces meeting there) since distorted-greedy only starts being valid at
# that point.

n_lambda_points = 100  # matches alphas.py's n_compare convention
half = n_lambda_points // 2 + 1

lambda_star_dict = {}
lambda_grid_dict = {}

for obj_name, obj_params in objective_dict.items():
    for rule_miner_name, (rule_miner, rules, rule_labels) in rule_miner_dict.items():
        module_name = f'dscluster; {obj_name}; {rule_miner_name}'
        alpha_val = fixed_parameters['alpha'][module_name]
        base_params = {'n_select': n_select, 'alpha_val': alpha_val} | obj_params

        # NOTE: obj_params already carries a 'selection_algorithm' key here
        # (mnist/fashion's objective_dict sets 'distorted-greedy' explicitly, unlike
        # the other datasets). Merging the override last (rather than spreading both
        # as separate kwargs) avoids a duplicate-keyword collision and guarantees
        # this probe always uses 'distorted-greedy' regardless of what obj_params
        # contains.
        probe_params = base_params | {'lambda_val': None, 'selection_algorithm': 'distorted-greedy'}
        probe = PEC(rules = rules, **probe_params)
        # compute_lambda_star does everything fit() would, minus the selection pass -- whose result
        # this probe discarded anyway. Same lambda*, one less full PEC fit per objective.
        lambda_star = probe.compute_lambda_star(data, kmeans_labels)

        lower = np.linspace(0.0, lambda_star, half)
        upper = np.linspace(lambda_star, 2 * lambda_star, half)
        lambda_grid = np.concatenate([lower, upper[1:]])

        lambda_star_dict[module_name] = float(lambda_star)
        lambda_grid_dict[module_name] = lambda_grid.tolist()

fixed_parameters['lambda_star'] = lambda_star_dict
fixed_parameters['lambda_grid'] = lambda_grid_dict
fixed_parameters['n_lambda_points'] = n_lambda_points

# Union of every objective's lambda grid -- used to broadcast each comparison
# model's single fit (which doesn't depend on lambda or the objective at all)
# across every lambda value any objective's plot might need to look up.
all_lambda_values = tuple(
    sorted(set().union(*(set(g) for g in lambda_grid_dict.values())))
)

####################################################################################################
# Decision Set Clustering Modules:
#
# Two modules per objective: 'lazy-greedy' is valid (and recorded) across the full
# [0, 2 * lambda*] grid, while 'distorted-greedy' is only valid -- and thus only
# fit/recorded -- for lambda >= lambda*.

dscluster_module_list = []
for obj_name, obj_params in objective_dict.items():
    for rule_miner_name, (rule_miner, rules, rule_labels) in rule_miner_dict.items():
        module_name = f'dscluster; {obj_name}; {rule_miner_name}'
        alpha_val = fixed_parameters['alpha'][module_name]
        base_params = {'n_select': n_select, 'alpha_val': alpha_val} | obj_params

        lambda_star = lambda_star_dict[module_name]
        lambda_grid = lambda_grid_dict[module_name]

        lazy_params = {
            (l,): base_params | {'lambda_val': l, 'selection_algorithm': 'lazy-greedy'}
            for l in lambda_grid
        }
        lazy_mod = DecisionSetMod(
            model = PEC,
            rules = rules,
            name = f'{module_name}; lazy-greedy'
        )
        dscluster_module_list.append((lazy_mod, lazy_params))

        distorted_params = {
            (l,): base_params | {'lambda_val': l, 'selection_algorithm': 'distorted-greedy'}
            for l in lambda_grid if l >= lambda_star
        }
        distorted_mod = DecisionSetMod(
            model = PEC,
            rules = rules,
            name = f'{module_name}; distorted-greedy'
        )
        dscluster_module_list.append((distorted_mod, distorted_params))


####################################################################################################


baseline = kmeans_base
# Decision-Tree and IDS are handled separately below via `fit_stochastic_shared`
# (see "Stochastic module trials"), since they need to be refit per trial seed
# rather than dispatched once through `Experiment`'s joblib-parallel `run()`
# (whose worker processes do not inherit this script's seeded global NumPy
# state, which would make single-fit results irreproducible for exactly these
# randomized modules). Each is fit once per trial seed here (not once per
# lambda value) and its per-trial result is broadcast across
# `all_lambda_values`, exactly like the deterministic comparison modules below.
# ExKMC is handled in the separate `lambda_exkmc.py` script.
module_list = [
    (cba_mod, {all_lambda_values: cba_shared_params}),
    #(cn2_mod, {all_lambda_values: cn2_shared_params}),
] + dscluster_module_list

measurement_fns = [
    TotalCoverage(),
    TotalCoverage(weights = weights, name = 'total-coverage-weighted'),
    TotalCoverageSet(),
    ClusterCoverage(baseline_assignment = kmeans_assignment),
    ClusterCoverage(
        baseline_assignment = kmeans_assignment,
        weights = weights,
        name = 'cluster-coverage-weighted'
    ),
    ClusterCoverageSet(baseline_assignment = kmeans_assignment),
    Overlap(),
    Mistakes(baseline_assignment = kmeans_assignment),
    ClusteringCost(data = data, average = True, normalize = True, method = "kmeans"),
    RuleClusteringCost(data = data, cluster_centers = kmeans_base.centers, method = "kmeans"),
    #PairwiseDistance(baseline_assignment = kmeans_assignment),
    RulePairwiseDistance(baseline_assignment = kmeans_assignment),
]

exp = Experiment(
    data = data,
    baseline = kmeans_base,
    module_list = module_list,
    measurement_fns= measurement_fns,
    fixed_parameters = fixed_parameters,
    cpu_count = experiment_cpu_count,
    verbose = True
)

import time
start = time.time()
exp_results = exp.run()

####################################################################################################
# Stochastic module trials
#
# Decision-Tree and IDS each have a fitted solution that depends on randomness.
# Rather than record one arbitrarily-seeded fit, each is refit once per seed in
# `trial_seeds` and the results across trials are aggregated into
# {'mean', 'std', 'values'} via `aggregate_trials` (see experiments/modules.py).
# This runs single-process (not through `Experiment`'s joblib dispatch) specifically
# so each trial's explicit seed is what controls its randomness.
#
# Neither varies with lambda (only PEC does), so both are handled with
# `fit_stochastic_shared`: fit once per trial seed, and the trial-aggregated
# result is broadcast across every value in `all_lambda_values`.

def _seed_and_fit(mod, params, trial_seed):
    """
    Fits `mod` for one trial. Sets the trial's explicit seed both as a fitting
    parameter (for classes that thread it through properly, e.g. IDS,
    DecisionTree, ShallowTree) AND as the global NumPy seed immediately before
    fit() -- some dependencies still read the global RNG state directly and
    aren't fully parameterized by a passed-in random_state alone. This call
    runs single-process, so setting the global seed here is safe and
    sufficient for reproducibility (unlike doing so at the top of the script,
    which does not survive joblib worker dispatch).
    """
    np.random.seed(trial_seed)
    mod.update_fitting_params(params)
    return mod.fit(data, kmeans_labels)


def _module_trial_result(mod, assignments, measurement_fns):
    data_to_rule, rule_to_cluster, data_to_cluster = assignments
    return {
        'lambda': mod.lambda_val if hasattr(mod, 'lambda_val') else None,
        'lambda_n_rules': getattr(mod, 'n_available_decisions', np.nan),
        'max-rule-length': mod.max_rule_length,
        'sum-rule-length': mod.sum_rule_length,
        'weighted-avg-length': mod.weighted_average_rule_length,
    } | {
        fn.name: fn(data_to_rule, rule_to_cluster, data_to_cluster)
        for fn in measurement_fns
    }


def fit_stochastic_shared(mod, shared_params, r_values, trial_seeds, measurement_fns, seed_key='random_state'):
    """
    Refits `mod` once per trial seed -- for modules whose output does not vary
    with the swept lambda value -- and broadcasts the trial-aggregated result
    across every value in `r_values` (matching the pre-existing convention for
    these modules in max_rules.py).
    """
    result = (
        {'lambda': {}, 'lambda_n_rules': {}, 'max-rule-length': {},
         'sum-rule-length': {}, 'weighted-avg-length': {}} |
        {fn.name: {} for fn in measurement_fns}
    )
    trial_dicts = []
    for trial_seed in trial_seeds:
        assignments = _seed_and_fit(mod, dict(shared_params) | {seed_key: trial_seed}, trial_seed)
        trial_dicts.append(_module_trial_result(mod, assignments, measurement_fns))
    aggregated = aggregate_trials(trial_dicts)
    for r in r_values:
        for key, agg_val in aggregated.items():
            result[key][r] = agg_val
    return result


print(f"Fitting stochastic modules across {n_trials} trials each...")
exp_results['modules']['Decision-Tree'] = fit_stochastic_shared(
    decision_tree_mod, decision_tree_shared_params, all_lambda_values, trial_seeds, measurement_fns,
    seed_key='random_state'
)
exp_results['modules']['IDS'] = fit_stochastic_shared(
    ids_mod, ids_shared_params, all_lambda_values, trial_seeds, measurement_fns,
    seed_key='random_state'
)
print("Stochastic modules done.")

exp.save_results(outfile, outfile_ref)
end = time.time()
print("Experiment time:", end - start)


####################################################################################################
