####################################################################################################
# Path setup

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

####################################################################################################

import os
import json
import math
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from intercluster import *
from intercluster.decision_trees import *
from intercluster.decision_sets import *
from intercluster.decision_sets.objectives import *
from intercluster.decision_sets.mining import *
from intercluster.measurements import *


# Prevents memory leakage for KMeans:
os.environ["OMP_NUM_THREADS"] = "1"

experiment_cpu_count = 4

# REMINDER: The seed should only be initialized here. It should NOT
# within the parameters of any sub-function or class (except for select
# baseline experiments like KMeans), since these will
# reset the seed each time they are given one.
# Classes with their own internal randomness (IDS, ExplanationTree, DecisionTree,
# ShallowTree) accept an explicit random_state / kmeans_random_state instead of
# relying on this global seed -- see `trial_seeds` below, which derives one seed
# per trial so those modules can be refit across multiple trials and their results
# reported as mean/std rather than a single, arbitrarily-seeded point estimate.
seed = 342

# Number of independent random-seed trials used to evaluate stochastic modules
# (IDS, Exp-Tree, Decision-Tree, Shallow-Tree). Deterministic modules (PEC, ExKMC,
# CN2, CBA, WRA) are fit once, since repeating them would just reproduce the same
# result. `trial_seeds` is derived deterministically from `seed` so that re-running
# this script reproduces the exact same set of trials.
n_trials = 10
trial_seeds = [seed + i for i in range(n_trials)]

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
data, data_labels, feature_labels, scaler = load_preprocessed_climate('data/climate')
data = _memoryview_safe(data)
n,d = data.shape

fixed_parameters = {
    'n': n,
    'd': d,
    'n_clusters': 6,
    'n_select': 6,
    'max_rules': 12,
    'shallow_tree_depth_factor': 0.03,
    'n_forest': 100,
    'forest_max_depth': 6,
    'car_min_support': 0.025,
    'car_min_confidence': 0.85,
    'car_max_rule_length': 3, # (really means 6 by pyfim convention)
    'filter_confidence': 0.85,
    'seed': seed,
    'n_trials': n_trials,
    'trial_seeds': trial_seeds,
}

n_rules_list = list(range(fixed_parameters['n_clusters'], fixed_parameters['max_rules'] + 1))

np.random.seed(fixed_parameters['seed'])

# Baseline KMeans
kmeans_base = KMeansBase(n_clusters = fixed_parameters['n_clusters'], random_seed = fixed_parameters['seed'])
kmeans_assignment = kmeans_base.assign(data)
kmeans_labels = kmeans_base.labels

# Weights for uncertainty objectives
weights = distance_ratio_score(data, kmeans_base.centers)
fixed_parameters['weights'] = weights.tolist()

# Alpha values for objectives:
with open("data/experiments/climate/alphas/selected_alphas_rule_length.json") as f:
    selected_alpha_dict = json.load(f)
fixed_parameters['alpha'] = selected_alpha_dict

decision_info_dict_directory = 'data/experiments/climate/rules/'

outfile = 'data/experiments/climate/max_rules/'
outfile_ref = '_rule_length'

####################################################################################################
# Load pre-mined rules:


ensemble_rules = load_rules('data/experiments/climate/rules/ensemble_rules.pkl')

with open('data/experiments/climate/rules/ensemble_labels.pkl', 'rb') as f:
    ensemble_labels = pickle.load(f)

rule_miner_dict = {
    'ensemble': (None, ensemble_rules, None),
}

####################################################################################################
# Comparison Modules:
#
# NOTE on reproducibility: Decision-Tree, Exp-Tree, Shallow-Tree, and IDS all have
# inherent randomness in their fitted solution (sklearn tree tie-breaking, heap
# tie-breaking, internal KMeans re-initialization, and randomized-greedy/SLS
# selection respectively). Rather than fit each once under the single global
# `seed`, these four are refit across `trial_seeds` further below (see
# "Stochastic module trials") and their results are recorded as mean/std/values
# instead of a single point estimate. Their *_params dicts below therefore omit
# any seed -- the per-trial seed is injected at fit time. PEC, ExKMC, WRA, CBA,
# and CN2 have no internal randomness given fixed inputs, so they keep the
# original single-fit-per-parameter-value treatment via `Experiment`.

# Decision Tree
decision_tree_params_by_r = {i : {'max_leaf_nodes' : i} for i in n_rules_list}
decision_tree_mod = DecisionTreeMod(
    model = DecisionTree,
    name = 'Decision-Tree'
)


# Explanation Tree
# (ExplanationTree's leaf count is fixed at num_clusters, independent of the rule
# budget r, so -- as before -- a single fit's result is recorded under every r label.)
exp_tree_shared_params = {'num_clusters' : fixed_parameters['n_clusters']}
exp_tree_mod = DecisionTreeMod(
    model = ExplanationTree,
    name = 'Exp-Tree'
)


# ExKMC
exkmc_params = {
    (i,) : {
        'k' : fixed_parameters['n_clusters'],
        'kmeans': kmeans_base.clustering,
        'max_leaf_nodes': i
    } for i in n_rules_list
}
exkmc_mod = DecisionTreeMod(
    model = ExkmcTree,
    name = 'ExKMC'
)


# Shallow Tree
# (ShallowTree's structure is controlled by depth_factor, not by a rule-count/
# max_leaf_nodes parameter, so -- as before -- a single fit's result is recorded
# under every r label.)
shallow_tree_shared_params = {
    'n_clusters' : fixed_parameters['n_clusters'],
    'depth_factor' : fixed_parameters['shallow_tree_depth_factor'],
}
shallow_tree_mod = DecisionTreeMod(
    model = ShallowTree,
    name = 'Shallow-Tree'
)

# WRA:
wra_params = {(r,): {'n_select': r} for r in n_rules_list}
wra_mod = DecisionSetMod(
    model=WRABaseline,
    rules=ensemble_rules,
    rule_labels=ensemble_labels,
    name='WRA'
)

wra_weighted_params = {(r,): {'n_select': r, 'weights': weights} for r in n_rules_list}
wra_weighted_mod = DecisionSetMod(
    model=WRABaseline,
    rules=ensemble_rules,
    rule_labels=ensemble_labels,
    name='WRA-weighted'
)


# CBA:
cba_params = {(r,): {'n_select': r} for r in n_rules_list}
cba_mod = DecisionSetMod(
    model=CBA,
    rules=ensemble_rules,
    rule_labels=ensemble_labels,
    name='CBA'
)


# CN2:
cn2_params = {(r,): {'n_select': r} for r in n_rules_list}
cn2_mod = DecisionSetMod(
    model=CN2,
    rules=None,
    name='CN2'
)


# IDS:
with open('data/experiments/climate/rules/ids_lambdas.json') as f:
    ids_lambdas = json.load(f)
if isinstance(ids_lambdas, dict):
    ids_lambdas = list(ids_lambdas.values())

_ids_cache_path = 'data/experiments/climate/rules/ids_coverage_cache.pkl'
if os.path.exists(_ids_cache_path):
    print("Loading pre-built IDS cache...")
    with open(_ids_cache_path, 'rb') as f:
        ids_cache = pickle.load(f)
    print(f"IDS cache loaded ({len(ids_cache.decisions)} decisions).")
else:
    print("Pre-computing IDS cache...")
    _ids_pre = IDS(rules=ensemble_rules, n_select=None, lambdas=ids_lambdas, random_state=seed)
    _ids_pre.fit(data, kmeans_labels)
    ids_cache = _ids_pre.get_cache()
    print("IDS cache ready.")

ids_params_by_r = {
    r: {
        'n_select': r,
        'lambdas': ids_lambdas,
        'cache': ids_cache,
        'optimizer': 'random_greedy',
    } for r in n_rules_list
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
        'precomputed_path': os.path.join(
            decision_info_dict_directory, 'mistake_info_dict.pkl.gz'
        )
    },
    'coverage-cost': {
        'objective_type': 'coverage-cost',
        'cluster_centers': kmeans_base.centers,
        'cluster_cost_method': 'kmeans',
        'precomputed_path': os.path.join(
            decision_info_dict_directory, 'cost_info_dict.pkl.gz'
        )
    },
    'coverage-pairwise-distance': {
        'objective_type': 'coverage-pairwise-distance',
        'precomputed_path': os.path.join(
            decision_info_dict_directory, 'pairwise_distance_info_dict.pkl.gz'
        )
    },
    'coverage-mistake-weighted': {
        'objective_type': 'coverage-mistake',
        'weights': weights,
        'precomputed_path': os.path.join(
            decision_info_dict_directory, 'mistake_info_dict.pkl.gz'
        )
    },
    'coverage-cost-weighted': {
        'objective_type': 'coverage-cost',
        'cluster_centers': kmeans_base.centers,
        'weights': weights,
        'cluster_cost_method': 'kmeans',
        'precomputed_path': os.path.join(
            decision_info_dict_directory, 'cost_info_dict.pkl.gz'
        )
    },
    'coverage-pairwise-distance-weighted': {
        'objective_type': 'coverage-pairwise-distance',
        'weights': weights,
        'precomputed_path': os.path.join(
            decision_info_dict_directory, 'pairwise_distance_info_dict.pkl.gz'
        )
    },
}

####################################################################################################
# Decision Set Clustering Modules:

dscluster_module_list = []
for obj_name, obj_params in objective_dict.items():
    for rule_miner_name, (rule_miner, rules, rule_labels) in rule_miner_dict.items():
        module_name = f'dscluster; {obj_name}; {rule_miner_name}'
        alpha_val = fixed_parameters['alpha'][module_name]
        dsclust_params = {
            (r,) : {'n_select' : r, 'alpha_val' : alpha_val} | obj_params
            for i,r in enumerate(n_rules_list)
        }
        dsclust_mod = DecisionSetMod(
            model = PEC,
            rules = rules,
            name = module_name
        )
        dscluster_module_list.append((dsclust_mod, dsclust_params))


####################################################################################################


baseline = kmeans_base
# Decision-Tree, Exp-Tree, Shallow-Tree, and IDS are handled separately below via
# `fit_stochastic_varying`/`fit_stochastic_shared` (see "Stochastic module trials"),
# since they need to be refit per trial seed rather than dispatched once through
# `Experiment`'s joblib-parallel `run()` (whose worker processes do not inherit this
# script's seeded global NumPy state, which would make single-fit results
# irreproducible for exactly these randomized modules).
module_list = [
    (exkmc_mod, exkmc_params),
    (wra_mod, wra_params),
    (wra_weighted_mod, wra_weighted_params),
    (cba_mod, cba_params),
    (cn2_mod, cn2_params),
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
# Decision-Tree, Exp-Tree, Shallow-Tree, and IDS each have a fitted solution that
# depends on randomness. Rather than record one arbitrarily-seeded fit, each is
# refit once per seed in `trial_seeds` and the results across trials are aggregated
# into {'mean', 'std', 'values'} via `aggregate_trials` (see experiments/modules.py).
# This runs single-process (not through `Experiment`'s joblib dispatch) specifically
# so each trial's explicit seed is what controls its randomness.

def _seed_and_fit(mod, params, trial_seed):
    """
    Fits `mod` for one trial. Sets the trial's explicit seed both as a fitting
    parameter (for classes that thread it through properly, e.g. IDS,
    DecisionTree, ShallowTree) AND as the global NumPy seed immediately before
    fit() -- some dependencies (e.g. ExplanationTree's compiled Cython splitter,
    see explanation_tree.py's `random_state` docstring caveat) still read the
    global RNG state directly and aren't fully parameterized by a passed-in
    random_state alone. This call runs single-process, so setting the global
    seed here is safe and sufficient for reproducibility (unlike doing so at
    the top of the script, which does not survive joblib worker dispatch).
    """
    np.random.seed(trial_seed)
    mod.update_fitting_params(params)
    return mod.fit(data, kmeans_labels)


def _module_trial_result(mod, assignments, measurement_fns):
    data_to_rule, rule_to_cluster, data_to_cluster = assignments
    return {
        'lambda': mod.lambda_val if hasattr(mod, 'lambda_val') else None,
        'max-rule-length': mod.max_rule_length,
        'sum-rule-length': mod.sum_rule_length,
        'weighted-avg-length': mod.weighted_average_rule_length,
    } | {
        fn.name: fn(data_to_rule, rule_to_cluster, data_to_cluster)
        for fn in measurement_fns
    }


def fit_stochastic_varying(mod, params_by_r, trial_seeds, measurement_fns, seed_key='random_state'):
    """
    Refits `mod` once per (rule-count r, trial seed) pair -- for modules whose
    output genuinely varies with the rule-count budget r -- and aggregates
    results across trials for each r.
    """
    result = (
        {'lambda': {}, 'max-rule-length': {}, 'sum-rule-length': {}, 'weighted-avg-length': {}} |
        {fn.name: {} for fn in measurement_fns}
    )
    for r, base_params in params_by_r.items():
        trial_dicts = []
        for trial_seed in trial_seeds:
            assignments = _seed_and_fit(mod, dict(base_params) | {seed_key: trial_seed}, trial_seed)
            trial_dicts.append(_module_trial_result(mod, assignments, measurement_fns))
        for key, agg_val in aggregate_trials(trial_dicts).items():
            result[key][r] = agg_val
    return result


def fit_stochastic_shared(mod, shared_params, r_values, trial_seeds, measurement_fns, seed_key='random_state'):
    """
    Refits `mod` once per trial seed -- for modules whose output does not vary
    with the rule-count budget r -- and broadcasts the trial-aggregated result
    across every r label (matching the pre-existing convention for these modules).
    """
    result = (
        {'lambda': {}, 'max-rule-length': {}, 'sum-rule-length': {}, 'weighted-avg-length': {}} |
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
exp_results['modules']['Decision-Tree'] = fit_stochastic_varying(
    decision_tree_mod, decision_tree_params_by_r, trial_seeds, measurement_fns,
    seed_key='random_state'
)
exp_results['modules']['Exp-Tree'] = fit_stochastic_shared(
    exp_tree_mod, exp_tree_shared_params, n_rules_list, trial_seeds, measurement_fns,
    seed_key='random_state'
)
exp_results['modules']['Shallow-Tree'] = fit_stochastic_shared(
    shallow_tree_mod, shallow_tree_shared_params, n_rules_list, trial_seeds, measurement_fns,
    seed_key='kmeans_random_state'
)
exp_results['modules']['IDS'] = fit_stochastic_varying(
    ids_mod, ids_params_by_r, trial_seeds, measurement_fns,
    seed_key='random_state'
)
print("Stochastic modules done.")

exp.save_results(outfile, outfile_ref)
end = time.time()
print("Experiment time:", end - start)


####################################################################################################

