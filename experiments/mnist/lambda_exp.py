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
from experiments.cli_utils import conf_tag, parse_experiment_args

####################################################################################################

import os
import json
import math
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

args = parse_experiment_args(confidence_default=0.75, cpu_count_default=1)
confidence_threshold = args.confidence
tag = conf_tag(confidence_threshold)
experiment_cpu_count = args.cpu_count

# REMINDER: The seed should only be initialized here. It should NOT
# within the parameters of any sub-function or class (except for select
# baseline experiments like KMeans), since these will
# reset the seed each time they are given one.
# ExplanationTree accepts an explicit random_state instead of relying on this
# global seed -- see `trial_seeds` below, which derives one seed per trial so
# Exp-Tree can be refit across multiple trials and its results reported as
# mean/std rather than a single, arbitrarily-seeded point estimate.
seed = 342

# Number of independent random-seed trials used to evaluate Exp-Tree (matches
# `lambda.py`'s Decision-Tree/Shallow-Tree/IDS trial count). `trial_seeds` is
# derived deterministically from `seed` so re-running this script reproduces the
# exact same set of trials.
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
data, data_labels, feature_labels, scaler = load_preprocessed_mnist()
data = _memoryview_safe(data)
n,d = data.shape

fixed_parameters = {
    'n': n,
    'd': d,
    'n_clusters': 10,
    'n_select': 10,
    'max_rules': 16,
    'shallow_tree_depth_factor': 0.03,
    'n_forest': 100,
    'forest_max_depth': 6,
    'car_min_support': 0.025,
    'car_min_confidence': 0.75,
    'car_max_rule_length': 2, # (really means 4 by pyfim convention)
    'filter_confidence': confidence_threshold,
    'seed': seed,
    'n_trials': n_trials,
    'trial_seeds': trial_seeds,
}

# Unlike max_rules_exp.py (whose n_rules_list is a trivial fixed range), this
# script needs the same lambda grid as lambda.py to know which values to
# broadcast Exp-Tree's result across -- see "Lambda grids" below.
n_select = fixed_parameters['n_select']

np.random.seed(fixed_parameters['seed'])

# Baseline KMeans
kmeans_base = KMeansBase(n_clusters = fixed_parameters['n_clusters'], random_seed = fixed_parameters['seed'])
kmeans_assignment = kmeans_base.assign(data)
kmeans_labels = kmeans_base.labels

# Weights for uncertainty objectives
weights = distance_ratio_score(data, kmeans_base.centers)
fixed_parameters['weights'] = weights.tolist()

# Alpha values for objectives:
with open(f"data/experiments/mnist/alphas/selected_alphas_resub_conf_{tag}.json") as f:
    selected_alpha_dict = json.load(f)
fixed_parameters['alpha'] = selected_alpha_dict

decision_info_dict_directory = 'data/experiments/mnist/rules/'

outfile = 'data/experiments/mnist/lambda/'
outfile_ref = f'_resub_exp_conf_{tag}'

####################################################################################################
# Load pre-mined rules:

ensemble_rules = load_rules(f'data/experiments/mnist/rules/ensemble_rules_conf_{tag}.pkl')

rule_miner_dict = {
    'ensemble': (None, ensemble_rules, None),
}

####################################################################################################
# Objectives for Decision Set Clustering:
#
# Needed only to reproduce the same lambda* / lambda grid as lambda.py (see
# below) -- this script does not fit PEC itself.

objective_dict = {
    'coverage-mistake': {
        'objective_type': 'coverage-mistake',
        'selection_algorithm': 'distorted-greedy',
        'precomputed_path': os.path.join(
            decision_info_dict_directory, f'mistake_info_dict_conf_{tag}.pkl.gz'
        )
    },
    'coverage-cost': {
        'objective_type': 'coverage-cost',
        'cluster_centers': kmeans_base.centers,
        'cluster_cost_method': 'kmeans',
        'selection_algorithm': 'distorted-greedy',
        'precomputed_path': os.path.join(
            decision_info_dict_directory, f'cost_info_dict_conf_{tag}.pkl.gz'
        )
    },
    'coverage-pairwise-distance': {
        'objective_type': 'coverage-pairwise-distance',
        'selection_algorithm': 'distorted-greedy',
        'precomputed_path': os.path.join(
            decision_info_dict_directory, f'pairwise_distance_info_dict_conf_{tag}.pkl.gz'
        )
    },
    # 'coverage-mistake-weighted': {
    #     'objective_type': 'coverage-mistake',
    #     'weights': weights,
    #     'selection_algorithm': 'distorted-greedy',
    #     'precomputed_path': os.path.join(
    #         decision_info_dict_directory, f'mistake_info_dict_conf_{tag}.pkl.gz'
    #     )
    # },
    # 'coverage-cost-weighted': {
    #     'objective_type': 'coverage-cost',
    #     'cluster_centers': kmeans_base.centers,
    #     'weights': weights,
    #     'cluster_cost_method': 'kmeans',
    #     'selection_algorithm': 'distorted-greedy',
    #     'precomputed_path': os.path.join(
    #         decision_info_dict_directory, f'cost_info_dict_conf_{tag}.pkl.gz'
    #     )
    # },
    # 'coverage-pairwise-distance-weighted': {
    #     'objective_type': 'coverage-pairwise-distance',
    #     'weights': weights,
    #     'selection_algorithm': 'distorted-greedy',
    #     'precomputed_path': os.path.join(
    #         decision_info_dict_directory, f'pairwise_distance_info_dict_conf_{tag}.pkl.gz'
    #     )
    # },
}

####################################################################################################
# Lambda grids:
#
# Reproduces exactly the same lambda* / lambda grid computation as lambda.py (see
# that script's comment for the full rationale) -- this is the only reason
# `objective_dict`/alpha values/rules are loaded here at all: Exp-Tree's result
# doesn't depend on lambda, but it still needs to know every lambda value that
# `lambda.py`'s PEC modules will be evaluated at, so its single per-trial fit can
# be broadcast under the same set of keys (`all_lambda_values`) once results are
# merged by `lambda_combine.py`.

n_lambda_points = 25  # matches alphas.py's n_compare convention
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

# Union of every objective's lambda grid -- used to broadcast Exp-Tree's single
# per-trial fit across every lambda value any objective's plot might need to
# look up (matches lambda.py's `all_lambda_values`).
all_lambda_values = tuple(
    sorted(set().union(*(set(g) for g in lambda_grid_dict.values())))
)

####################################################################################################
# Comparison Modules:
#
# Exp-Tree has a fitted solution that depends on randomness (heap tie-breaking,
# and -- per explanation_tree.py's `random_state` docstring caveat -- its
# compiled Cython splitter's own tie-breaks, which read the global NumPy RNG
# state directly). Rather than fit once under the single global `seed`, it is
# refit across `trial_seeds` below (see "Stochastic module trial") and its
# results are recorded as mean/std/values instead of a single point estimate.
# (ExplanationTree's leaf count is fixed at num_clusters, independent of both
# the rule budget and lambda, so a single fit's result is recorded under every
# lambda label.)

exp_tree_shared_params = {'num_clusters' : fixed_parameters['n_clusters']}
exp_tree_mod = DecisionTreeMod(
    model = ExplanationTree,
    name = 'Exp-Tree'
)


####################################################################################################


baseline = kmeans_base
module_list = []

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
# Stochastic module trial
#
# Exp-Tree's fitted solution depends on randomness. Rather than record one
# arbitrarily-seeded fit, it is refit once per seed in `trial_seeds` and the
# results across trials are aggregated into {'mean', 'std', 'values'} via
# `aggregate_trials` (see experiments/modules.py). This mirrors the treatment
# of Decision-Tree/Shallow-Tree/IDS in `lambda.py`.

def _seed_and_fit(mod, params, trial_seed):
    """
    Fits `mod` for one trial. Sets the trial's explicit seed both as a fitting
    parameter AND as the global NumPy seed immediately before fit() -- some
    dependencies (e.g. ExplanationTree's compiled Cython splitter, see
    explanation_tree.py's `random_state` docstring caveat) still read the
    global RNG state directly and aren't fully parameterized by a passed-in
    random_state alone. This call runs single-process, so setting the global
    seed here is safe and sufficient for reproducibility.
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
    this module in `lambda.py`).
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


print(f"Fitting Exp-Tree across {n_trials} trials...")
exp_results['modules']['Exp-Tree'] = fit_stochastic_shared(
    exp_tree_mod, exp_tree_shared_params, all_lambda_values, trial_seeds, measurement_fns,
    seed_key='random_state'
)
print("Exp-Tree trials done.")

exp.save_results(outfile, outfile_ref)
end = time.time()
print("Experiment time:", end - start)


####################################################################################################
