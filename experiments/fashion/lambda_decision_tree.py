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
from experiments.fashion.config import (
    SEED, N_CLUSTERS, N_SELECT_DEFAULT, MAX_RULES, SHALLOW_TREE_DEPTH_FACTOR,
    N_FOREST, FOREST_MAX_DEPTH, CAR_MIN_SUPPORT, CAR_MIN_CONFIDENCE,
    CAR_MAX_RULE_LENGTH, CONFIDENCE_DEFAULT, N_TRIALS, TRIAL_SEEDS, CPU_COUNT,
    OUTFILE_REF, RULES_DIR, ALPHAS_DIR, LAMBDA_DIR,
)

####################################################################################################
# This is the Decision-Tree-only counterpart to lambda.py -- part of the same per-model split as
# lambda_exkmc.py (see lambda_combine.py, which now merges together whichever of
# lambda_{cba,cn2,dtree,ids,pec,exkmc}.py have completed). lambda.py itself is left unchanged
# and can still be run standalone as a single all-in-one job.
#
# Decision-Tree's fitted solution doesn't depend on lambda at all (it never varies with PEC's
# lambda hyperparameter), so it's refit once per trial seed here and the trial-aggregated result
# is broadcast across every value in `all_lambda_values` -- the union of every objective's lambda
# grid. Computing that grid still requires the same per-objective PEC probe lambda.py/lambda_pec.py
# use (cheap: it reuses the precomputed coverage/cost caches), so this script still loads
# ensemble_rules and the alpha/precomputed-cache paths purely to reproduce that grid identically,
# even though Decision-Tree itself never touches the mined ensemble pool.

import os
import json
import numpy as np
from intercluster import *
from intercluster.decision_trees import *
from intercluster.decision_sets import *
from intercluster.decision_sets.objectives import *
from intercluster.decision_sets.mining import *
from intercluster.measurements import *


# Prevents memory leakage for KMeans:
os.environ["OMP_NUM_THREADS"] = "1"

experiment_cpu_count = CPU_COUNT

# REMINDER: The seed should only be initialized here. It should NOT
# within the parameters of any sub-function or class (except for select
# baseline experiments like KMeans), since these will
# reset the seed each time they are given one.
# Decision-Tree has inherent randomness in its fitted solution (sklearn tree
# tie-breaking), so it's refit across `trial_seeds` below and its results
# recorded as mean/std/values instead of a single point estimate.
seed = SEED

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
with open(ALPHAS_DIR + 'selected_alphas' + OUTFILE_REF + '.json') as f:
    selected_alpha_dict = json.load(f)
fixed_parameters['alpha'] = selected_alpha_dict

decision_info_dict_directory = RULES_DIR

outfile = LAMBDA_DIR
outfile_ref = '_dtree' + OUTFILE_REF

####################################################################################################
# Load pre-mined rules (only needed for the lambda-grid probe below -- Decision-Tree fits
# directly on data/kmeans_labels and never touches the mined ensemble pool):

ensemble_rules = load_rules(RULES_DIR + f'ensemble_rules{OUTFILE_REF}.pkl')

rule_miner_dict = {
    'ensemble': (None, ensemble_rules, None),
}

####################################################################################################
# Decision Tree:

decision_tree_shared_params = {'max_leaf_nodes': n_select}
decision_tree_mod = DecisionTreeMod(
    model = DecisionTree,
    name = 'Decision-Tree'
)

####################################################################################################
# Objectives for Decision Set Clustering (needed only to reproduce the lambda grid
# Decision-Tree's result gets broadcast across -- see the note at the top of this file):

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
}

####################################################################################################
# Lambda grid (see lambda.py/lambda_pec.py for the full rationale):

n_lambda_points = 10  # matches alphas.py's n_compare convention
half = n_lambda_points // 2 + 1

lambda_star_dict = {}
lambda_grid_dict = {}

for obj_name, obj_params in objective_dict.items():
    for rule_miner_name, (rule_miner, rules, rule_labels) in rule_miner_dict.items():
        module_name = f'dscluster; {obj_name}; {rule_miner_name}'
        alpha_val = fixed_parameters['alpha'][module_name]
        base_params = {'n_select': n_select, 'alpha_val': alpha_val} | obj_params

        probe_params = base_params | {'lambda_val': None, 'selection_algorithm': 'distorted-greedy'}
        probe = PEC(rules = rules, **probe_params)
        lambda_star = probe.compute_lambda_star(data, kmeans_labels)

        lower = np.linspace(0.0, lambda_star, half)
        upper = np.linspace(lambda_star, 2 * lambda_star, half)
        lambda_grid = np.concatenate([lower, upper[1:]])

        lambda_star_dict[module_name] = float(lambda_star)
        lambda_grid_dict[module_name] = lambda_grid.tolist()

fixed_parameters['lambda_star'] = lambda_star_dict
fixed_parameters['lambda_grid'] = lambda_grid_dict
fixed_parameters['n_lambda_points'] = n_lambda_points

# Union of every objective's lambda grid -- used to broadcast Decision-Tree's trial-aggregated
# result (which doesn't depend on lambda or the objective at all) across every lambda value any
# objective's plot might need to look up.
all_lambda_values = tuple(
    sorted(set().union(*(set(g) for g in lambda_grid_dict.values())))
)

####################################################################################################

baseline = kmeans_base
# No joblib-dispatched modules here -- Decision-Tree is refit once per trial seed single-process
# below (see lambda.py's "Stochastic module trials" for why: joblib worker processes do not
# inherit this script's seeded global NumPy state, which would make results irreproducible for
# this randomized module). `module_list` is left empty so `exp.run()` still produces the baseline
# entry this script's output JSON needs.
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
# Decision-Tree has a fitted solution that depends on randomness. Rather than record one
# arbitrarily-seeded fit, it is refit once per seed in `trial_seeds` and the results across
# trials are aggregated into {'mean', 'std', 'values'} via `aggregate_trials` (see
# experiments/modules.py). This runs single-process (not through `Experiment`'s joblib dispatch)
# specifically so each trial's explicit seed is what controls its randomness. Since Decision-Tree
# doesn't vary with lambda, it's fit once per trial seed (not once per lambda value) and the
# trial-aggregated result is broadcast across `all_lambda_values`.

def _seed_and_fit(mod, params, trial_seed):
    """
    Fits `mod` for one trial. Sets the trial's explicit seed both as a fitting
    parameter and as the global NumPy seed immediately before fit() -- some
    dependencies still read the global RNG state directly and aren't fully
    parameterized by a passed-in random_state alone. This call runs
    single-process, so setting the global seed here is safe and sufficient
    for reproducibility.
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
    across every value in `r_values`.
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


print(f"Fitting Decision-Tree across {n_trials} trials...")
exp_results['modules']['Decision-Tree'] = fit_stochastic_shared(
    decision_tree_mod, decision_tree_shared_params, all_lambda_values, trial_seeds, measurement_fns,
    seed_key='random_state'
)
print("Decision-Tree done.")

exp.save_results(outfile, outfile_ref)
end = time.time()
print("Experiment time:", end - start)


####################################################################################################
