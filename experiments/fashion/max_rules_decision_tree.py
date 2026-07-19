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
    OUTFILE_REF, MAX_RULES_DIR,
)

####################################################################################################
# This is the Decision-Tree-only counterpart to max_rules.py -- part of the same per-model split
# as max_rules_exkmc.py (see max_rules_combine.py, which now merges together whichever of
# max_rules_{cba,cn2,dtree,ids,pec,exkmc}.py have completed). max_rules.py itself is left
# unchanged and can still be run standalone as a single all-in-one job.
#
# Decision-Tree fits directly on data/kmeans_labels (it doesn't draw from the mined ensemble
# pool), so this script never loads ensemble_rules{OUTFILE_REF}.pkl at all.

import os
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
# tie-breaking), so it's refit across `trial_seeds` below (see "Stochastic
# module trials" in max_rules.py) and its results recorded as mean/std/values
# instead of a single point estimate.
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

n_rules_list = list(range(fixed_parameters['n_clusters'], fixed_parameters['max_rules'] + 1))

np.random.seed(fixed_parameters['seed'])

# Baseline KMeans
kmeans_base = KMeansBase(n_clusters = fixed_parameters['n_clusters'], random_seed = fixed_parameters['seed'])
kmeans_assignment = kmeans_base.assign(data)
kmeans_labels = kmeans_base.labels

# Weights for uncertainty objectives
weights = distance_ratio_score(data, kmeans_base.centers)
fixed_parameters['weights'] = weights.tolist()

outfile = MAX_RULES_DIR
outfile_ref = '_dtree' + OUTFILE_REF

####################################################################################################

decision_tree_params_by_r = {i : {'max_leaf_nodes' : i} for i in n_rules_list}
decision_tree_mod = DecisionTreeMod(
    model = DecisionTree,
    name = 'Decision-Tree'
)

####################################################################################################

baseline = kmeans_base
# No joblib-dispatched modules here -- Decision-Tree is refit once per (rule-count, trial seed)
# pair single-process below (see "Stochastic module trials" in max_rules.py for why: joblib
# worker processes do not inherit this script's seeded global NumPy state, which would make
# results irreproducible for this randomized module). `module_list` is left empty so
# `exp.run()` still produces the baseline entry this script's output JSON needs.
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
    measurement_fns = measurement_fns,
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
# specifically so each trial's explicit seed is what controls its randomness.

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


def fit_stochastic_varying(mod, params_by_r, trial_seeds, measurement_fns, seed_key='random_state'):
    """
    Refits `mod` once per (rule-count r, trial seed) pair -- for modules whose
    output genuinely varies with the rule-count budget r -- and aggregates
    results across trials for each r.
    """
    result = (
        {'lambda': {}, 'lambda_n_rules': {}, 'max-rule-length': {},
         'sum-rule-length': {}, 'weighted-avg-length': {}} |
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


print(f"Fitting Decision-Tree across {n_trials} trials each...")
exp_results['modules']['Decision-Tree'] = fit_stochastic_varying(
    decision_tree_mod, decision_tree_params_by_r, trial_seeds, measurement_fns,
    seed_key='random_state'
)
print("Decision-Tree done.")

exp.save_results(outfile, outfile_ref)
end = time.time()
print("Experiment time:", end - start)


####################################################################################################
