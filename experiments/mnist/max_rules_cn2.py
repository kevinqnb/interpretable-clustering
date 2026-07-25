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
from experiments.mnist.config import (
    SEED, N_CLUSTERS, N_SELECT_DEFAULT, MAX_RULES, SHALLOW_TREE_DEPTH_FACTOR,
    N_FOREST, FOREST_MAX_DEPTH, CAR_MIN_SUPPORT, CAR_MIN_CONFIDENCE,
    CAR_MAX_RULE_LENGTH, CONFIDENCE_DEFAULT, CPU_COUNT, OUTFILE_REF, MAX_RULES_DIR,
)

####################################################################################################
# This is the CN2-only counterpart to max_rules.py -- part of the same per-model split as
# max_rules_exkmc.py (see max_rules_combine.py, which now merges together whichever of
# max_rules_{cba,cn2,dtree,ids,pec,exkmc}.py have completed). max_rules.py itself is left
# unchanged and can still be run standalone as a single all-in-one job.
#
# CN2 induces its own rules directly from data/kmeans_labels (it doesn't draw from the mined
# ensemble pool), so this script never loads ensemble_rules{OUTFILE_REF}.pkl at all.

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

# REMINDER: Initialize the seed only here, not inside any sub-function or
# class (except select baseline experiments like KMeans) -- passing a seed
# there resets it on every call.
seed = SEED

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
}

n_rules_list = list(range(fixed_parameters['n_clusters'], fixed_parameters['max_rules'] + 1))

np.random.seed(fixed_parameters['seed'])

kmeans_base = KMeansBase(n_clusters = fixed_parameters['n_clusters'], random_seed = fixed_parameters['seed'])
kmeans_assignment = kmeans_base.assign(data)
kmeans_labels = kmeans_base.labels

# Weights for uncertainty objectives
weights = distance_ratio_score(data, kmeans_base.centers)
fixed_parameters['weights'] = weights.tolist()

outfile = MAX_RULES_DIR
outfile_ref = '_cn2' + OUTFILE_REF

####################################################################################################

baseline = kmeans_base
# No joblib-dispatched modules here -- CN2's induction doesn't depend on n_select (only the
# post-hoc truncation to the first n_select rules does), so it is fit outside `Experiment.run()`
# below (induce once, then finalize + measure cheaply per budget) rather than rerunning the same
# expensive induction once per rule budget the way CBA is swept. `module_list` is left empty so
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
# CN2
#
# CN2's beam-search induction doesn't depend on n_select -- only the post-hoc
# truncation to the first n_select rules does (see cn2.py's induce()/
# finalize() split). CN2 is deterministic (no seed dependence), so it's fit
# outside `Experiment.run()` here: induce once, then finalize + measure
# cheaply per budget.

def fit_cn2_varying(n_rules_list, measurement_fns):
    result = (
        {'lambda': {}, 'lambda_n_rules': {}, 'max-rule-length': {},
         'sum-rule-length': {}, 'weighted-avg-length': {}} |
        {fn.name: {} for fn in measurement_fns}
    )
    cn2 = CN2()
    cn2.induce(data, kmeans_labels)
    n_unique = len(unique_labels(kmeans_labels))
    for r in n_rules_list:
        cn2.finalize(r)
        assignments = (
            cn2.get_data_to_rules_assignment(data),
            cn2.get_rules_to_clusters_assignment(n_labels=n_unique),
            labels_to_assignment(cn2.predict(data), n_labels=n_unique),
        )
        trial_result = {
            'lambda': np.nan,
            'lambda_n_rules': np.nan,
            'max-rule-length': cn2.max_rule_length,
            'sum-rule-length': cn2.get_sum_of_rule_lengths(),
            'weighted-avg-length': cn2.get_weighted_average_rule_length(data),
        } | {
            fn.name: fn(*assignments) for fn in measurement_fns
        }
        for key, val in trial_result.items():
            result[key][r] = val
    return result


print("Fitting CN2 (induce once, finalize per rule budget)...")
exp_results['modules']['CN2'] = fit_cn2_varying(n_rules_list, measurement_fns)
print("CN2 done.")

exp.save_results(outfile, outfile_ref)
end = time.time()
print("Experiment time:", end - start)


####################################################################################################
