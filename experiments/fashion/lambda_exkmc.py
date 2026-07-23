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
    CAR_MAX_RULE_LENGTH, CONFIDENCE_DEFAULT, OUTFILE_REF, RULES_DIR, ALPHAS_DIR,
    LAMBDA_DIR,
)

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

# This script's fit (a single deterministic ExKMC fit, broadcast across the
# lambda grid) is already cheap, so it always requests 1 core regardless of
# the pipeline's overall CPU budget -- see experiments/README.md's note on why
# mnist/fashion split lambda across files.
experiment_cpu_count = 1

# REMINDER: The seed should only be initialized here. It should NOT
# within the parameters of any sub-function or class (except for select
# baseline experiments like KMeans), since these will
# reset the seed each time they are given one.
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
    'seed': seed
}

# Unlike max_rules_exkmc.py (whose n_rules_list is a trivial fixed range), this
# script needs the same lambda grid as lambda.py to know which values to
# broadcast ExKMC's single fit across -- see "Lambda grids" below.
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
outfile_ref = '_exkmc' + OUTFILE_REF

####################################################################################################
# Load pre-mined rules:

ensemble_rules = load_rules(RULES_DIR + f'ensemble_rules{OUTFILE_REF}.pkl')

rule_miner_dict = {
    'ensemble': (None, ensemble_rules, None),
}

####################################################################################################
# Objectives for Decision Set Clustering:
#
# Needed only to reproduce the same lambda* / lambda grid as lambda.py (see
# that script's comment for the full rationale) -- this script does not fit PEC itself.

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
# Reproduces exactly the same lambda* / lambda grid computation as lambda.py (see
# that script's comment for the full rationale) -- this is the only reason
# `objective_dict`/alpha values/rules are loaded here at all: ExKMC's result
# doesn't depend on lambda, but it still needs to know every lambda value that
# `lambda.py`'s PEC modules will be evaluated at, so its single fit can be
# broadcast under the same set of keys (`all_lambda_values`) once results are
# merged by `lambda_combine.py`.

n_lambda_points = 10  # matches alphas.py's n_compare convention
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

# Union of every objective's lambda grid -- used to broadcast ExKMC's single
# fit across every lambda value any objective's plot might need to look up
# (matches lambda.py's `all_lambda_values`).
all_lambda_values = tuple(
    sorted(set().union(*(set(g) for g in lambda_grid_dict.values())))
)

####################################################################################################
# Comparison Modules:

# ExKMC
exkmc_shared_params = {
    'k' : fixed_parameters['n_clusters'],
    'kmeans': kmeans_base.clustering,
    'max_leaf_nodes': n_select
}
exkmc_mod = DecisionTreeMod(
    model = ExkmcTree,
    name = 'ExKMC'
)

####################################################################################################


baseline = kmeans_base
module_list = [(exkmc_mod, {all_lambda_values: exkmc_shared_params})]

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
exp.save_results(outfile, outfile_ref)
end = time.time()
print("Experiment time:", end - start)


####################################################################################################
