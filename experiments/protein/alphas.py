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
import math
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

experiment_cpu_count = 12

# REMINDER: The seed should only be initialized here. It should NOT
# within the parameters of any sub-function or class (except for select
# baseline experiments like KMeans), since these will
# reset the seed each time they are given one.
# alphas.py (alpha selection) is a one-time, cached hyperparameter-selection
# step rather than a model under evaluation, so it is run once under this
# single seed rather than repeated across trials -- see experiments/README.md
# ("Reproducibility") for which downstream models (IDS, ExplanationTree,
# DecisionTree, ShallowTree) are instead re-fit across multiple trial seeds in
# max_rules.py/confidence.py.
seed = 342

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
data, data_labels, feature_labels, scaler = load_preprocessed_protein()
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
    'car_min_confidence': 0.75,
    'car_max_rule_length': 3, # (really means 6 by pyfim convention)
    'filter_confidence': 0.75,
    'seed': seed
}

np.random.seed(fixed_parameters['seed'])

# Do baseline clustering
kmeans_base = KMeansBase(n_clusters = fixed_parameters['n_clusters'], random_seed = fixed_parameters['seed'])
kmeans_assignment = kmeans_base.assign(data)
kmeans_labels = kmeans_base.labels
kmeans_distances = pairwise_distances(data, kmeans_base.centers)
furthest_distance = np.max(np.max(kmeans_distances, axis=1))
largest_cluster_size = np.max(np.bincount(flatten_labels(kmeans_labels)))

# Weights for uncertainty objectives
weights = distance_ratio_score(data, kmeans_base.centers)
fixed_parameters['weights'] = weights.tolist()

decision_info_dict_directory = 'data/experiments/protein/rules/'

outfile = 'data/experiments/protein/alphas/'
outfile_ref = '_resub'

####################################################################################################
# Load pre-mined rules:

ensemble_rules = load_rules('data/experiments/protein/rules/ensemble_rules.pkl')

rule_miner_dict = {
    'ensemble': (None, ensemble_rules, None),
}

####################################################################################################
# Define Objectives:


objective_dict = {
    'coverage-mistake': {
        'objective_type': 'coverage-mistake',
        'n_select': fixed_parameters['n_select'],
        'precomputed_path': os.path.join(
            decision_info_dict_directory, 'mistake_info_dict.pkl.gz'
        )
    },
    'coverage-cost': {
        'objective_type': 'coverage-cost',
        'n_select': fixed_parameters['n_select'],
        'cluster_centers': kmeans_base.centers,
        'cluster_cost_method': 'kmeans',
        'precomputed_path': os.path.join(
            decision_info_dict_directory, 'cost_info_dict.pkl.gz'
        )
    },
    'coverage-pairwise-distance': {
        'objective_type': 'coverage-pairwise-distance',
        'n_select': fixed_parameters['n_select'],
        'precomputed_path': os.path.join(
            decision_info_dict_directory, 'pairwise_distance_info_dict.pkl.gz'
        )
    },
    'coverage-mistake-weighted': {
        'objective_type': 'coverage-mistake',
        'n_select': fixed_parameters['n_select'],
        'weights': weights,
        'precomputed_path': os.path.join(
            decision_info_dict_directory, 'mistake_info_dict.pkl.gz'
        )
    },
    'coverage-cost-weighted': {
        'objective_type': 'coverage-cost',
        'n_select': fixed_parameters['n_select'],
        'cluster_centers': kmeans_base.centers,
        'weights': weights,
        'cluster_cost_method': 'kmeans',
        'precomputed_path': os.path.join(
            decision_info_dict_directory, 'cost_info_dict.pkl.gz'
        )
    },
    'coverage-pairwise-distance-weighted': {
        'objective_type': 'coverage-pairwise-distance',
        'n_select': fixed_parameters['n_select'],
        'weights': weights,
        'precomputed_path': os.path.join(
            decision_info_dict_directory, 'pairwise_distance_info_dict.pkl.gz'
        )
    },
}


# List of alpha values to try for each objective
n_compare = 25
objective_alpha_dict = {
    'coverage-mistake': np.linspace(0.0, 0.5 * fixed_parameters['n'], num = n_compare),
    'coverage-cost': np.linspace(0.0, 0.5 * furthest_distance * n, num = n_compare),
    'coverage-pairwise-distance': np.linspace(0.0, 0.01 * n * 2 * largest_cluster_size, num = n_compare),
    'coverage-mistake-weighted': np.linspace(0.0, 0.5 * fixed_parameters['n'], num = n_compare),
    'coverage-cost-weighted': np.linspace(0.0, 0.5 * furthest_distance * n, num = n_compare),
    'coverage-pairwise-distance-weighted': np.linspace(0.0, 0.01 * n * 2 * largest_cluster_size, num = n_compare),
}


####################################################################################################
# Create experiment modules:

module_list = []
for rule_miner_name, (rule_miner, rules, rule_labels) in rule_miner_dict.items():
    for obj_name, obj_params in objective_dict.items():
        module_name = f'dscluster; {obj_name}; {rule_miner_name}'
        alpha_vals = objective_alpha_dict[obj_name]
        for alpha in alpha_vals:
            obj_parameterized = {
                (alpha_val,): obj_params | {'alpha_val': alpha_val}
                for alpha_val in objective_alpha_dict[obj_name]
            }

            module = DecisionSetMod(
                model = PEC,
                rules = rules,
                name = f'dscluster; {obj_name}; {rule_miner_name}'
            )
            module_list.append((module, obj_parameterized))


####################################################################################################
# Run Experiment:

measurement_fns = [
    TotalCoverage(),
    TotalCoverage(weights = weights, name = 'total-coverage-weighted'),
    ClusterCoverage(baseline_assignment = kmeans_assignment),
    ClusterCoverage(
        baseline_assignment = kmeans_assignment, weights = weights, name = 'cluster-coverage-weighted'
    ),
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