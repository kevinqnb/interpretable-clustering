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

experiment_cpu_count = 12

# REMINDER: The seed should only be initialized here. It should NOT 
# within the parameters of any sub-function or class (except for select 
# baseline experiments like KMeans), since these will 
# reset the seed each time they are given one. 
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
data, data_labels, feature_labels, scaler = load_preprocessed_yeast()
data = _memoryview_safe(data)
n,d = data.shape

fixed_parameters = {
    'n': n,
    'd': d,
    'n_clusters': 9,
    'n_select': 9,
    'max_rules': 15,
    'shallow_tree_depth_factor': 0.03,
    'n_forest': 100,
    'forest_max_depth': 6,
    'car_min_support': 0.025,
    'car_min_confidence': 0.85,
    'car_max_rule_length': 3, # (really means 6 by pyfim convention)
    'filter_confidence': 0.85,
    'seed': seed
}


confidence_list = np.linspace(0.0, 1.0, num=11)[:-1]

np.random.seed(fixed_parameters['seed'])

# Baseline KMeans
kmeans_base = KMeansBase(n_clusters = fixed_parameters['n_clusters'], random_seed = fixed_parameters['seed'])
kmeans_assignment = kmeans_base.assign(data)
kmeans_labels = kmeans_base.labels

# Weights for uncertainty objectives
weights = distance_ratio_score(data, kmeans_base.centers)
fixed_parameters['weights'] = weights.tolist()

# Alpha values for objectives:
with open("data/experiments/yeast/alphas/selected_alphas_rule_length.json") as f:
    selected_alpha_dict = json.load(f)
fixed_parameters['alpha'] = selected_alpha_dict

decision_info_dict_directory = 'data/experiments/yeast/rules/'

outfile = 'data/experiments/yeast/sensitivity/'
outfile_ref = '_min_filter_confidence'

####################################################################################################
# Load pre-mined rules:

decision_tree_rules = load_rules('data/experiments/yeast/rules/decision_tree_rules.pkl')
exkmc_rules = load_rules('data/experiments/yeast/rules/exkmc_rules.pkl')
shallow_tree_rules = load_rules('data/experiments/yeast/rules/shallow_rules.pkl')
forest_rules = load_rules('data/experiments/yeast/rules/forest_rules.pkl')


rule_miner_dict = {}
for c in confidence_list:
    ensemble_rules = decision_tree_rules + shallow_tree_rules + forest_rules + exkmc_rules
    rule_miner_dict[f'ensemble_conf={c:.2f}'] = (None, filter_rules(
        ensemble_rules, data, kmeans_labels, confidence = c
    ), None)

for rule_miner_name, (rule_miner, rules, rule_labels) in rule_miner_dict.items():
    print(f"{rule_miner_name}: {len(rules)} rules")

####################################################################################################
# Comparison Modules:

# Decision Tree
decision_tree_params = {
    (1,) : {'max_leaf_nodes' : fixed_parameters['n_select'], 'random_state' : fixed_parameters['seed']}
}
decision_tree_mod = DecisionTreeMod(
    model = DecisionTree,
    name = 'Decision-Tree'
)


# Explanation Tree
exp_tree_params = {(1,) : {'num_clusters' : fixed_parameters['n_clusters']}}
exp_tree_mod = DecisionTreeMod(
    model = ExplanationTree,
    name = 'Exp-Tree'
)


# ExKMC
exkmc_params = {
    (1,) : {
        'k' : fixed_parameters['n_clusters'],
        'kmeans': kmeans_base.clustering,
        'max_leaf_nodes': fixed_parameters['n_select']
    }
}
exkmc_mod = DecisionTreeMod(
    model = ExkmcTree,
    name = 'ExKMC'
)


# Shallow Tree
shallow_tree_params = {
    (1,) : {
        'n_clusters' : fixed_parameters['n_select'],
        'depth_factor' : fixed_parameters['shallow_tree_depth_factor'],
        'kmeans_random_state' : fixed_parameters['seed']
    }
}
shallow_tree_mod = DecisionTreeMod(
    model = ShallowTree,
    name = 'Shallow-Tree'
)

####################################################################################################
# Objectives for Decision Set Clustering:

objective_dict = {
    'coverage-mistake': {
        'objective_type': 'coverage-mistake',
        'selection_algorithm': 'distorted-greedy',
        #'precomputed_path': os.path.join(
        #    decision_info_dict_directory, 'mistake_info_dict_sensitivity.pkl.gz'
        #)
    },
    'coverage-cost': {
        'objective_type': 'coverage-cost',
        'cluster_centers': kmeans_base.centers,
        'cluster_cost_method': 'kmeans',
        'selection_algorithm': 'distorted-greedy',
        #'precomputed_path': os.path.join(
        #    decision_info_dict_directory, 'cost_info_dict_sensitivity.pkl.gz'
        #)
    },
    'coverage-pairwise-distance': {
        'objective_type': 'coverage-pairwise-distance',
        'selection_algorithm': 'distorted-greedy',
        #'precomputed_path': os.path.join(
        #    decision_info_dict_directory, 'pairwise_distance_info_dict_sensitivity.pkl.gz'
        #)
    },
}

####################################################################################################
# Decision Set Clustering Modules:

dscluster_module_list = []
for obj_name, obj_params in objective_dict.items():
    for rule_miner_name, (rule_miner, rules, rule_labels) in rule_miner_dict.items():
        module_name = f'dscluster; {obj_name}; ensemble'
        module_full_name = f'dscluster; {obj_name}; {rule_miner_name}'
        alpha_val = fixed_parameters['alpha'][module_name]
        dsclust_params = {
            (1,) : {'n_select' : fixed_parameters['n_select'], 'alpha_val' : alpha_val} | obj_params
        }
        dsclust_mod = DecisionSetMod(
            model = PEC,
            rules = rules,
            name = module_full_name
        )
        dscluster_module_list.append((dsclust_mod, dsclust_params))


####################################################################################################


baseline = kmeans_base
module_list = dscluster_module_list
module_list = [
    (decision_tree_mod, decision_tree_params),
    (exp_tree_mod, exp_tree_params),
    (exkmc_mod, exkmc_params),
    (shallow_tree_mod, shallow_tree_params),
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
exp.save_results(outfile, outfile_ref)
end = time.time()
print("Experiment time:", end - start)


####################################################################################################

