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
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from intercluster import *
from intercluster.decision_trees import *
from intercluster.decision_sets import *
from intercluster.decision_sets.objectives import *
from intercluster.decision_sets.mining import *
from intercluster.measurements import *
from intercluster.rules import load_rules

# Prevents memory leakage for KMeans:
os.environ["OMP_NUM_THREADS"] = "1"

experiment_cpu_count = 16

# REMINDER: The seed should only be initialized here. It should NOT 
# within the parameters of any sub-function or class (except for select 
# baseline experiments like KMeans), since these will 
# reset the seed each time they are given one. 
seed = 342

####################################################################################################
# Read and process data:
data, data_labels, feature_labels, scaler = load_preprocessed_mnist()
#euclidean_distances = pairwise_distances(data)
n,d = data.shape

##### Parameters #####
#lambdas_fname = 'data/experiments/climate/lambdas/selected_lambdas_alpha_zero.json'
#with open(lambdas_fname, 'r') as f:
#    selected_lambdas = json.load(f)

fixed_parameters = {
    'n' : n,
    'd' : d,
    'n_clusters': 10,
    'max_rules': 20,
    'min_support': 0.1,
    'min_confidence': 0.9,
    'max_rule_length': 4,
    'depth_factor': 0.03,
    'ids_samples': 1,
    'forest_samples': 10,
    'alpha_mistakes': 0.01 * n * 1.0,
    'lambdas' : {},
}

n_rules_list = list(range(fixed_parameters['n_clusters'], fixed_parameters['max_rules'] + 1))

np.random.seed(seed)

# Baseline KMeans
kmeans_base = KMeansBase(n_clusters = fixed_parameters['n_clusters'], random_seed = seed)
kmeans_assignment = kmeans_base.assign(data)
kmeans_labels = kmeans_base.labels

# Find average distance of points to their closest cluster center
kmeans_distances = pairwise_distances(data, kmeans_base.centers)
max_distances = np.max(kmeans_distances, axis=1)
max_distance = np.max(max_distances)
fixed_parameters['alpha_rule_clustering_cost'] = 0.01 * n * max_distance

####################################################################################################
# Load pre-mined rules:

ensemble_rules = load_rules('data/experiments/mnist/rules/ensemble_rules.pkl')

rule_miner_dict = {
    'ensemble': (None, ensemble_rules, None),
}


####################################################################################################
# Comparison Modules:

# Decision Tree
decision_tree_params = {(i,) : {'max_leaf_nodes' : i, 'random_state' : seed}
                        for i in n_rules_list}
decision_tree_mod = DecisionTreeMod(
    model = DecisionTree,
    name = 'Decision-Tree'
)


# Explanation Tree
exp_tree_params = {tuple(n_rules_list) : {'num_clusters' : fixed_parameters['n_clusters']}}
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
shallow_tree_params = {
    tuple(n_rules_list) : {
        'n_clusters' : fixed_parameters['n_clusters'],
        'depth_factor' : fixed_parameters['depth_factor'],
        'kmeans_random_state' : seed
    } for i in n_rules_list
}
shallow_tree_mod = DecisionTreeMod(
    model = ShallowTree,
    name = 'Shallow-Tree'
)


# IDS:
'''
rule_comb = len(class_association_rules) * fixed_parameters['n_clusters']
ids_lambdas = [
    1/rule_comb,
    1/(2 * data.shape[1] * rule_comb),
    1/(len(data) * (rule_comb**2)),
    1/(len(data) * (rule_comb**2)),
    1/fixed_parameters['n_clusters'],
    1/(data.shape[0] * rule_comb),
    1/(data.shape[0])
]

ids_module_list = []
for s in range(fixed_parameters['ids_samples']):
    ids_params = {
        tuple(n_rules_list) : {
            'lambdas' : ids_lambdas
        }
    }
    ids_mod = DecisionSetMod(
        model = IDS,
        fitting_params = {'bin_df': class_association_rule_miner.bin_df},
        rules = class_association_rules,
        name = f"IDS_{s}"
    )
    ids_module_list.append((ids_mod, ids_params))
'''

####################################################################################################
# Objectives for Decision Set Clustering:

objective_dict = {
    'coverage-mistake': {
        'alpha_val': fixed_parameters['alpha_mistakes'],
        'objective_type': 'coverage-mistake'
    },
    'total-coverage-mistake': {
        'alpha_val': fixed_parameters['alpha_mistakes'],
        'objective_type': 'total-coverage-mistake'
    },
    'coverage-cost': {
        'alpha_val': fixed_parameters['alpha_rule_clustering_cost'],
        'cluster_centers': kmeans_base.centers,
        'objective_type': 'coverage-cost',
        'cluster_cost_method': 'kmeans'
    },
    'total-coverage-cost': {
        'alpha_val': fixed_parameters['alpha_rule_clustering_cost'],
        'cluster_centers': kmeans_base.centers,
        'objective_type': 'total-coverage-cost',
        'cluster_cost_method': 'kmeans'
    }
}


####################################################################################################
# Find max lambda values among all rule miners

for obj_name, obj_params in objective_dict.items():
    obj_mod_name = f'dscluster; {obj_name}'
    max_lambda = 0.0
    for rule_miner_name, (rule_miner, rules, rule_labels) in rule_miner_dict.items():
        dsclust = DSCluster(
            rules = rules,
            n_select = n_rules_list[0],
            **obj_params
        )
        dsclust.fit(data, kmeans_labels)
        lambda_val = dsclust.objective.lambda_val
        if lambda_val > max_lambda:
            max_lambda = lambda_val
    
    print(f'Found max lambda for {obj_mod_name}: {max_lambda}')
    for rule_miner_name, (rule_miner, rules, rule_labels) in rule_miner_dict.items():
        module_name = obj_mod_name + f'; {rule_miner_name}'
        fixed_parameters['lambdas'][module_name] = max_lambda


####################################################################################################
# Decision Set Clustering Modules:

dscluster_module_list = []
for obj_name, obj_params in objective_dict.items():
    obj_mod_name = f'dscluster; {obj_name}'
    for rule_miner_name, (rule_miner, rules, rule_labels) in rule_miner_dict.items():
        module_name = obj_mod_name + f'; {rule_miner_name}'
        lambda_val = fixed_parameters['lambdas'][module_name]

        # Decision Set Clustering Parameters:
        dsclust_params = {
            (r,) : {'n_select' : r, 'lambda_val' : lambda_val} | obj_params
            for i,r in enumerate(n_rules_list)
        }

        dsclust_mod = DecisionSetMod(
            model = DSCluster,
            rules = rules,
            name = module_name
        )
        dscluster_module_list.append((dsclust_mod, dsclust_params))


####################################################################################################


baseline = kmeans_base
module_list = [
    (decision_tree_mod, decision_tree_params),
    (exp_tree_mod, exp_tree_params),
    (exkmc_mod, exkmc_params),
    (shallow_tree_mod, shallow_tree_params),
] + dscluster_module_list #+ ids_module_list


measurement_fns = [
    TotalCoverage(),
    ClusterCoverage(baseline_assignment = kmeans_assignment),
    Mistakes(baseline_assignment = kmeans_assignment),
    ClusteringCost(data = data, average = True, normalize = True, method = "kmeans"),
    RuleClusteringCost(data = data, cluster_centers = kmeans_base.centers, method = "kmeans"),
    PairwiseDistance(baseline_assignment = kmeans_assignment),
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
exp.save_results('data/experiments/mnist/max_rules/', '')
end = time.time()
print("Experiment time:", end - start)


####################################################################################################

