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
import numpy as np
import pandas as pd
from intercluster import *
from intercluster.decision_trees import *
from intercluster.decision_sets import *
from intercluster.decision_sets.objectives import *
from intercluster.decision_sets.mining import *
from intercluster.measurements import *
from intercluster.rules import save_rules

# Prevents memory leakage for KMeans:
os.environ["OMP_NUM_THREADS"] = "1"

# REMINDER: The seed should only be initialized here. It should NOT 
# within the parameters of any sub-function or class (except for select 
# baseline experiments like KMeans), since these will 
# reset the seed each time they are given one. 
seed = 342

####################################################################################################
# Read and process data:
data, data_labels, feature_labels, scaler = load_preprocessed_climate("data/climate")
n,d = data.shape

fixed_parameters = {
    'n' : n,
    'd' : d,
    'n_clusters': 6,
    'max_rules': 12,
    'min_support': 0.05,
    'min_confidence': 0.85,
    'car_max_rule_length': 4,
    'n_forest': 100,
    'max_depth': None,
    'depth_factor': 0.03,
    'ids_samples': 10,
    'seed': seed,
}

np.random.seed(fixed_parameters['seed'])

# Do baseline clustering
kmeans_base = KMeansBase(n_clusters = fixed_parameters['n_clusters'], random_seed = fixed_parameters['seed'])
kmeans_assignment = kmeans_base.assign(data)
kmeans_labels = kmeans_base.labels

# Weights for uncertainty objectives
weights = distance_ratio_score(data, kmeans_base.centers)
fixed_parameters['weights'] = weights.tolist()

rules_directory = 'data/experiments/climate/rules/'
os.makedirs(rules_directory, exist_ok = True)

####################################################################################################
# Create bin_df for rule mining:

bin_df = entropy_bin(
    data, kmeans_labels, random_state = fixed_parameters['seed']
)

bin_df.to_csv(rules_directory + 'bin_df.csv', index = False)

####################################################################################################
# Mine for rules:

decision_tree_rule_miner = TreeMiner(
    tree = DecisionTree(random_state = fixed_parameters['seed']),
)
decision_tree_rules, decision_tree_rule_labels = decision_tree_rule_miner.fit(
    X = data, y = kmeans_base.labels
)

print("Mined DT rules:", len(decision_tree_rules))
save_rules(decision_tree_rules, rules_directory + 'decision_tree_rules.pkl')


exkmc_rule_miner = TreeMiner(
    tree = ExkmcTree(
        k = fixed_parameters['n_clusters'],
        max_leaf_nodes = fixed_parameters['max_rules'],
        kmeans = kmeans_base.clustering,
        imm = True
    )
)
exkmc_rules, exkmc_rule_labels = exkmc_rule_miner.fit(
    X = data, y = kmeans_base.labels
)

print("Mined ExKMC rules:", len(exkmc_rules))
save_rules(exkmc_rules, rules_directory + 'exkmc_rules.pkl')

shallow_tree_miner = TreeMiner(
    tree = ShallowTree(
        n_clusters = fixed_parameters['n_clusters'],
        depth_factor = fixed_parameters['depth_factor'],
        kmeans_random_state = fixed_parameters['seed']
    )
)
shallow_rules, shallow_rule_labels = shallow_tree_miner.fit(
    X = data, y = kmeans_labels
)

print("Mined Shallow rules:", len(shallow_rules))
save_rules(shallow_rules, rules_directory + 'shallow_rules.pkl')


forest_rule_miner = RandomForestMiner(
    forest_params = {
        'n_estimators': fixed_parameters['n_forest'],
        'max_depth': fixed_parameters['max_depth'],
        'random_state': fixed_parameters['seed']
    }
)
forest_rules, forest_rule_labels = forest_rule_miner.fit(data, kmeans_base.labels)

print("Mined Forest rules:", len(forest_rules))
save_rules(forest_rules, rules_directory + 'forest_rules.pkl')

class_association_rule_miner = ClassAssociationRuleMiner(
    min_support = fixed_parameters['min_support'],
    min_confidence = fixed_parameters['min_confidence'],
    max_length = fixed_parameters['car_max_rule_length'],
    bin_df = bin_df
)
class_association_rules, class_association_rule_labels = class_association_rule_miner.fit(
    X = data, y = kmeans_base.labels
)

print("Mined CAR rules:", len(class_association_rules))
save_rules(class_association_rules, rules_directory + 'class_association_rules.pkl')

ensemble_rules = decision_tree_rules + exkmc_rules + shallow_rules + forest_rules + class_association_rules
ensemble_rules = filter_rules(
    ensemble_rules, data, kmeans_labels, confidence = fixed_parameters['min_confidence']
)

print("Total ensemble rules after filtering:", len(ensemble_rules))
save_rules(ensemble_rules, rules_directory + 'ensemble_rules.pkl')


####################################################################################################
# Objectives for Decision Set Clustering:

objective_dict = {
    'coverage-mistake': {
        'objective_type': 'coverage-mistake'
    },
    'total-coverage-mistake': {
        'objective_type': 'total-coverage-mistake'
    },
    'coverage-cost': {
        'cluster_centers': kmeans_base.centers,
        'objective_type': 'coverage-cost',
        'cluster_cost_method': 'kmeans'
    },
    'total-coverage-cost': {
        'cluster_centers': kmeans_base.centers,
        'objective_type': 'total-coverage-cost',
        'cluster_cost_method': 'kmeans'
    },
    'coverage-pairwise-distance': {
        'objective_type': 'coverage-pairwise-distance',
    },
    'total-coverage-pairwise-distance': {
        'objective_type': 'total-coverage-pairwise-distance',
    },
}


####################################################################################################
# Save decision info dict for ensemble rules:

for objective_type in objective_dict.keys():
    print("Processing objective:", objective_type)
    dsclust = DSCluster(
        rules = ensemble_rules,
        n_select = fixed_parameters['n_clusters'],
        alpha_val = 0.0,
        lambda_val = 0.0,
        **objective_dict[objective_type]
    )
    dsclust.fit(data, kmeans_labels)
    decision_info_dict = dsclust.objective.decision_info_dict

    # Save decision info dict
    save_path = rules_directory + f'decision_info_dict_{objective_type.replace("-", "_")}.pkl'
    dsclust.objective.save_decision_info_dict(save_path)