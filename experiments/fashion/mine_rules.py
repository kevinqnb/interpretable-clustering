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
from experiments.cli_utils import conf_tag

####################################################################################################

import os
import argparse
import pickle
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
# Rule mining here (TreeMiner/RandomForestMiner/ClassAssociationRuleMiner) is a
# one-time, cached step -- like alphas.py's alpha selection -- so it is run once
# under this single seed rather than repeated across trials. See
# experiments/README.md ("Reproducibility") for which downstream models (IDS,
# ExplanationTree, DecisionTree, ShallowTree) are instead re-fit across multiple
# trial seeds in max_rules.py/confidence.py, since their fitted solution has
# inherent randomness.
seed = 342

####################################################################################################
# Parse confidence thresholds to filter the mined ensemble at. Each threshold
# gets its own tagged, saved rule pool (see conf_tag) so downstream scripts
# can each be pointed at a specific threshold via their own --confidence flag.

parser = argparse.ArgumentParser()
parser.add_argument(
    '--confidence-thresholds', type=float, nargs='+', default=[0.25, 0.5, 0.75]
)
args = parser.parse_args()
confidence_thresholds = args.confidence_thresholds

####################################################################################################
# Read and process data:
data, data_labels, feature_labels, scaler = load_preprocessed_fashion()
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
    'confidence_thresholds': confidence_thresholds,
    'seed': seed
}

np.random.seed(fixed_parameters['seed'])

# Do baseline clustering
kmeans_base = KMeansBase(n_clusters = fixed_parameters['n_clusters'], random_seed = fixed_parameters['seed'])
kmeans_assignment = kmeans_base.assign(data)
kmeans_labels = kmeans_base.labels

# Weights for uncertainty objectives
weights = distance_ratio_score(data, kmeans_base.centers)
fixed_parameters['weights'] = weights.tolist()

rules_directory = 'data/experiments/fashion/rules/'
os.makedirs(rules_directory, exist_ok = True)

####################################################################################################
# Create bin_df for rule mining:

bin_df_path = rules_directory + 'bin_df.csv'
if os.path.exists(bin_df_path):
    bin_df = pd.read_csv(bin_df_path)
else:
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
        depth_factor = fixed_parameters['shallow_tree_depth_factor'],
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
        'max_depth': fixed_parameters['forest_max_depth'],
        'random_state': fixed_parameters['seed']
    }
)
forest_rules, forest_rule_labels = forest_rule_miner.fit(data, kmeans_base.labels)

print("Mined Forest rules:", len(forest_rules))
save_rules(forest_rules, rules_directory + 'forest_rules.pkl')


class_association_rule_miner = ClassAssociationRuleMiner(
    min_support = fixed_parameters['car_min_support'],
    min_confidence = fixed_parameters['car_min_confidence'],
    max_length = fixed_parameters['car_max_rule_length'],
    bin_df = bin_df
)
class_association_rules, class_association_rule_labels = class_association_rule_miner.fit(
    X = data, y = kmeans_base.labels
)

print("Mined CAR rules:", len(class_association_rules))
save_rules(class_association_rules, rules_directory + 'class_association_rules.pkl')

pre_filter_ensemble = decision_tree_rules + shallow_rules + forest_rules + class_association_rules

'''
# Load if pre-computed
decision_tree_rules = load_rules('data/experiments/fashion/rules/decision_tree_rules.pkl')
exkmc_rules = load_rules('data/experiments/fashion/rules/exkmc_rules.pkl')
shallow_rules = load_rules('data/experiments/fashion/rules/shallow_rules.pkl')
forest_rules = load_rules('data/experiments/fashion/rules/forest_rules.pkl')
class_association_rules = load_rules('data/experiments/fashion/rules/class_association_rules.pkl')

pre_filter_ensemble = decision_tree_rules + shallow_rules + forest_rules + class_association_rules

'''

print("Total pre-filter ensemble rules:", len(pre_filter_ensemble))
save_rules(pre_filter_ensemble, rules_directory + 'pre_filter_ensemble_rules.pkl')

####################################################################################################
# Compute and save majority-class rule labels for the pre-filter pool.
#
# Each rule is assigned the cluster label that appears most often among the
# data points it covers.  The format is List[Set[int]] to match the
# DecisionSet.rule_labels convention.

def _majority_labels(rules, X, y_flat, n_clusters):
    labels = []
    for rule in rules:
        mask = rule.evaluate(X)
        if mask.sum() == 0:
            labels.append({0})
        else:
            labels.append({int(np.bincount(y_flat[mask], minlength=n_clusters).argmax())})
    return labels

y_flat = flatten_labels(kmeans_labels)
n_clusters = fixed_parameters['n_clusters']

pre_filter_labels = _majority_labels(pre_filter_ensemble, data, y_flat, n_clusters)
with open(rules_directory + 'pre_filter_ensemble_labels.pkl', 'wb') as f:
    pickle.dump(pre_filter_labels, f)
print(f"Pre-filter ensemble labels saved ({len(pre_filter_labels)} rules).")

####################################################################################################
# Filter the pre-filter pool at each confidence threshold and save a tagged
# rule pool + labels + per-objective decision-info cache for each. The cached
# rule_coverage/decision_info arrays PEC persists are positional to the exact
# rule pool passed to it, so each threshold's (differently-sized) filtered
# pool needs its own cache -- these can't be shared across thresholds.

for confidence in confidence_thresholds:
    tag = conf_tag(confidence)
    print(f"\n=== confidence threshold {confidence} (tag={tag}) ===")

    ensemble_rules = filter_rules(
        pre_filter_ensemble, data, kmeans_labels, confidence = confidence
    )
    print("Total ensemble rules after filtering:", len(ensemble_rules))
    save_rules(ensemble_rules, rules_directory + f'ensemble_rules_conf_{tag}.pkl')

    ensemble_labels = _majority_labels(ensemble_rules, data, y_flat, n_clusters)
    with open(rules_directory + f'ensemble_labels_conf_{tag}.pkl', 'wb') as f:
        pickle.dump(ensemble_labels, f)
    print(f"Ensemble labels saved ({len(ensemble_labels)} rules).")

    objective_dict = {
        'coverage-mistake': {
            'objective_type': 'coverage-mistake',
            'n_select': fixed_parameters['n_clusters'],
            'alpha_val': 0.0,
            'lambda_val': 0.0,
            'output_path': rules_directory + f"mistake_info_dict_conf_{tag}.pkl.gz"
        },
        'coverage-cost': {
            'objective_type': 'coverage-cost',
            'cluster_centers': kmeans_base.centers,
            'cluster_cost_method': 'kmeans',
            'n_select': fixed_parameters['n_clusters'],
            'alpha_val': 0.0,
            'lambda_val': 0.0,
            'output_path': rules_directory + f"cost_info_dict_conf_{tag}.pkl.gz"
        },
        'coverage-pairwise-distance': {
            'objective_type': 'coverage-pairwise-distance',
            'n_select': fixed_parameters['n_clusters'],
            'alpha_val': 0.0,
            'lambda_val': 0.0,
            'output_path': rules_directory + f"pairwise_distance_info_dict_conf_{tag}.pkl.gz"
        },
    }

    for objective_type in objective_dict.keys():
        print("Processing objective:", objective_type)
        pec = PEC(
            rules = ensemble_rules,
            **objective_dict[objective_type]
        )
        pec.fit(data, kmeans_labels)
