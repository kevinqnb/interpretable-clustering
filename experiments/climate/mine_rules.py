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
from experiments.climate.config import (
    SEED, N_CLUSTERS, N_SELECT_DEFAULT, MAX_RULES, SHALLOW_TREE_DEPTH_FACTOR,
    N_FOREST, FOREST_MAX_DEPTH, CAR_MIN_SUPPORT, CAR_MIN_CONFIDENCE,
    CAR_MAX_RULE_LENGTH, CONFIDENCE_DEFAULT, OUTFILE_REF, RULES_DIR,
)

####################################################################################################

import os
import json
import pickle
import numpy as np
import pandas as pd
from intercluster import *
from intercluster.decision_trees import *
from intercluster.decision_sets import *
from intercluster.decision_sets.objectives import *
from intercluster.decision_sets.mining import *
from intercluster.measurements import *
from intercluster.rules import save_rules, load_rules

# Prevents memory leakage for KMeans:
os.environ["OMP_NUM_THREADS"] = "1"

# REMINDER: Initialize the seed only here, not inside any sub-function or
# class (except select baseline experiments like KMeans) -- passing a seed
# there resets it on every call.
# Rule mining here (TreeMiner/RandomForestMiner/ClassAssociationRuleMiner) is a
# one-time, cached step -- like alphas.py's alpha selection -- so it runs once
# under this single seed rather than repeated across trials. See
# experiments/README.md ("Reproducibility") for which downstream models (IDS,
# DecisionTree) are instead re-fit across multiple trial seeds in
# max_rules.py/confidence.py, since their fitted solution has inherent
# randomness.
seed = SEED

####################################################################################################
# Read and process data:
data, data_labels, feature_labels, scaler = load_preprocessed_climate("data/climate")
n,d = data.shape

confidence = CONFIDENCE_DEFAULT

fixed_parameters = {
    'n' : n,
    'd' : d,
    'n_clusters': N_CLUSTERS,
    'n_select': N_SELECT_DEFAULT,
    'max_rules': MAX_RULES,
    'shallow_tree_depth_factor': SHALLOW_TREE_DEPTH_FACTOR,
    'n_forest': N_FOREST,
    'forest_max_depth': FOREST_MAX_DEPTH,
    'car_min_support': CAR_MIN_SUPPORT,
    'car_min_confidence': CAR_MIN_CONFIDENCE,
    'car_max_rule_length': CAR_MAX_RULE_LENGTH, # (really means 6 by pyfim convention)
    'filter_confidence': confidence,
    'seed': seed
}

np.random.seed(fixed_parameters['seed'])

kmeans_base = KMeansBase(n_clusters = fixed_parameters['n_clusters'], random_seed = fixed_parameters['seed'])
kmeans_assignment = kmeans_base.assign(data)
kmeans_labels = kmeans_base.labels

# Weights for uncertainty objectives
weights = distance_ratio_score(data, kmeans_base.centers)
fixed_parameters['weights'] = weights.tolist()

rules_directory = RULES_DIR
os.makedirs(rules_directory, exist_ok = True)

####################################################################################################
# Cache helpers for the pre-filtration mining stage below.
#
# Every cached artifact (bin_df, each miner's rule pool) is accompanied by a
# `.params.json` sidecar recording the parameters that produced it. On each
# run, the sidecar is compared against the current parameters: a match loads
# the cache, a mismatch (or a missing/deleted cache) re-derives it and
# overwrites both files. This makes the cache self-invalidating when e.g.
# `car_min_support` changes, instead of relying on manually deleting stale
# `.pkl`/`.csv` files.

def _cache_params_match(params_path, params):
    if not os.path.exists(params_path):
        return False
    with open(params_path, 'r') as f:
        cached_params = json.load(f)
    return cached_params == params

def _save_cache_params(params_path, params):
    with open(params_path, 'w') as f:
        json.dump(params, f)

def _load_rules_if_cached(rules_path, params_path, params):
    if os.path.exists(rules_path) and _cache_params_match(params_path, params):
        return load_rules(rules_path)
    return None

base_cache_params = {
    'n_clusters': fixed_parameters['n_clusters'],
    'seed': fixed_parameters['seed'],
}

####################################################################################################
# Create bin_df for rule mining:

bin_df_path = rules_directory + 'bin_df.csv'
bin_df_params_path = rules_directory + 'bin_df.params.json'
bin_df_cache_params = dict(base_cache_params)
if os.path.exists(bin_df_path) and _cache_params_match(bin_df_params_path, bin_df_cache_params):
    bin_df = pd.read_csv(bin_df_path)
else:
    bin_df = entropy_bin(
        data, kmeans_labels, random_state = fixed_parameters['seed']
    )
    bin_df.to_csv(bin_df_path, index = False)
    _save_cache_params(bin_df_params_path, bin_df_cache_params)

####################################################################################################
# Mine for rules. Each rule pool is cached to disk, keyed to the parameters
# that produced it (see cache helpers above): if a cache from a previous run
# is still valid, it is loaded rather than re-mined, so that rule sets stay
# identical across runs regardless of any residual non-determinism in the
# underlying miners. Delete the corresponding .pkl/.params.json files (or
# just change a relevant parameter) to force a fresh mine.
#
# NOTE: Shallow-tree rule mining is disabled -- it is no longer part of the
# pre-filter ensemble.

decision_tree_rules_path = rules_directory + 'decision_tree_rules.pkl'
decision_tree_params_path = rules_directory + 'decision_tree_rules.params.json'
decision_tree_cache_params = dict(base_cache_params)
decision_tree_rules = _load_rules_if_cached(
    decision_tree_rules_path, decision_tree_params_path, decision_tree_cache_params
)
if decision_tree_rules is not None:
    print("Loaded cached DT rules:", len(decision_tree_rules))
else:
    decision_tree_rule_miner = TreeMiner(
        tree = DecisionTree(random_state = fixed_parameters['seed']),
    )
    decision_tree_rules, _ = decision_tree_rule_miner.fit(
        X = data, y = kmeans_base.labels
    )
    print("Mined DT rules:", len(decision_tree_rules))
    save_rules(decision_tree_rules, decision_tree_rules_path)
    _save_cache_params(decision_tree_params_path, decision_tree_cache_params)

exkmc_rules_path = rules_directory + 'exkmc_rules.pkl'
exkmc_params_path = rules_directory + 'exkmc_rules.params.json'
exkmc_cache_params = dict(base_cache_params, max_rules = fixed_parameters['max_rules'])
exkmc_rules = _load_rules_if_cached(exkmc_rules_path, exkmc_params_path, exkmc_cache_params)
if exkmc_rules is not None:
    print("Loaded cached ExKMC rules:", len(exkmc_rules))
else:
    exkmc_rule_miner = TreeMiner(
        tree = ExkmcTree(
            k = fixed_parameters['n_clusters'],
            max_leaf_nodes = fixed_parameters['max_rules'],
            kmeans = kmeans_base.clustering,
            imm = True
        )
    )
    exkmc_rules, _ = exkmc_rule_miner.fit(
        X = data, y = kmeans_base.labels
    )
    print("Mined ExKMC rules:", len(exkmc_rules))
    save_rules(exkmc_rules, exkmc_rules_path)
    _save_cache_params(exkmc_params_path, exkmc_cache_params)

# shallow_rules_path = rules_directory + 'shallow_rules.pkl'
# shallow_params_path = rules_directory + 'shallow_rules.params.json'
# shallow_cache_params = dict(
#     base_cache_params, shallow_tree_depth_factor = fixed_parameters['shallow_tree_depth_factor']
# )
# shallow_rules = _load_rules_if_cached(shallow_rules_path, shallow_params_path, shallow_cache_params)
# if shallow_rules is not None:
#     print("Loaded cached Shallow rules:", len(shallow_rules))
# else:
#     shallow_tree_miner = TreeMiner(
#         tree = ShallowTree(
#             n_clusters = fixed_parameters['n_clusters'],
#             depth_factor = fixed_parameters['shallow_tree_depth_factor'],
#             kmeans_random_state = fixed_parameters['seed']
#         )
#     )
#     shallow_rules, _ = shallow_tree_miner.fit(
#         X = data, y = kmeans_labels
#     )
#     print("Mined Shallow rules:", len(shallow_rules))
#     save_rules(shallow_rules, shallow_rules_path)
#     _save_cache_params(shallow_params_path, shallow_cache_params)

forest_rules_path = rules_directory + 'forest_rules.pkl'
forest_params_path = rules_directory + 'forest_rules.params.json'
forest_cache_params = dict(
    base_cache_params,
    n_forest = fixed_parameters['n_forest'],
    forest_max_depth = fixed_parameters['forest_max_depth'],
)
forest_rules = _load_rules_if_cached(forest_rules_path, forest_params_path, forest_cache_params)
if forest_rules is not None:
    print("Loaded cached Forest rules:", len(forest_rules))
else:
    forest_rule_miner = RandomForestMiner(
        forest_params = {
            'n_estimators': fixed_parameters['n_forest'],
            'max_depth': fixed_parameters['forest_max_depth'],
            'random_state': fixed_parameters['seed']
        }
    )
    forest_rules, _ = forest_rule_miner.fit(data, kmeans_base.labels)
    print("Mined Forest rules:", len(forest_rules))
    save_rules(forest_rules, forest_rules_path)
    _save_cache_params(forest_params_path, forest_cache_params)

class_association_rules_path = rules_directory + 'class_association_rules.pkl'
class_association_params_path = rules_directory + 'class_association_rules.params.json'
class_association_cache_params = dict(
    base_cache_params,
    car_min_support = fixed_parameters['car_min_support'],
    car_min_confidence = fixed_parameters['car_min_confidence'],
    car_max_rule_length = fixed_parameters['car_max_rule_length'],
)
class_association_rules = _load_rules_if_cached(
    class_association_rules_path, class_association_params_path, class_association_cache_params
)
if class_association_rules is not None:
    print("Loaded cached CAR rules:", len(class_association_rules))
else:
    class_association_rule_miner = ClassAssociationRuleMiner(
        min_support = fixed_parameters['car_min_support'],
        min_confidence = fixed_parameters['car_min_confidence'],
        max_length = fixed_parameters['car_max_rule_length'],
        bin_df = bin_df
    )
    class_association_rules, _ = class_association_rule_miner.fit(
        X = data, y = kmeans_base.labels
    )
    print("Mined CAR rules:", len(class_association_rules))
    save_rules(class_association_rules, class_association_rules_path)
    _save_cache_params(class_association_params_path, class_association_cache_params)

####################################################################################################
# Tag each rule with the miner that produced it (decision tree / random forest / CAR), so that
# selection results downstream (PEC/CBA/IDS) can report a provenance breakdown of the rules they
# selected. `source` is not part of `Rule`'s identity (see `Rule.__hash__`/`__eq__` in rules.py),
# so this is purely additive metadata -- it does not change rule content, order, or any
# content-keyed dict/set behavior downstream.

def _tag_source(rules, source):
    return [Rule(list(r.conditions), source=source) for r in rules]

decision_tree_rules = _tag_source(decision_tree_rules, 'decision_tree')
forest_rules = _tag_source(forest_rules, 'random_forest')
class_association_rules = _tag_source(class_association_rules, 'car')

pre_filter_ensemble = decision_tree_rules + forest_rules + class_association_rules
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
# Filter the pre-filter pool at the default confidence threshold and save the
# tagged rule pool + labels + per-objective decision-info cache. These are
# positional to the exact rule pool passed to PEC, so they must be rebuilt
# whenever the confidence threshold (or the pre-filter pool) changes.

print(f"\n=== filtering at confidence threshold {confidence} ===")

ensemble_rules = filter_rules(
    pre_filter_ensemble, data, kmeans_labels, confidence = confidence
)
print("Total ensemble rules after filtering:", len(ensemble_rules))
save_rules(ensemble_rules, rules_directory + f'ensemble_rules{OUTFILE_REF}.pkl')

ensemble_labels = _majority_labels(ensemble_rules, data, y_flat, n_clusters)
with open(rules_directory + f'ensemble_labels{OUTFILE_REF}.pkl', 'wb') as f:
    pickle.dump(ensemble_labels, f)
print(f"Ensemble labels saved ({len(ensemble_labels)} rules).")

objective_dict = {
    'coverage-mistake': {
        'objective_type': 'coverage-mistake',
        'n_select': fixed_parameters['n_clusters'],
        'alpha_val': 0.0,
        'lambda_val': 0.0,
        'output_path': rules_directory + f"mistake_info_dict{OUTFILE_REF}.pkl.gz"
    },
    'coverage-cost': {
        'objective_type': 'coverage-cost',
        'cluster_centers': kmeans_base.centers,
        'cluster_cost_method': 'kmeans',
        'n_select': fixed_parameters['n_clusters'],
        'alpha_val': 0.0,
        'lambda_val': 0.0,
        'output_path': rules_directory + f"cost_info_dict{OUTFILE_REF}.pkl.gz"
    },
    'coverage-pairwise-distance': {
        'objective_type': 'coverage-pairwise-distance',
        'n_select': fixed_parameters['n_clusters'],
        'alpha_val': 0.0,
        'lambda_val': 0.0,
        'output_path': rules_directory + f"pairwise_distance_info_dict{OUTFILE_REF}.pkl.gz"
    },
}

for objective_type in objective_dict.keys():
    print("Processing objective:", objective_type)
    dsclust = PEC(
        rules = ensemble_rules,
        **objective_dict[objective_type]
    )
    dsclust.fit(data, kmeans_labels)
