####################################################################################################
# Path setup

import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
PROJECT_ROOT = next((p for p in _HERE.parents if (p / "data").is_dir()), None)
if PROJECT_ROOT is None:
    raise ModuleNotFoundError("Could not locate repository root.")
sys.path.insert(0, str(PROJECT_ROOT))

from data.preprocessing import *
from experiments.modules import *

####################################################################################################

import os
import json
import math
import time
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
from intercluster import *
from intercluster.decision_trees import *
from intercluster.decision_sets import *
from intercluster.decision_sets.objectives import *
from intercluster.decision_sets.mining import *
from intercluster.measurements import *

os.environ["OMP_NUM_THREADS"] = "1"

seed = 342

def _memoryview_safe(x):
    if not x.flags.writeable:
        if not x.flags.owndata:
            x = x.copy(order='C')
        x.setflags(write=True)
    return x

####################################################################################################
# Data + clustering

data, data_labels, feature_labels, scaler = load_preprocessed_climate('data/climate')
data = _memoryview_safe(data)
n, d = data.shape

fixed_parameters = {
    'n': n,
    'd': d,
    'n_clusters': 6,
    'n_select': 6,
    'shallow_tree_depth_factor': 0.03,
    'n_forest': 100,
    'forest_max_depth': 6,
    'car_min_support': 0.025,
    'car_min_confidence': 0.85,
    'car_max_rule_length': 3,
    'seed': seed,
    'confidence_values': list(np.round(np.arange(0.0, 1.0, 0.05), 2).tolist()),
}

n_select = fixed_parameters['n_select']
n_labels = fixed_parameters['n_clusters']

np.random.seed(seed)

kmeans_base = KMeansBase(n_clusters=n_labels, random_seed=seed)
kmeans_assignment = kmeans_base.assign(data)
kmeans_labels = kmeans_base.labels

weights = distance_ratio_score(data, kmeans_base.centers)
fixed_parameters['weights'] = weights.tolist()

####################################################################################################
# Load alpha values (fixed from alphas.py selection)

with open("data/experiments/climate/alphas/selected_alphas_rule_length.json") as f:
    selected_alpha_dict = json.load(f)

####################################################################################################
# Load rule pools and IDS lambdas

pre_filter_ensemble = load_rules('data/experiments/climate/rules/pre_filter_ensemble_rules.pkl')

with open('data/experiments/climate/rules/ids_lambdas.json') as f:
    ids_lambdas = json.load(f)
if isinstance(ids_lambdas, dict):
    ids_lambdas = list(ids_lambdas.values())

outfile = 'data/experiments/climate/confidence/'
os.makedirs(outfile, exist_ok=True)

####################################################################################################
# Objective configuration
# No precomputed_paths: the rule pool changes at each confidence level, so everything
# must be computed from scratch each iteration.

objective_config = {
    'coverage-mistake': {
        'objective_type': 'coverage-mistake',
    },
    'coverage-cost': {
        'objective_type': 'coverage-cost',
        'cluster_centers': kmeans_base.centers,
        'cluster_cost_method': 'kmeans',
    },
    'coverage-pairwise-distance': {
        'objective_type': 'coverage-pairwise-distance',
    },
    'coverage-mistake-weighted': {
        'objective_type': 'coverage-mistake',
        'weights': weights,
    },
    'coverage-cost-weighted': {
        'objective_type': 'coverage-cost',
        'cluster_centers': kmeans_base.centers,
        'cluster_cost_method': 'kmeans',
        'weights': weights,
    },
    'coverage-pairwise-distance-weighted': {
        'objective_type': 'coverage-pairwise-distance',
        'weights': weights,
    },
}

####################################################################################################
# Helpers

def _make_objective(objective_type, n_select, alpha_val, lambda_val=None,
                    cluster_centers=None, cluster_cost_method='kmeans', weights=None):
    """Instantiate the right Objective subclass without precomputed data."""
    common = dict(n_select=n_select, alpha_val=alpha_val, lambda_val=lambda_val, weights=weights)
    if objective_type == 'coverage-mistake':
        return CoverageMistakeObjective(**common)
    elif objective_type == 'coverage-cost':
        return CoverageCostObjective(
            cluster_centers=cluster_centers,
            cluster_cost_method=cluster_cost_method,
            **common,
        )
    elif objective_type == 'coverage-pairwise-distance':
        return CoveragePairwiseDistanceObjective(**common)
    else:
        raise ValueError(f'Unknown objective type: {objective_type}')


def score_decision_set(decisions, X, y, n_select, objective_type, alpha_val, lambda_val,
                       cluster_centers=None, cluster_cost_method='kmeans', weights=None):
    """
    Evaluate the PEC objective on a fixed decision set using a forced lambda.
    n_select must match the budget used when lambda was computed (for cost normalization).
    The full decision set is scored as-is (no greedy selection step).
    """
    if not decisions:
        return 0.0
    dset = set(decisions)
    obj = _make_objective(
        objective_type,
        n_select=n_select,
        alpha_val=alpha_val,
        lambda_val=None,  # set explicitly below after initialization
        cluster_centers=cluster_centers,
        cluster_cost_method=cluster_cost_method,
        weights=weights,
    )
    obj.initialize_data(X, y)
    obj.initialize_decision_set(dset)
    obj.set_lambda(lambda_val)
    return float(obj.compute_objective(obj.decision_info_dict))


def tree_to_decisions(tree):
    """Extract Decision objects from a fitted tree (one per leaf, in traversal order)."""
    leaf_rules = collect_leaf_rules(tree.root)
    leaf_labels = tree.get_leaf_labels()
    return [Decision(r, next(iter(lbl))) for r, lbl in zip(leaf_rules, leaf_labels)]


def _tree_info(tree, decisions, data, n_labels):
    return {
        'decisions': decisions,
        'data_to_rule': tree.get_data_to_rules_assignment(data),
        'rule_to_cluster': labels_to_assignment(tree.get_leaf_labels(), n_labels=n_labels),
        'data_to_cluster': labels_to_assignment(tree.predict(data), n_labels=n_labels),
        'n-rules': tree.leaf_count,
        'max-rule-length': tree.depth,
        'sum-rule-length': tree.get_sum_of_depths(),
        'weighted-avg-length': tree.get_weighted_average_depth(data),
    }


def _dset_info(model, data, n_labels):
    dset = model.decision_set  # list after fit()
    return {
        'decisions': dset,
        'data_to_rule': model.get_data_to_rules_assignment(data),
        'rule_to_cluster': model.get_rules_to_clusters_assignment(n_labels=n_labels),
        'data_to_cluster': labels_to_assignment(model.predict(data), n_labels=n_labels),
        'n-rules': len(dset),
        'max-rule-length': model.max_rule_length,
        'sum-rule-length': model.get_sum_of_rule_lengths(),
        'weighted-avg-length': model.get_weighted_average_rule_length(data),
    }


def _empty_info():
    return {
        'decisions': [],
        'data_to_rule': None,
        'rule_to_cluster': None,
        'data_to_cluster': None,
        'n-rules': 0,
        'max-rule-length': np.nan,
        'sum-rule-length': np.nan,
        'weighted-avg-length': np.nan,
    }


def measure_algo(data_to_rule, rule_to_cluster, data_to_cluster, measurement_fns):
    return {fn.name: fn(data_to_rule, rule_to_cluster, data_to_cluster) for fn in measurement_fns}


####################################################################################################
# Measurement functions (identical set to max_rules.py)

measurement_fns = [
    TotalCoverage(),
    TotalCoverage(weights=weights, name='total-coverage-weighted'),
    TotalCoverageSet(),
    ClusterCoverage(baseline_assignment=kmeans_assignment),
    ClusterCoverage(
        baseline_assignment=kmeans_assignment,
        weights=weights,
        name='cluster-coverage-weighted',
    ),
    ClusterCoverageSet(baseline_assignment=kmeans_assignment),
    Overlap(),
    Mistakes(baseline_assignment=kmeans_assignment),
    ClusteringCost(data=data, average=True, normalize=True, method="kmeans"),
    RuleClusteringCost(data=data, cluster_centers=kmeans_base.centers, method="kmeans"),
    RulePairwiseDistance(baseline_assignment=kmeans_assignment),
]

####################################################################################################
# Pool-independent algorithms — run once, standard measurements precomputed

print("Fitting pool-independent algorithms...")

_dt = DecisionTree(max_leaf_nodes=n_select, random_state=seed)
_dt.fit(data, kmeans_labels)
dt_decisions = tree_to_decisions(_dt)

_exp_tree = ExplanationTree(num_clusters=n_labels)
_exp_tree.fit(data, kmeans_labels)
exp_tree_decisions = tree_to_decisions(_exp_tree)

_exkmc = ExkmcTree(k=n_labels, kmeans=kmeans_base.clustering, max_leaf_nodes=n_select)
_exkmc.fit(data, kmeans_labels)
exkmc_decisions = tree_to_decisions(_exkmc)

_shallow = ShallowTree(
    n_clusters=n_labels,
    depth_factor=fixed_parameters['shallow_tree_depth_factor'],
    kmeans_random_state=seed,
)
_shallow.fit(data, kmeans_labels)
shallow_decisions = tree_to_decisions(_shallow)

_cn2 = CN2(n_select=n_select)
_cn2.fit(data, kmeans_labels)

pool_indep = {
    'Decision-Tree': _tree_info(_dt, dt_decisions, data, n_labels),
    'Exp-Tree': _tree_info(_exp_tree, exp_tree_decisions, data, n_labels),
    'ExKMC': _tree_info(_exkmc, exkmc_decisions, data, n_labels),
    'Shallow-Tree': _tree_info(_shallow, shallow_decisions, data, n_labels),
    'CN2': _dset_info(_cn2, data, n_labels),
}

# Precompute standard measurements (these do not change across confidence levels)
pool_indep_measurements = {
    name: measure_algo(
        info['data_to_rule'], info['rule_to_cluster'], info['data_to_cluster'], measurement_fns
    )
    for name, info in pool_indep.items()
}

# KMeans baseline measurements (fixed)
baseline_measurements = measure_algo(None, None, kmeans_assignment, measurement_fns)

print("Pool-independent algorithms ready.")

####################################################################################################
# IDS full-pool cache — built once on all CAR rules, then subset per confidence level

_ids_cache_path = 'data/experiments/climate/rules/ids_coverage_cache.pkl'
if os.path.exists(_ids_cache_path):
    print("Loading pre-built IDS cache...")
    with open(_ids_cache_path, 'rb') as f:
        ids_full_cache = pickle.load(f)
    print(f"IDS cache loaded ({len(ids_full_cache.decisions)} decisions).")
else:
    print("Pre-computing IDS coverage cache on full pre-filter ensemble...")
    _ids_full = IDS(rules=pre_filter_ensemble, n_select=n_select, lambdas=ids_lambdas, optimizer='random_greedy')
    _ids_full.fit(data, kmeans_labels)
    ids_full_cache = _ids_full.get_cache()
    print(f"IDS cache ready: {len(ids_full_cache.decisions)} decisions.")

####################################################################################################
# Main confidence sweep

confidence_values = np.round(np.arange(0.0, 1.0, 0.05), 2)

result = {'fixed-parameters': fixed_parameters}

for conf in confidence_values:
    conf_key = float(conf)
    t0 = time.time()
    print(f"\n[confidence={conf_key:.2f}] filtering rules...")

    filtered_rules = filter_rules(pre_filter_ensemble, data, kmeans_labels, confidence=conf)
    has_rules = len(filtered_rules) > 0

    print(f"  {len(filtered_rules)} ensemble rules")

    conf_result = {
        'n_filtered_rules': len(filtered_rules),
        'lambda': {},
    }

    # ----------------------------------------------------------------
    # PEC — one fit per objective type; captures the lambda used
    # ----------------------------------------------------------------
    pec_lambdas = {}
    pec_info = {}

    for obj_name, obj_cfg in objective_config.items():
        pec_name = f'dscluster; {obj_name}; ensemble'
        alpha = selected_alpha_dict[pec_name]
        if has_rules:
            pec = PEC(rules=filtered_rules, n_select=n_select, alpha_val=alpha, **obj_cfg)
            pec.fit(data, kmeans_labels)
            pec_lambdas[obj_name] = pec.objective.lambda_val
            pec_info[pec_name] = _dset_info(pec, data, n_labels)
        else:
            pec_lambdas[obj_name] = np.nan
            pec_info[pec_name] = _empty_info()

    conf_result['lambda'] = pec_lambdas

    # ----------------------------------------------------------------
    # WRA, WRA-weighted, CBA (pool-dependent)
    # ----------------------------------------------------------------
    pool_dep = {}

    if has_rules:
        _wra = WRABaseline(rules=filtered_rules, n_select=n_select)
        _wra.fit(data, kmeans_labels)
        pool_dep['WRA'] = _dset_info(_wra, data, n_labels)

        _wra_w = WRABaseline(rules=filtered_rules, n_select=n_select, weights=weights)
        _wra_w.fit(data, kmeans_labels)
        pool_dep['WRA-weighted'] = _dset_info(_wra_w, data, n_labels)

        _cba = CBA(rules=filtered_rules, n_select=n_select)
        _cba.fit(data, kmeans_labels)
        pool_dep['CBA'] = _dset_info(_cba, data, n_labels)
    else:
        pool_dep['WRA'] = _empty_info()
        pool_dep['WRA-weighted'] = _empty_info()
        pool_dep['CBA'] = _empty_info()

    # ----------------------------------------------------------------
    # IDS — subset the full cache to filtered ensemble rules each iteration;
    # lambdas are fixed (fitted via coordinate ascent on the full pool).
    # ----------------------------------------------------------------
    if filtered_rules:
        filtered_rule_set = set(filtered_rules)
        filtered_indices = [
            i for i, d in enumerate(ids_full_cache.decisions)
            if d.rule in filtered_rule_set
        ]
        ids_sub_cache = ids_full_cache.subset(filtered_indices)
        _ids = IDS(
            rules=filtered_rules,
            n_select=n_select,
            lambdas=ids_lambdas,
            cache=ids_sub_cache,
            optimizer='random_greedy',
        )
        _ids.fit(data, kmeans_labels)
        pool_dep['IDS'] = _dset_info(_ids, data, n_labels)
    else:
        pool_dep['IDS'] = _empty_info()
    # ----------------------------------------------------------------
    # Aggregate all algorithm info for evaluation
    # ----------------------------------------------------------------

    # KMeans baseline
    conf_result['KMeans'] = {
        'n-rules': np.nan,
        'max-rule-length': np.nan,
        'sum-rule-length': np.nan,
        'weighted-avg-length': np.nan,
    }
    conf_result['KMeans'].update(baseline_measurements)
    # No objective score for KMeans (no decision set)

    # All other algorithms: pool-independent + pool-dependent + PEC variants
    all_algo_info = {
        **{name: (info, pool_indep_measurements[name]) for name, info in pool_indep.items()},
        **{name: (info, None) for name, info in pool_dep.items()},
        **{name: (info, None) for name, info in pec_info.items()},
    }

    for algo_name, (info, precomputed_meas) in all_algo_info.items():
        algo_result = {
            'n-rules': info['n-rules'],
            'max-rule-length': info['max-rule-length'],
            'sum-rule-length': info['sum-rule-length'],
            'weighted-avg-length': info['weighted-avg-length'],
        }

        # Standard measurements
        if precomputed_meas is not None:
            algo_result.update(precomputed_meas)
        else:
            algo_result.update(
                measure_algo(
                    info['data_to_rule'],
                    info['rule_to_cluster'],
                    info['data_to_cluster'],
                    measurement_fns,
                )
            )

        # PEC objective score for each of the 6 objective types
        algo_result['objective'] = {}
        for obj_name, obj_cfg in objective_config.items():
            lambda_c = pec_lambdas[obj_name]
            alpha = selected_alpha_dict[f'dscluster; {obj_name}; ensemble']
            decisions = info['decisions']
            if np.isnan(lambda_c) or not decisions:
                algo_result['objective'][obj_name] = np.nan
            else:
                algo_result['objective'][obj_name] = score_decision_set(
                    decisions,
                    data,
                    kmeans_labels,
                    n_select=n_select,
                    objective_type=obj_cfg['objective_type'],
                    alpha_val=alpha,
                    lambda_val=lambda_c,
                    cluster_centers=obj_cfg.get('cluster_centers'),
                    cluster_cost_method=obj_cfg.get('cluster_cost_method', 'kmeans'),
                    weights=obj_cfg.get('weights'),
                )

        conf_result[algo_name] = algo_result

    result[conf_key] = conf_result
    print(f"  confidence={conf_key:.2f} done in {time.time() - t0:.1f}s")

####################################################################################################
# Save results

import math

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return None if math.isnan(obj) else float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

fname = os.path.join(outfile, 'exp_confidence.json')
with open(fname, 'w') as f:
    json.dump(result, f, indent=4, cls=NumpyEncoder)

print(f"\nResults saved to {fname}")

####################################################################################################
