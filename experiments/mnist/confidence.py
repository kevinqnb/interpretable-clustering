####################################################################################################

import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
PROJECT_ROOT = next((p for p in _HERE.parents if (p / "data").is_dir()), None)
if PROJECT_ROOT is None:
    raise ModuleNotFoundError("Could not locate repository root.")
sys.path.insert(0, str(PROJECT_ROOT))

from data.preprocessing import *
from experiments.modules import *
from experiments.profiling import stamp, stamp_reset
from experiments.mnist.config import (
    SEED, N_CLUSTERS, N_SELECT_DEFAULT, SHALLOW_TREE_DEPTH_FACTOR, N_FOREST,
    FOREST_MAX_DEPTH, CAR_MIN_SUPPORT, CAR_MIN_CONFIDENCE, CAR_MAX_RULE_LENGTH,
    N_TRIALS, TRIAL_SEEDS, CPU_COUNT, OUTFILE_REF, RULES_DIR, ALPHAS_DIR, CONFIDENCE_DIR,
)
stamp_reset()

####################################################################################################

import os
import json
import math
import time
import pickle
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics.pairwise import pairwise_distances
from intercluster import *
from intercluster.decision_trees import *
from intercluster.decision_sets import *
from intercluster.decision_sets.ids import IDSCoverageCache
from intercluster.decision_sets.objectives import *
from intercluster.decision_sets.mining import *
from intercluster.measurements import *

os.environ["OMP_NUM_THREADS"] = "1"

# REMINDER: Initialize the seed only here, not inside any sub-function or
# class -- except classes with their own internal randomness (IDS,
# DecisionTree), which accept an explicit random_state instead of relying on
# this global seed. See `trial_seeds` below, which derives one seed per trial
# so those modules can be refit across multiple trials and reported as
# mean/std rather than a single, arbitrarily-seeded point estimate.
seed = SEED

# Number of independent random-seed trials used to evaluate stochastic modules
# (IDS, Decision-Tree). Deterministic modules (PEC, ExKMC, CN2, CBA) are fit
# once. `trial_seeds` is derived deterministically from `seed` so re-running
# this script reproduces the exact same set of trials.
n_trials = N_TRIALS
trial_seeds = TRIAL_SEEDS

# Confidence levels are evaluated concurrently (see run_confidence_level below). Each worker
# builds its own IDS sub-cache -- ids_full_cache.subset() copies the selected rows, which is
# close to the whole cache at conf=0.0 -- so peak memory scales with this number. Lower it if
# RSS becomes a problem on the larger rule pools.
confidence_cpu_count = CPU_COUNT

def _memoryview_safe(x):
    if not x.flags.writeable:
        if not x.flags.owndata:
            x = x.copy(order='C')
        x.setflags(write=True)
    return x

####################################################################################################
# Data + clustering

data, data_labels, feature_labels, scaler = load_preprocessed_mnist()
stamp("data loaded")
data = _memoryview_safe(data)
n, d = data.shape

fixed_parameters = {
    'n': n,
    'd': d,
    'n_clusters': N_CLUSTERS,
    'n_select': N_SELECT_DEFAULT,
    'shallow_tree_depth_factor': SHALLOW_TREE_DEPTH_FACTOR,
    'n_forest': N_FOREST,
    'forest_max_depth': FOREST_MAX_DEPTH,
    'car_min_support': CAR_MIN_SUPPORT,
    'car_min_confidence': CAR_MIN_CONFIDENCE,
    'car_max_rule_length': CAR_MAX_RULE_LENGTH,
    'seed': seed,
    'n_trials': n_trials,
    'trial_seeds': trial_seeds,
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

with open(ALPHAS_DIR + 'selected_alphas' + OUTFILE_REF + '.json') as f:
    selected_alpha_dict = json.load(f)

####################################################################################################
# Load rule pools and IDS lambdas

pre_filter_ensemble = load_rules(RULES_DIR + 'pre_filter_ensemble_rules.pkl')

with open(RULES_DIR + 'pre_filter_ensemble_labels.pkl', 'rb') as f:
    pre_filter_labels = pickle.load(f)

# Dict for O(1) label lookup when filtering rules by confidence level
_pre_filter_label_map = {r: lbl for r, lbl in zip(pre_filter_ensemble, pre_filter_labels)}

with open(RULES_DIR + f'ids_lambdas{OUTFILE_REF}.json') as f:
    ids_lambdas = json.load(f)
if isinstance(ids_lambdas, dict):
    ids_lambdas = list(ids_lambdas.values())

outfile = CONFIDENCE_DIR
os.makedirs(outfile, exist_ok=True)

####################################################################################################
# Objective configuration
# No precomputed_paths: the rule pool changes at each confidence level, so everything
# must be computed from scratch each iteration.
#
# The point-to-center distance matrix is the one exception: it depends only on the data and the
# kmeans centers, neither of which changes with the rule pool, so it is computed once here and
# handed to every cost-based objective. Otherwise each of the many objectives built per confidence
# level (one per PEC fit, plus one per score_decision_set call) would rebuild the same (n x k)
# matrix from scratch.
data_to_center_distances = compute_data_to_center_distances(
    data, kmeans_base.centers, 'kmeans'
)

# Only the 3 unweighted objectives -- examples/experiments.ipynb's confidence-sweep
# plots (confidence_line_dict, confidence_lambda_dict, confidence_component_dict) all
# loop `objective_names`, which never includes a `-weighted` entry. The weighted
# objectives' alpha values are still selected in alphas.py (needed by max_rules.py's
# Uncertainty section), but nothing here reads a per-confidence-level weighted PEC
# fit or a weighted objective score, so refitting/scoring them at every one of the
# 20 confidence levels below would be pure waste.
objective_config = {
    'coverage-mistake': {
        'objective_type': 'coverage-mistake',
    },
    'coverage-cost': {
        'objective_type': 'coverage-cost',
        'cluster_centers': kmeans_base.centers,
        'cluster_cost_method': 'kmeans',
        'data_to_center_distances': data_to_center_distances,
    },
    'coverage-pairwise-distance': {
        'objective_type': 'coverage-pairwise-distance',
    },
}

####################################################################################################
# Helpers
#
# _make_objective, score_decision_set, and tree_to_decisions now live in experiments/modules.py
# (imported above via `from experiments.modules import *`) -- factored out so max_rules.py's
# per-rule-budget objective re-scoring can share the identical logic instead of duplicating it.

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
# Pool-independent algorithms — run once (or, for stochastic models, once per
# trial seed), standard measurements precomputed. These don't depend on the rule
# pool, so they're unaffected by the confidence-threshold sweep below.

print("Fitting pool-independent algorithms...")

_exkmc = ExkmcTree(k=n_labels, kmeans=kmeans_base.clustering, max_leaf_nodes=n_select)
_exkmc.fit(data, kmeans_labels)
exkmc_decisions = tree_to_decisions(_exkmc)

#_cn2 = CN2(n_select=n_select)
#_cn2.fit(data, kmeans_labels)

pool_indep = {
    'ExKMC': _tree_info(_exkmc, exkmc_decisions, data, n_labels),
    #'CN2': _dset_info(_cn2, data, n_labels),
}

# Precompute standard measurements (these do not change across confidence levels)
pool_indep_measurements = {
    name: measure_algo(
        info['data_to_rule'], info['rule_to_cluster'], info['data_to_cluster'], measurement_fns
    )
    for name, info in pool_indep.items()
}

# Decision-Tree has a fitted solution that depends on randomness (sklearn tie-
# breaking). Rather than record one arbitrarily seeded fit, it is refit once per
# seed in `trial_seeds`. Standard measurements are aggregated into
# {'mean','std','values'} via `aggregate_trials`; the per-trial decision sets are
# kept so the PEC-objective score (computed inside the confidence sweep below,
# since it depends on that level's PEC lambda) can also be aggregated across
# trials.

def _fit_pool_indep_trials(model_cls, base_params, seed_key):
    """
    NOTE: sets the global NumPy seed immediately before each trial's fit, in
    addition to passing the trial seed as `seed_key`. Some dependencies (e.g.
    ExplanationTree's compiled Cython splitter -- see explanation_tree.py's
    `random_state` docstring caveat) read the global RNG state directly and
    aren't fully parameterized by a passed-in random_state alone. This runs
    single-process, so setting the global seed here is safe and sufficient.
    """
    trial_infos = []
    trial_measurements = []
    for trial_seed in trial_seeds:
        np.random.seed(trial_seed)
        model = model_cls(**(base_params | {seed_key: trial_seed}))
        model.fit(data, kmeans_labels)
        decisions = tree_to_decisions(model)
        info = _tree_info(model, decisions, data, n_labels)
        trial_infos.append(info)
        trial_measurements.append(
            {
                'n-rules': info['n-rules'],
                'max-rule-length': info['max-rule-length'],
                'sum-rule-length': info['sum-rule-length'],
                'weighted-avg-length': info['weighted-avg-length'],
            } | measure_algo(
                info['data_to_rule'], info['rule_to_cluster'], info['data_to_cluster'], measurement_fns
            )
        )
    return trial_infos, aggregate_trials(trial_measurements)

dt_trial_infos, dt_agg_measurements = _fit_pool_indep_trials(
    DecisionTree, {'max_leaf_nodes': n_select}, 'random_state'
)

# Per-algo list of per-trial decision sets, used for objective scoring in the
# confidence sweep below. Deterministic pool-indep/pool-dep algos are absent here
# and take the single-decisions path instead.
stochastic_trial_infos = {
    'Decision-Tree': dt_trial_infos,
}
stochastic_pool_indep_measurements = {
    'Decision-Tree': dt_agg_measurements,
}

# KMeans baseline measurements (fixed)
baseline_measurements = measure_algo(None, None, kmeans_assignment, measurement_fns)

print("Pool-independent algorithms ready.")
stamp("pool-independent algos (ExKMC/CN2/trees x trials)")

####################################################################################################
# IDS full-pool cache — built once on all CAR rules, then subset per confidence level

_ids_cache_path = RULES_DIR + 'ids_coverage_cache_prefilter.pkl'
if os.path.exists(_ids_cache_path):
    print("Loading pre-built IDS cache...")
    with open(_ids_cache_path, 'rb') as f:
        ids_full_cache = pickle.load(f)
    print(f"IDS cache loaded ({len(ids_full_cache.decisions)} decisions).")
    stamp("IDS full-cache loaded from disk")
else:
    print("Pre-computing IDS coverage cache on full pre-filter ensemble...")
    # Same construction as ids_lambda_search.py and max_rules.py/lambda.py: from_rules keys
    # decisions to rule order and keeps one per rule, and runs no optimizer. (Routing through
    # IDS.fit() instead would build the decision set as a set -- hash order, duplicates collapsed
    # -- and pay for a selection pass that is immediately discarded.)
    ids_full_cache = IDSCoverageCache.from_rules(
        pre_filter_ensemble, pre_filter_labels, data, kmeans_labels
    )
    os.makedirs(os.path.dirname(_ids_cache_path), exist_ok=True)
    with open(_ids_cache_path, 'wb') as f:
        pickle.dump(ids_full_cache, f)
    print(f"IDS cache ready: {len(ids_full_cache.decisions)} decisions.")
    stamp("IDS full-cache BUILT")

####################################################################################################
# Main confidence sweep

confidence_values = np.round(np.arange(0.0, 1.0, 0.05), 2)

result = {'fixed-parameters': fixed_parameters}

# Precompute each pre-filter rule's majority-class fraction ONCE. It does not depend
# on the confidence threshold, so calling filter_rules() inside the sweep below would
# re-evaluate every rule against all N points at each of the 20 levels -- 20x redundant
# O(R*N) work. A rule is kept at threshold `conf` iff it covers >=1 point and its
# majority-class fraction is >= conf, exactly matching filter_rules(..., support=0.0).
_y_flat = flatten_labels(kmeans_labels)
_rule_confidence = []
for _rule in pre_filter_ensemble:
    _idx = satisfies_rule(data, _rule)
    if len(_idx) == 0:
        _rule_confidence.append(None)
    else:
        _labs, _counts = np.unique(_y_flat[_idx], return_counts=True)
        _rule_confidence.append(_counts.max() / len(_idx))

def run_confidence_level(
    conf,
    data,
    kmeans_labels,
    weights,
    n_select,
    n_labels,
    trial_seeds,
    ids_lambdas,
    ids_full_cache,
    pre_filter_ensemble,
    rule_confidence,
    pre_filter_label_map,
    objective_config,
    selected_alpha_dict,
    measurement_fns,
    pool_indep,
    pool_indep_measurements,
    baseline_measurements,
    stochastic_trial_infos,
    stochastic_pool_indep_measurements,
):
    """
    Evaluates one confidence level and returns (conf_key, conf_result).

    Levels are independent -- each one only reads the shared state above and writes its own
    result -- so they are dispatched across processes by the Parallel call below. Everything
    stochastic in here (the IDS trials) is seeded explicitly via random_state=trial_seed and
    reads no global RNG, which is what makes that safe: worker processes do not inherit the
    parent's seeded global NumPy state. The pool-independent tree trials do rely on the global
    seed, which is exactly why they are fit above, in the parent, rather than in here.

    Shared state is passed as explicit arguments rather than read from module globals so that the
    large arrays reliably go through joblib's memmap reducer (which also de-duplicates them by
    identity across tasks) instead of being pickled per task.
    """
    conf_key = float(conf)
    t0 = time.time()

    filtered_rules = [
        r for r, c in zip(pre_filter_ensemble, rule_confidence)
        if c is not None and c >= conf
    ]
    has_rules = len(filtered_rules) > 0
    filtered_labels = [pre_filter_label_map[r] for r in filtered_rules]

    conf_result = {
        'confidence_n_rules': len(filtered_rules),
        'lambda': {},
        'lambda_n_rules': {},
    }

    # ----------------------------------------------------------------
    # PEC — one fit per objective type; captures the lambda used
    # ----------------------------------------------------------------
    pec_lambdas = {}
    pec_n_available = {}
    pec_info = {}

    for obj_name, obj_cfg in objective_config.items():
        pec_name = f'dscluster; {obj_name}; ensemble'
        lazy_pec_name = f'{pec_name}; lazy-greedy'
        alpha = selected_alpha_dict[pec_name]
        if has_rules:
            pec = PEC(rules=filtered_rules, n_select=n_select, alpha_val=alpha, **obj_cfg)
            pec.fit(data, kmeans_labels)
            pec_lambdas[obj_name] = pec.objective.lambda_val
            # The confidence threshold is only the first filter on the pool. lambda* imposes a
            # second one: distorted greedy permanently discards every decision failing
            # g(e | {}) - lambda* h(e) > 0, so this is what PEC actually had to choose from.
            pec_n_available[obj_name] = pec.n_available_decisions
            pec_info[pec_name] = _dset_info(pec, data, n_labels)

            # Lazy-greedy counterpart: same objective/alpha/lambda as the distorted-greedy
            # PEC above (mirrors max_rules.py's dscluster_module_list loop), but with
            # selection_algorithm='lazy-greedy'. A genuine second model (lazy-greedy carries
            # no approximation-guarantee threshold on lambda), not just the degenerate-case
            # fallback. If the primary fit above already degenerated (no valid lambda*, so
            # set_lambda switched it to lazy-greedy with lambda=0), reusing that lambda_val
            # would just be lambda_val=0 with no fallback distinction anyway -- but passing
            # lambda_val=None here instead lets this fit recompute the same fallback itself,
            # matching max_rules.py's "leave those objectives on the original per-fit path".
            lazy_lambda_val = (
                pec.objective.lambda_val
                if pec.objective.selection_algorithm == 'distorted-greedy'
                else None
            )
            lazy_pec = PEC(
                rules=filtered_rules, n_select=n_select, alpha_val=alpha,
                lambda_val=lazy_lambda_val, selection_algorithm='lazy-greedy', **obj_cfg
            )
            lazy_pec.fit(data, kmeans_labels)
            pec_info[lazy_pec_name] = _dset_info(lazy_pec, data, n_labels)
        else:
            pec_lambdas[obj_name] = np.nan
            pec_n_available[obj_name] = np.nan
            pec_info[pec_name] = _empty_info()
            pec_info[lazy_pec_name] = _empty_info()

    conf_result['lambda'] = pec_lambdas
    conf_result['lambda_n_rules'] = pec_n_available

    # ----------------------------------------------------------------
    # CBA (pool-dependent)
    # ----------------------------------------------------------------
    pool_dep = {}

    if has_rules:
        _cba = CBA(rules=filtered_rules, n_select=n_select, rule_labels=filtered_labels)
        _cba.fit(data, kmeans_labels)
        pool_dep['CBA'] = _dset_info(_cba, data, n_labels)
    else:
        pool_dep['CBA'] = _empty_info()

    # ----------------------------------------------------------------
    # IDS — subset the full cache to filtered ensemble rules each iteration;
    # lambdas are fixed (fitted via coordinate ascent on the full pool). IDS's
    # selection step (randomized-greedy / SLS) has inherent randomness, so it is
    # refit once per seed in `trial_seeds` at every confidence level and the
    # results aggregated, rather than recording one arbitrarily-seeded fit.
    # ----------------------------------------------------------------
    ids_trial_infos = []
    if filtered_rules:
        filtered_rule_set = set(filtered_rules)
        filtered_indices = [
            i for i, d in enumerate(ids_full_cache.decisions)
            if d.rule in filtered_rule_set
        ]
        ids_sub_cache = ids_full_cache.subset(filtered_indices)
        ids_trial_measurements = []
        for trial_seed in trial_seeds:
            _ids = IDS(
                rules=filtered_rules,
                rule_labels=filtered_labels,
                n_select=n_select,
                lambdas=ids_lambdas,
                cache=ids_sub_cache,
                optimizer='random_greedy',
                random_state=trial_seed,
            )
            _ids.fit(data, kmeans_labels)
            info = _dset_info(_ids, data, n_labels)
            ids_trial_infos.append(info)
            ids_trial_measurements.append(
                {
                    'n-rules': info['n-rules'],
                    'max-rule-length': info['max-rule-length'],
                    'sum-rule-length': info['sum-rule-length'],
                    'weighted-avg-length': info['weighted-avg-length'],
                } | measure_algo(
                    info['data_to_rule'], info['rule_to_cluster'], info['data_to_cluster'], measurement_fns
                )
            )
        ids_agg_measurements = aggregate_trials(ids_trial_measurements)
    else:
        ids_agg_measurements = aggregate_trials([
            {
                'n-rules': 0, 'max-rule-length': np.nan, 'sum-rule-length': np.nan,
                'weighted-avg-length': np.nan,
            } | measure_algo(None, None, None, measurement_fns)
        ])

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

        # PEC objective score for each of the (3) objective types in objective_config
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
                    data_to_center_distances=obj_cfg.get('data_to_center_distances'),
                )

        conf_result[algo_name] = algo_result

    # ----------------------------------------------------------------
    # Stochastic algorithms (Decision-Tree, IDS): standard measurements were
    # already aggregated across trials above (either once, for the
    # pool-independent tree, or per confidence level, for IDS). The PEC-objective
    # score is computed per trial (since it depends on this confidence level's
    # PEC lambda) and aggregated the same way.
    # ----------------------------------------------------------------
    stochastic_agg_measurements = {
        'Decision-Tree': stochastic_pool_indep_measurements['Decision-Tree'],
        'IDS': ids_agg_measurements,
    }
    stochastic_infos_for_objective = {
        'Decision-Tree': stochastic_trial_infos['Decision-Tree'],
        'IDS': ids_trial_infos,
    }

    for algo_name, agg_measurements in stochastic_agg_measurements.items():
        algo_result = dict(agg_measurements)
        trial_infos_list = stochastic_infos_for_objective[algo_name]

        algo_result['objective'] = {}
        for obj_name, obj_cfg in objective_config.items():
            lambda_c = pec_lambdas[obj_name]
            alpha = selected_alpha_dict[f'dscluster; {obj_name}; ensemble']
            trial_scores = []
            if not np.isnan(lambda_c):
                for info in trial_infos_list:
                    decisions = info['decisions']
                    if not decisions:
                        continue
                    trial_scores.append(
                        score_decision_set(
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
                            data_to_center_distances=obj_cfg.get('data_to_center_distances'),
                        )
                    )
            if trial_scores:
                algo_result['objective'][obj_name] = {
                    'mean': float(np.mean(trial_scores)),
                    'std': float(np.std(trial_scores)),
                    'values': trial_scores,
                }
            else:
                algo_result['objective'][obj_name] = np.nan

        conf_result[algo_name] = algo_result

    print(
        f"  [confidence={conf_key:.2f}] {len(filtered_rules)} ensemble rules, "
        f"done in {time.time() - t0:.1f}s"
    )
    return conf_key, conf_result


stamp("starting confidence sweep (parallel)")

level_results = Parallel(n_jobs=confidence_cpu_count, backend='loky')(
    delayed(run_confidence_level)(
        conf,
        data,
        kmeans_labels,
        weights,
        n_select,
        n_labels,
        trial_seeds,
        ids_lambdas,
        ids_full_cache,
        pre_filter_ensemble,
        _rule_confidence,
        _pre_filter_label_map,
        objective_config,
        selected_alpha_dict,
        measurement_fns,
        pool_indep,
        pool_indep_measurements,
        baseline_measurements,
        stochastic_trial_infos,
        stochastic_pool_indep_measurements,
    )
    for conf in confidence_values
)

# Parallel preserves submission order, so levels land in confidence_values order.
for conf_key, conf_result in level_results:
    result[conf_key] = conf_result

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

fname = os.path.join(outfile, f'exp_confidence{OUTFILE_REF}.json')
with open(fname, 'w') as f:
    json.dump(result, f, indent=4, cls=NumpyEncoder)

print(f"\nResults saved to {fname}")
stamp("confidence sweep complete + saved")

####################################################################################################
