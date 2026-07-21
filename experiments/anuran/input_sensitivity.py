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
from experiments.profiling import stamp, stamp_reset
from experiments.anuran.config import (
    SEED, N_CLUSTERS, N_SELECT_DEFAULT, N_TRIALS, TRIAL_SEEDS, CPU_COUNT,
    OUTFILE_REF, RULES_DIR, ALPHAS_DIR, INPUT_SENSITIVITY_DIR,
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

# REMINDER: The seed should only be initialized here. Classes with their own
# internal randomness (IDS, DecisionTree) accept an explicit random_state
# instead of relying on this global seed. See trial_seeds below: repeat r
# pairs TRIAL_SEEDS[r] as the seed for that repeat's random CAR draw *and* as
# the random_state passed to IDS's own fit, and is paired with the
# pool-independent Decision-Tree's own trial-r fit for objective scoring --
# one seed per repeat, not a separate nested trial axis.
seed = SEED

n_trials = N_TRIALS
trial_seeds = TRIAL_SEEDS

# (p, repeat) draws are evaluated concurrently (see run_pool_repeat below). Each worker builds
# its own IDS sub-cache -- ids_full_cache.subset() copies the selected rows -- so peak memory
# scales with this number. Lower it if RSS becomes a problem on the larger rule pools.
input_sensitivity_cpu_count = CPU_COUNT

def _memoryview_safe(x):
    if not x.flags.writeable:
        if not x.flags.owndata:
            x = x.copy(order='C')
        x.setflags(write=True)
    return x

####################################################################################################
# Data + clustering

data, data_labels, feature_labels, scaler = load_preprocessed_anuran('data/anuran')
stamp("data loaded")
data = _memoryview_safe(data)
n, d = data.shape

p_values = np.round(np.arange(0.0, 1.01, 0.1), 2)

fixed_parameters = {
    'n': n,
    'd': d,
    'n_clusters': N_CLUSTERS,
    'n_select': N_SELECT_DEFAULT,
    'seed': seed,
    'n_repeats': n_trials,
    'trial_seeds': trial_seeds,
    'p_values': p_values.tolist(),
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
# Load alpha values (fixed from alphas.py selection -- pool-composition-independent, so this is
# reused unchanged across every p / repeat, exactly as confidence.py already does)

with open(ALPHAS_DIR + f'selected_alphas{OUTFILE_REF}.json') as f:
    selected_alpha_dict = json.load(f)

####################################################################################################
# Load rule pools and IDS lambdas
#
# The base pool (decision_tree_rules + forest_rules) is loaded directly from its own per-miner
# cache files rather than recovered from pre_filter_ensemble by set difference -- this mirrors
# mine_rules.py's own construction (pre_filter_ensemble = decision_tree_rules + forest_rules +
# class_association_rules) and sidesteps any ambiguity if a rule happened to appear in two
# source lists. Note ExKMC rules are mined into exkmc_rules.pkl but were never folded into
# pre_filter_ensemble by mine_rules.py, so they are excluded from the base pool here too.

decision_tree_rules = load_rules(RULES_DIR + 'decision_tree_rules.pkl')
forest_rules = load_rules(RULES_DIR + 'forest_rules.pkl')
class_association_rules = load_rules(RULES_DIR + 'class_association_rules.pkl')
base_rules = decision_tree_rules + forest_rules
n_car_total = len(class_association_rules)

pre_filter_ensemble = load_rules(RULES_DIR + 'pre_filter_ensemble_rules.pkl')

with open(RULES_DIR + 'pre_filter_ensemble_labels.pkl', 'rb') as f:
    pre_filter_labels = pickle.load(f)

# Dict for O(1) label lookup for any rule drawn from the three source lists above -- every one
# of those rules is, by construction, a member of pre_filter_ensemble.
_pre_filter_label_map = {r: lbl for r, lbl in zip(pre_filter_ensemble, pre_filter_labels)}

with open(RULES_DIR + f'ids_lambdas{OUTFILE_REF}.json') as f:
    ids_lambdas = json.load(f)
if isinstance(ids_lambdas, dict):
    ids_lambdas = list(ids_lambdas.values())

outfile = INPUT_SENSITIVITY_DIR
os.makedirs(outfile, exist_ok=True)

####################################################################################################
# Objective configuration
# No precomputed_paths: the rule pool changes at each (p, repeat) draw, so everything must be
# computed from scratch each time.

data_to_center_distances = compute_data_to_center_distances(
    data, kmeans_base.centers, 'kmeans'
)

# Only the 3 unweighted objectives -- see confidence.py's identical comment: nothing downstream
# reads a per-draw weighted PEC fit or objective score.
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
# Helpers (identical to confidence.py)

def _make_objective(objective_type, n_select, alpha_val, lambda_val=None,
                    cluster_centers=None, cluster_cost_method='kmeans', weights=None,
                    data_to_center_distances=None):
    """Instantiate the right Objective subclass without precomputed data."""
    common = dict(n_select=n_select, alpha_val=alpha_val, lambda_val=lambda_val, weights=weights)
    if objective_type == 'coverage-mistake':
        return CoverageMistakeObjective(**common)
    elif objective_type == 'coverage-cost':
        return CoverageCostObjective(
            cluster_centers=cluster_centers,
            cluster_cost_method=cluster_cost_method,
            data_to_center_distances=data_to_center_distances,
            **common,
        )
    elif objective_type == 'coverage-pairwise-distance':
        return CoveragePairwiseDistanceObjective(**common)
    else:
        raise ValueError(f'Unknown objective type: {objective_type}')


def score_decision_set(decisions, X, y, n_select, objective_type, alpha_val, lambda_val,
                       cluster_centers=None, cluster_cost_method='kmeans', weights=None,
                       data_to_center_distances=None):
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
        data_to_center_distances=data_to_center_distances,
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


def measure_algo(data_to_rule, rule_to_cluster, data_to_cluster, measurement_fns):
    return {fn.name: fn(data_to_rule, rule_to_cluster, data_to_cluster) for fn in measurement_fns}


####################################################################################################
# Measurement functions (identical set to confidence.py/max_rules.py)

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
# Pool-independent algorithms — run once (or, for stochastic models, once per trial seed),
# standard measurements precomputed. These don't depend on the rule pool, so they're unaffected
# by the CAR-fraction sweep below. Identical to confidence.py's pool-independent section.

print("Fitting pool-independent algorithms...")

_exkmc = ExkmcTree(k=n_labels, kmeans=kmeans_base.clustering, max_leaf_nodes=n_select)
_exkmc.fit(data, kmeans_labels)
exkmc_decisions = tree_to_decisions(_exkmc)

_cn2 = CN2(n_select=n_select)
_cn2.fit(data, kmeans_labels)

pool_indep = {
    'ExKMC': _tree_info(_exkmc, exkmc_decisions, data, n_labels),
    'CN2': _dset_info(_cn2, data, n_labels),
}

pool_indep_measurements = {
    name: measure_algo(
        info['data_to_rule'], info['rule_to_cluster'], info['data_to_cluster'], measurement_fns
    )
    for name, info in pool_indep.items()
}

# Decision-Tree is refit once per seed in `trial_seeds` (sklearn tie-breaking randomness),
# exactly as in confidence.py. Repeat r's PEC-objective score below pairs with dt_trial_infos[r].
def _fit_pool_indep_trials(model_cls, base_params, seed_key):
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

# KMeans baseline measurements (fixed)
baseline_measurements = measure_algo(None, None, kmeans_assignment, measurement_fns)

print("Pool-independent algorithms ready.")
stamp("pool-independent algos (ExKMC/CN2/trees x trials)")

####################################################################################################
# IDS full-pool cache — built once on the entire pre-filter ensemble (must exist already, built
# by confidence.py or mine_rules.py's pipeline), then subset per (p, repeat) draw.

_ids_cache_path = RULES_DIR + 'ids_coverage_cache_prefilter.pkl'
if os.path.exists(_ids_cache_path):
    print("Loading pre-built IDS cache...")
    with open(_ids_cache_path, 'rb') as f:
        ids_full_cache = pickle.load(f)
    print(f"IDS cache loaded ({len(ids_full_cache.decisions)} decisions).")
    stamp("IDS full-cache loaded from disk")
else:
    print("Pre-computing IDS coverage cache on full pre-filter ensemble...")
    ids_full_cache = IDSCoverageCache.from_rules(
        pre_filter_ensemble, pre_filter_labels, data, kmeans_labels
    )
    os.makedirs(os.path.dirname(_ids_cache_path), exist_ok=True)
    with open(_ids_cache_path, 'wb') as f:
        pickle.dump(ids_full_cache, f)
    print(f"IDS cache ready: {len(ids_full_cache.decisions)} decisions.")
    stamp("IDS full-cache BUILT")

####################################################################################################
# Main CAR-fraction sweep

def run_pool_repeat(
    p,
    repeat_idx,
    trial_seed,
    data,
    kmeans_labels,
    n_select,
    n_labels,
    base_rules,
    class_association_rules,
    ids_lambdas,
    ids_full_cache,
    pre_filter_label_map,
    objective_config,
    selected_alpha_dict,
    measurement_fns,
    pool_indep,
    dt_trial_infos,
):
    """
    Draws a random p-fraction of the CAR pool (seeded by trial_seed), refits every
    pool-dependent algorithm (PEC x3 objectives x{distorted-greedy, lazy-greedy}, CBA, IDS) on
    base_rules + that draw, and scores the pool-independent algorithms' fixed decision sets
    (whichever of ExKMC/CN2 are present, plus this repeat's Decision-Tree trial) against this
    draw's own PEC lambda*.

    (p, repeat) draws are independent -- each only reads the shared state above and writes its
    own result -- so they are dispatched across processes by the Parallel call below. trial_seed
    seeds both the CAR draw (via np.random.RandomState) and IDS's own random_state, so IDS's
    internal randomness and the CAR subsample are paired to a single per-repeat seed rather than
    treated as two independent axes (10 fits per p, not 100). Shared state is passed as explicit
    arguments rather than read from module globals so that the large arrays reliably go through
    joblib's memmap reducer instead of being pickled per task.
    """
    t0 = time.time()
    p = float(p)

    rng = np.random.RandomState(trial_seed)
    n_car = int(round(p * len(class_association_rules)))
    if n_car > 0:
        car_indices = rng.choice(len(class_association_rules), size=n_car, replace=False)
        car_subset = [class_association_rules[i] for i in car_indices]
    else:
        car_subset = []
    rule_pool = base_rules + car_subset
    rule_labels = [pre_filter_label_map[r] for r in rule_pool]

    repeat_result = {'car_n_rules': n_car}

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

        pec = PEC(rules=rule_pool, n_select=n_select, alpha_val=alpha, **obj_cfg)
        pec.fit(data, kmeans_labels)
        pec_lambdas[obj_name] = pec.objective.lambda_val
        pec_n_available[obj_name] = pec.n_available_decisions
        pec_info[pec_name] = _dset_info(pec, data, n_labels)

        lazy_lambda_val = (
            pec.objective.lambda_val
            if pec.objective.selection_algorithm == 'distorted-greedy'
            else None
        )
        lazy_pec = PEC(
            rules=rule_pool, n_select=n_select, alpha_val=alpha,
            lambda_val=lazy_lambda_val, selection_algorithm='lazy-greedy', **obj_cfg
        )
        lazy_pec.fit(data, kmeans_labels)
        pec_info[lazy_pec_name] = _dset_info(lazy_pec, data, n_labels)

    repeat_result['lambda'] = pec_lambdas
    repeat_result['lambda_n_rules'] = pec_n_available

    # ----------------------------------------------------------------
    # CBA (pool-dependent)
    # ----------------------------------------------------------------
    _cba = CBA(rules=rule_pool, n_select=n_select, rule_labels=rule_labels)
    _cba.fit(data, kmeans_labels)
    cba_info = _dset_info(_cba, data, n_labels)

    # ----------------------------------------------------------------
    # IDS — subset the full pre-filter cache to this repeat's drawn pool.
    # ----------------------------------------------------------------
    rule_pool_set = set(rule_pool)
    pool_indices = [
        i for i, dec in enumerate(ids_full_cache.decisions)
        if dec.rule in rule_pool_set
    ]
    ids_sub_cache = ids_full_cache.subset(pool_indices)
    _ids = IDS(
        rules=rule_pool,
        rule_labels=rule_labels,
        n_select=n_select,
        lambdas=ids_lambdas,
        cache=ids_sub_cache,
        optimizer='random_greedy',
        random_state=trial_seed,
    )
    _ids.fit(data, kmeans_labels)
    ids_info = _dset_info(_ids, data, n_labels)

    # ----------------------------------------------------------------
    # Standard measurements + PEC-objective score for every pool-dependent algorithm
    # (PEC variants, CBA, IDS) -- their entire decision set varies with this repeat's draw.
    # ----------------------------------------------------------------
    pool_dep_measurements = {}
    all_pool_dep_info = {**pec_info, 'CBA': cba_info, 'IDS': ids_info}
    for algo_name, info in all_pool_dep_info.items():
        algo_result = {
            'n-rules': info['n-rules'],
            'max-rule-length': info['max-rule-length'],
            'sum-rule-length': info['sum-rule-length'],
            'weighted-avg-length': info['weighted-avg-length'],
        }
        algo_result.update(
            measure_algo(info['data_to_rule'], info['rule_to_cluster'], info['data_to_cluster'], measurement_fns)
        )
        algo_result['objective'] = {}
        for obj_name, obj_cfg in objective_config.items():
            lambda_c = pec_lambdas[obj_name]
            alpha = selected_alpha_dict[f'dscluster; {obj_name}; ensemble']
            decisions = info['decisions']
            algo_result['objective'][obj_name] = (
                np.nan if not decisions else score_decision_set(
                    decisions, data, kmeans_labels, n_select=n_select,
                    objective_type=obj_cfg['objective_type'], alpha_val=alpha,
                    lambda_val=lambda_c,
                    cluster_centers=obj_cfg.get('cluster_centers'),
                    cluster_cost_method=obj_cfg.get('cluster_cost_method', 'kmeans'),
                    weights=obj_cfg.get('weights'),
                    data_to_center_distances=obj_cfg.get('data_to_center_distances'),
                )
            )
        pool_dep_measurements[algo_name] = algo_result
    repeat_result['pool_dep_measurements'] = pool_dep_measurements

    # ----------------------------------------------------------------
    # Pool-independent algorithms' PEC-objective score against *this repeat's* lambda*.
    # Their decision sets don't change with the CAR draw, but lambda* does.
    # ----------------------------------------------------------------
    pool_indep_for_scoring = {name: info['decisions'] for name, info in pool_indep.items()}
    pool_indep_for_scoring['Decision-Tree'] = dt_trial_infos[repeat_idx]['decisions']
    pool_indep_objective = {}
    for algo_name, decisions in pool_indep_for_scoring.items():
        algo_obj = {}
        for obj_name, obj_cfg in objective_config.items():
            lambda_c = pec_lambdas[obj_name]
            alpha = selected_alpha_dict[f'dscluster; {obj_name}; ensemble']
            algo_obj[obj_name] = (
                np.nan if not decisions else score_decision_set(
                    decisions, data, kmeans_labels, n_select=n_select,
                    objective_type=obj_cfg['objective_type'], alpha_val=alpha,
                    lambda_val=lambda_c,
                    cluster_centers=obj_cfg.get('cluster_centers'),
                    cluster_cost_method=obj_cfg.get('cluster_cost_method', 'kmeans'),
                    weights=obj_cfg.get('weights'),
                    data_to_center_distances=obj_cfg.get('data_to_center_distances'),
                )
            )
        pool_indep_objective[algo_name] = algo_obj
    repeat_result['pool_indep_objective'] = pool_indep_objective

    print(
        f"  [p={p:.1f}, repeat={repeat_idx}] {n_car}/{len(class_association_rules)} CAR rules "
        f"({len(rule_pool)} total), done in {time.time() - t0:.1f}s"
    )
    return p, repeat_idx, repeat_result


stamp("starting input-sensitivity sweep (parallel)")

draw_results = Parallel(n_jobs=input_sensitivity_cpu_count, backend='loky')(
    delayed(run_pool_repeat)(
        p,
        repeat_idx,
        trial_seeds[repeat_idx],
        data,
        kmeans_labels,
        n_select,
        n_labels,
        base_rules,
        class_association_rules,
        ids_lambdas,
        ids_full_cache,
        _pre_filter_label_map,
        objective_config,
        selected_alpha_dict,
        measurement_fns,
        pool_indep,
        dt_trial_infos,
    )
    for p in p_values
    for repeat_idx in range(n_trials)
)

# Group the 110 (p, repeat) draws back by p.
grouped_by_p = {float(p): [] for p in p_values}
for p, repeat_idx, repeat_result in draw_results:
    grouped_by_p[p].append((repeat_idx, repeat_result))

####################################################################################################
# Aggregate each p level across its 10 repeats

result = {'fixed-parameters': fixed_parameters}

for p in p_values:
    p_key = float(p)
    repeats = sorted(grouped_by_p[p_key], key=lambda x: x[0])
    repeat_results = [rr for _, rr in repeats]

    p_result = {
        'car_n_rules': repeat_results[0]['car_n_rules'],
        'lambda': aggregate_trials([rr['lambda'] for rr in repeat_results]),
        'lambda_n_rules': aggregate_trials([rr['lambda_n_rules'] for rr in repeat_results]),
    }

    # KMeans baseline (fixed, no p dependence, no objective score)
    p_result['KMeans'] = {
        'n-rules': np.nan,
        'max-rule-length': np.nan,
        'sum-rule-length': np.nan,
        'weighted-avg-length': np.nan,
    }
    p_result['KMeans'].update(baseline_measurements)

    # Pool-dependent algorithms (PEC variants, CBA, IDS): every measurement (including
    # objective) is aggregated across the 10 repeats, since their whole decision set varies.
    pool_dep_algo_names = list(repeat_results[0]['pool_dep_measurements'].keys())
    for algo_name in pool_dep_algo_names:
        per_repeat_flat = []
        per_repeat_objective = []
        for rr in repeat_results:
            algo_result = rr['pool_dep_measurements'][algo_name]
            per_repeat_flat.append({k: v for k, v in algo_result.items() if k != 'objective'})
            per_repeat_objective.append(algo_result['objective'])
        agg = aggregate_trials(per_repeat_flat)
        agg['objective'] = aggregate_trials(per_repeat_objective)
        p_result[algo_name] = agg

    # Pool-independent algorithms other than Decision-Tree (ExKMC, CN2 -- whichever are
    # present in `pool_indep`; some datasets drop CN2 for runtime reasons): standard
    # measurements are fixed (precomputed once, unaffected by the CAR draw); only the
    # objective score is aggregated across the 10 repeats.
    for algo_name in pool_indep.keys():
        algo_result = {
            'n-rules': pool_indep[algo_name]['n-rules'],
            'max-rule-length': pool_indep[algo_name]['max-rule-length'],
            'sum-rule-length': pool_indep[algo_name]['sum-rule-length'],
            'weighted-avg-length': pool_indep[algo_name]['weighted-avg-length'],
        }
        algo_result.update(pool_indep_measurements[algo_name])
        algo_result['objective'] = aggregate_trials(
            [rr['pool_indep_objective'][algo_name] for rr in repeat_results]
        )
        p_result[algo_name] = algo_result

    # Decision-Tree: standard measurements come from its own 10-trial-seed aggregation
    # (dt_agg_measurements, computed once, unaffected by p); objective is aggregated across
    # the 10 (p, repeat) draws, pairing repeat r's Decision-Tree trial with repeat r's lambda*.
    dt_result = dict(dt_agg_measurements)
    dt_result['objective'] = aggregate_trials(
        [rr['pool_indep_objective']['Decision-Tree'] for rr in repeat_results]
    )
    p_result['Decision-Tree'] = dt_result

    result[p_key] = p_result

####################################################################################################
# Save results

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return None if math.isnan(obj) else float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

fname = os.path.join(outfile, f'exp_input_sensitivity{OUTFILE_REF}.json')
with open(fname, 'w') as f:
    json.dump(result, f, indent=4, cls=NumpyEncoder)

print(f"\nResults saved to {fname}")
stamp("input-sensitivity sweep complete + saved")

####################################################################################################
