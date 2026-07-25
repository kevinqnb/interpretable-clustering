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
from experiments.anuran.config import (
    SEED, N_CLUSTERS, N_SELECT_DEFAULT, MAX_RULES, SHALLOW_TREE_DEPTH_FACTOR,
    N_FOREST, FOREST_MAX_DEPTH, CAR_MIN_SUPPORT, CAR_MIN_CONFIDENCE,
    CAR_MAX_RULE_LENGTH, CONFIDENCE_DEFAULT, N_TRIALS, TRIAL_SEEDS, CPU_COUNT,
    OUTFILE_REF, RULES_DIR, ALPHAS_DIR, MAX_RULES_DIR,
)

####################################################################################################

import os
import json
import math
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from intercluster import *
from intercluster.decision_trees import *
from intercluster.decision_sets import *
from intercluster.decision_sets.ids import IDSCoverageCache
from intercluster.decision_sets.objectives import *
from intercluster.decision_sets.mining import *
from intercluster.measurements import *


# Prevents memory leakage for KMeans:
os.environ["OMP_NUM_THREADS"] = "1"

experiment_cpu_count = CPU_COUNT

# REMINDER: Initialize the seed only here, not inside any sub-function or
# class (except select baseline experiments like KMeans) -- passing a seed
# there resets it on every call.
# Classes with their own internal randomness (IDS, DecisionTree) accept an
# explicit random_state instead of relying on this global seed -- see
# `trial_seeds` below, which derives one seed per trial so those modules can be
# refit across multiple trials and their results reported as mean/std rather
# than a single, arbitrarily-seeded point estimate.
seed = SEED

# Number of independent random-seed trials used to evaluate stochastic modules
# (IDS, Decision-Tree). Deterministic modules (PEC, ExKMC, CN2, CBA) are fit
# once, since repeating them would just reproduce the same result. `trial_seeds`
# is derived deterministically from `seed` so that re-running this script
# reproduces the exact same set of trials.
n_trials = N_TRIALS
trial_seeds = TRIAL_SEEDS

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
data, data_labels, feature_labels, scaler = load_preprocessed_anuran('data/anuran')
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
    'car_max_rule_length': CAR_MAX_RULE_LENGTH, # (really means 6 by pyfim convention)
    'filter_confidence': CONFIDENCE_DEFAULT,
    'seed': seed,
    'n_trials': n_trials,
    'trial_seeds': trial_seeds,
}

n_rules_list = list(range(fixed_parameters['n_clusters'], fixed_parameters['max_rules'] + 1))

np.random.seed(fixed_parameters['seed'])

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

outfile = MAX_RULES_DIR
outfile_ref = OUTFILE_REF

####################################################################################################
# Load pre-mined rules:


ensemble_rules = load_rules(RULES_DIR + f'ensemble_rules{OUTFILE_REF}.pkl')

with open(RULES_DIR + f'ensemble_labels{OUTFILE_REF}.pkl', 'rb') as f:
    ensemble_labels = pickle.load(f)

rule_miner_dict = {
    'ensemble': (None, ensemble_rules, None),
}

####################################################################################################
# Comparison Modules:
#
# NOTE on reproducibility: Decision-Tree and IDS both have inherent randomness in
# their fitted solution (sklearn tree tie-breaking and randomized-greedy/SLS
# selection respectively). Rather than fit each once under the single global
# `seed`, these two are refit across `trial_seeds` further below (see
# "Stochastic module trials") and their results are recorded as mean/std/values
# instead of a single point estimate. Their *_params dicts below therefore omit
# any seed -- the per-trial seed is injected at fit time. PEC, ExKMC, CBA,
# and CN2 have no internal randomness given fixed inputs, so they keep the
# original single-fit-per-parameter-value treatment via `Experiment`.

decision_tree_params_by_r = {i : {'max_leaf_nodes' : i} for i in n_rules_list}
decision_tree_mod = DecisionTreeMod(
    model = DecisionTree,
    name = 'Decision-Tree'
)

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

cba_params = {(r,): {'n_select': r} for r in n_rules_list}
cba_mod = DecisionSetMod(
    model=CBA,
    rules=ensemble_rules,
    rule_labels=ensemble_labels,
    name='CBA'
)

with open(RULES_DIR + f'ids_lambdas{OUTFILE_REF}.json') as f:
    ids_lambdas = json.load(f)
if isinstance(ids_lambdas, dict):
    ids_lambdas = list(ids_lambdas.values())

_ids_cache_path = RULES_DIR + f'ids_coverage_cache_ensemble{OUTFILE_REF}.pkl'
if os.path.exists(_ids_cache_path):
    print("Loading pre-built IDS cache...")
    with open(_ids_cache_path, 'rb') as f:
        ids_cache = pickle.load(f)
    print(f"IDS cache loaded ({len(ids_cache.decisions)} decisions).")
else:
    print("Pre-computing IDS cache...")
    # Built exactly the way ids_lambda_search.py builds it -- IDSCoverageCache.from_rules over the
    # ensemble rules and their labels -- so this fallback reproduces the cached file rather than a
    # different one. Two things make that worth being careful about: from_rules keys decisions to
    # rule order and keeps one per rule, whereas routing through IDS.fit()/set_labels builds a set
    # (hash order, and duplicate decisions silently collapse), and the IDS optimizer indexes into
    # that ordering. from_rules also runs no optimizer, unlike fit(), which would run a full
    # selection pass over the whole pool purely as a side effect and then discard it.
    ids_cache = IDSCoverageCache.from_rules(
        ensemble_rules, ensemble_labels, data, kmeans_labels
    )
    with open(_ids_cache_path, 'wb') as f:
        pickle.dump(ids_cache, f)
    print(f"IDS cache ready: {len(ids_cache.decisions)} decisions.")

ids_params_by_r = {
    r: {
        'n_select': r,
        'lambdas': ids_lambdas,
        'cache': ids_cache,
        'optimizer': 'random_greedy',
    } for r in n_rules_list
}
ids_mod = DecisionSetMod(
    model=IDS,
    rules=ensemble_rules,
    rule_labels=ensemble_labels,
    name='IDS'
)

####################################################################################################
# Objectives for Decision Set Clustering:

objective_dict = {
    'coverage-mistake': {
        'objective_type': 'coverage-mistake',
        'precomputed_path': os.path.join(
            decision_info_dict_directory, f'mistake_info_dict{OUTFILE_REF}.pkl.gz'
        )
    },
    'coverage-cost': {
        'objective_type': 'coverage-cost',
        'cluster_centers': kmeans_base.centers,
        'cluster_cost_method': 'kmeans',
        'precomputed_path': os.path.join(
            decision_info_dict_directory, f'cost_info_dict{OUTFILE_REF}.pkl.gz'
        )
    },
    'coverage-pairwise-distance': {
        'objective_type': 'coverage-pairwise-distance',
        'precomputed_path': os.path.join(
            decision_info_dict_directory, f'pairwise_distance_info_dict{OUTFILE_REF}.pkl.gz'
        )
    },
    'coverage-mistake-weighted': {
        'objective_type': 'coverage-mistake',
        'weights': weights,
        'precomputed_path': os.path.join(
            decision_info_dict_directory, f'mistake_info_dict{OUTFILE_REF}.pkl.gz'
        )
    },
    'coverage-cost-weighted': {
        'objective_type': 'coverage-cost',
        'cluster_centers': kmeans_base.centers,
        'weights': weights,
        'cluster_cost_method': 'kmeans',
        'precomputed_path': os.path.join(
            decision_info_dict_directory, f'cost_info_dict{OUTFILE_REF}.pkl.gz'
        )
    },
    'coverage-pairwise-distance-weighted': {
        'objective_type': 'coverage-pairwise-distance',
        'weights': weights,
        'precomputed_path': os.path.join(
            decision_info_dict_directory, f'pairwise_distance_info_dict{OUTFILE_REF}.pkl.gz'
        )
    },
}

# The 3 unweighted objectives examples/experiments.ipynb's Bar Plots section actually reads an
# 'objective' value for (weighted objectives are read directly off their own measurements
# instead -- see the Uncertainty/Rule-Length-Distribution sections -- so scoring them here would
# be wasted work). Every module below is rescored against each of these using its OWN fixed
# lambda*/alpha (see `score_objectives_by_r`), the same way confidence.py scores its
# per-confidence-level comparison models against PEC's lambda -- rather than leaving the notebook
# to reconstruct an objective value from separately-reported reward/cost measurements, which
# silently drifts if a measurement's units don't exactly match the objective's internal cost.
SCORING_OBJECTIVE_NAMES = ['coverage-mistake', 'coverage-cost', 'coverage-pairwise-distance']

####################################################################################################
# Decision Set Clustering Modules:
#
# lambda* is probed ONCE per objective and then passed to every fit of that objective.
#
# With lambda_val left as None, each PEC fit calls Objective.compute_lambdas(), which evaluates
# reward() once per decision -- and since these modules pass rule_labels=None, the decision pool is
# every (rule, cluster) pair, so that is |rules| * k reward evaluations on every fit. Repeated
# across each rule budget r, it dominates this script.
#
# lambda* does not depend on n_select: it is derived from each decision's reward and cost, and
# n_select enters the objective only through the (1 - 1/n_select) factor inside
# distorted_greedy_select. This experiment holds alpha fixed per objective (alpha *does* affect
# lambda*, via cost) and varies only r, so a single probe is valid for the whole sweep.
#
# compute_lambda_star() runs the same code fit() would, minus the selection pass, and with the
# precomputed_path caches above it is essentially just the one compute_lambdas() call we intend
# to pay exactly once.

lambda_star_dict = {}

dscluster_module_list = []
for obj_name, obj_params in objective_dict.items():
    for rule_miner_name, (rule_miner, rules, rule_labels) in rule_miner_dict.items():
        module_name = f'dscluster; {obj_name}; {rule_miner_name}'
        alpha_val = fixed_parameters['alpha'][module_name]

        probe = PEC(
            rules = rules,
            **({'n_select' : fixed_parameters['n_select'], 'alpha_val' : alpha_val} | obj_params |
               {'lambda_val' : None, 'selection_algorithm' : 'distorted-greedy'})
        )
        lambda_star = probe.compute_lambda_star(data, kmeans_labels)

        # Degenerate case: with no valid lambda, set_lambda falls back to lambda 0 AND switches
        # the objective to lazy-greedy. Reusing the probed value would silently drop that switch,
        # so leave those objectives on the original per-fit path.
        if probe.objective.selection_algorithm != 'distorted-greedy':
            lambda_params = {}
            lambda_star_dict[module_name] = None
        else:
            lambda_params = {'lambda_val' : lambda_star}
            lambda_star_dict[module_name] = float(lambda_star)

        # Weighted objectives are read by examples/experiments.ipynb's Uncertainty
        # section only, which looks up a single fixed budget (the smallest r
        # present, i.e. n_select) rather than sweeping the rule budget the way the
        # unweighted objectives' Bar Plots / Weighted Average Rule Length sections
        # do. Fitting them across all of n_rules_list wasted 6 of every 7 PEC fits,
        # so restrict weighted objectives to just that one budget.
        r_values = [fixed_parameters['n_select']] if obj_name.endswith('-weighted') else n_rules_list

        dsclust_params = {
            (r,) : {'n_select' : r, 'alpha_val' : alpha_val} | obj_params | lambda_params
            for r in r_values
        }
        dsclust_mod = DecisionSetMod(
            model = PEC,
            rules = rules,
            name = module_name
        )
        dscluster_module_list.append((dsclust_mod, dsclust_params))

        # Lazy-greedy counterpart: same objective/alpha/lambda as the distorted-greedy module
        # above (reusing `lambda_params`, including the degenerate-case empty dict -- so a
        # degenerate objective's lazy-greedy module lets PEC recompute the same lambda=0
        # fallback per fit, rather than being handed a stale lambda_star), but with
        # selection_algorithm='lazy-greedy' instead of 'distorted-greedy'. This is a genuine
        # second model (lazy-greedy carries no approximation-guarantee threshold on lambda), not
        # just the degenerate-case fallback.
        lazy_dsclust_params = {
            (r,) : {'n_select' : r, 'alpha_val' : alpha_val} | obj_params | lambda_params |
                   {'selection_algorithm' : 'lazy-greedy'}
            for r in r_values
        }
        lazy_dsclust_mod = DecisionSetMod(
            model = PEC,
            rules = rules,
            name = module_name + '; lazy-greedy'
        )
        dscluster_module_list.append((lazy_dsclust_mod, lazy_dsclust_params))

fixed_parameters['lambda_star'] = lambda_star_dict


####################################################################################################


baseline = kmeans_base
# Decision-Tree and IDS are handled separately below via `fit_stochastic_varying`
# (see "Stochastic module trials"), since they need to be refit per trial seed
# rather than dispatched once through `Experiment`'s joblib-parallel `run()`
# (whose worker processes do not inherit this script's seeded global NumPy
# state, which would make single-fit results irreproducible for exactly these
# randomized modules). CN2 is also handled separately below, via
# `fit_cn2_varying` -- its induction doesn't depend on the rule budget r at
# all, so sweeping it through Experiment's per-(module, param) joblib
# dispatch the way ExKMC/CBA are swept would rerun the same expensive
# induction once per r for an identical result each time.
module_list = [
    (exkmc_mod, exkmc_params),
    (cba_mod, cba_params),
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

####################################################################################################
# Objective re-scoring: ExKMC / CBA / PEC (distorted-greedy + lazy-greedy)
#
# `Experiment.run()` now retains each fit's raw decision set under
# `exp_results['modules'][name]['decisions']` (see DecisionSetMod/DecisionTreeMod.get_decisions()
# and Experiment._run_fit) -- rescore every one of them against each of SCORING_OBJECTIVE_NAMES's
# fixed lambda*/alpha, exactly like confidence.py's per-confidence-level scoring. This gives
# max_rules.py a TRUE 'objective' value per (module, r) instead of leaving the analysis notebook
# to reconstruct one from separately-reported reward/cost measurements.
#
# `objective_dict` (built above for PEC's own precomputed-cache fits) doubles as the scoring
# config here: score_objectives_by_r only reads 'objective_type'/'cluster_centers'/
# 'cluster_cost_method'/'data_to_center_distances' off it, so the extra 'precomputed_path' key
# is harmlessly ignored.
_scoring_modules_decisions = {
    name: exp_results['modules'][name]['decisions']
    for name in (
        ['ExKMC', 'CBA'] +
        [
            f'dscluster; {obj_name}; ensemble{suffix}'
            for obj_name in SCORING_OBJECTIVE_NAMES
            for suffix in ('', '; lazy-greedy')
        ]
    )
    if name in exp_results['modules']
}
_objective_scores = score_objectives_by_r(
    modules_decisions = _scoring_modules_decisions,
    objective_names = SCORING_OBJECTIVE_NAMES,
    objective_config = objective_dict,
    selected_alpha_dict = fixed_parameters['alpha'],
    lambda_dict = lambda_star_dict,
    data = data,
    y = kmeans_labels,
    n_select = fixed_parameters['n_select'],
)
for _name, _obj_by_name in _objective_scores.items():
    exp_results['modules'][_name]['objective'] = _obj_by_name

####################################################################################################
# Stochastic module trials
#
# Decision-Tree and IDS each have a fitted solution that depends on randomness.
# Rather than record one arbitrarily-seeded fit, each is refit once per seed in
# `trial_seeds` and the results across trials are aggregated into
# {'mean', 'std', 'values'} via `aggregate_trials` (see experiments/modules.py).
# This runs single-process (not through `Experiment`'s joblib dispatch) specifically
# so each trial's explicit seed is what controls its randomness.

def _seed_and_fit(mod, params, trial_seed):
    """
    Fits `mod` for one trial. Sets the trial's explicit seed both as a fitting
    parameter (for classes that thread it through properly, e.g. IDS,
    DecisionTree, ShallowTree) AND as the global NumPy seed immediately before
    fit() -- some dependencies (e.g. ExplanationTree's compiled Cython splitter,
    see explanation_tree.py's `random_state` docstring caveat) still read the
    global RNG state directly and aren't fully parameterized by a passed-in
    random_state alone. This call runs single-process, so setting the global
    seed here is safe and sufficient for reproducibility (unlike doing so at
    the top of the script, which does not survive joblib worker dispatch).
    """
    np.random.seed(trial_seed)
    mod.update_fitting_params(params)
    return mod.fit(data, kmeans_labels)


def _module_trial_result(mod, assignments, measurement_fns):
    data_to_rule, rule_to_cluster, data_to_cluster = assignments
    return {
        'lambda': mod.lambda_val if hasattr(mod, 'lambda_val') else None,
        'lambda_n_rules': getattr(mod, 'n_available_decisions', np.nan),
        'max-rule-length': mod.max_rule_length,
        'sum-rule-length': mod.sum_rule_length,
        'weighted-avg-length': mod.weighted_average_rule_length,
    } | {
        fn.name: fn(data_to_rule, rule_to_cluster, data_to_cluster)
        for fn in measurement_fns
    }


def _score_decisions_all_objectives(decisions):
    """
    Scores one fitted module's decisions against each of SCORING_OBJECTIVE_NAMES's fixed
    lambda*/alpha (see the ExKMC/CBA/PEC re-scoring pass above), for models fit outside
    `Experiment.run()` (Decision-Tree/IDS per trial, CN2 per r) and so not covered by
    `score_objectives_by_r` there. Returns {objective_name: score}, NaN wherever that
    objective's lambda* is degenerate or `decisions` is empty/None.
    """
    scores = {}
    for obj_name in SCORING_OBJECTIVE_NAMES:
        base_name = f'dscluster; {obj_name}; ensemble'
        lambda_val = lambda_star_dict.get(base_name)
        if not decisions or lambda_val is None or (
            isinstance(lambda_val, float) and np.isnan(lambda_val)
        ):
            scores[obj_name] = np.nan
            continue
        cfg = objective_dict[obj_name]
        scores[obj_name] = score_decision_set(
            decisions,
            data,
            kmeans_labels,
            n_select = fixed_parameters['n_select'],
            objective_type = cfg['objective_type'],
            alpha_val = fixed_parameters['alpha'][base_name],
            lambda_val = lambda_val,
            cluster_centers = cfg.get('cluster_centers'),
            cluster_cost_method = cfg.get('cluster_cost_method', 'kmeans'),
            weights = cfg.get('weights'),
            data_to_center_distances = cfg.get('data_to_center_distances'),
        )
    return scores


def fit_stochastic_varying(mod, params_by_r, trial_seeds, measurement_fns, seed_key='random_state'):
    """
    Refits `mod` once per (rule-count r, trial seed) pair -- for modules whose
    output genuinely varies with the rule-count budget r -- and aggregates
    results across trials for each r.

    `rule-source-counts` (a {source: count} dict, from `DecisionSetMod`) is collected separately
    from the rest of the per-trial fields rather than run through `aggregate_trials`: that helper
    computes np.mean/np.std over trial values, which isn't meaningful for a dict-valued metric.
    It's instead stored per-r as the raw list of per-trial breakdowns.

    `objective` (the true PEC-objective score per SCORING_OBJECTIVE_NAMES entry, via
    `_score_decisions_all_objectives`) is likewise collected separately and reduced to
    {'mean', 'std', 'values'} per objective, mirroring confidence.py's aggregation for these
    same two stochastic modules.
    """
    result = (
        {'lambda': {}, 'lambda_n_rules': {}, 'max-rule-length': {},
         'sum-rule-length': {}, 'weighted-avg-length': {}, 'rule-source-counts': {},
         'objective': {obj_name: {} for obj_name in SCORING_OBJECTIVE_NAMES}} |
        {fn.name: {} for fn in measurement_fns}
    )
    for r, base_params in params_by_r.items():
        trial_dicts = []
        rule_source_counts_by_trial = []
        trial_objective_dicts = {obj_name: [] for obj_name in SCORING_OBJECTIVE_NAMES}
        for trial_seed in trial_seeds:
            assignments = _seed_and_fit(mod, dict(base_params) | {seed_key: trial_seed}, trial_seed)
            trial_dicts.append(_module_trial_result(mod, assignments, measurement_fns))
            rule_source_counts_by_trial.append(getattr(mod, 'rule_source_counts', None))
            decisions = mod.get_decisions() if hasattr(mod, 'get_decisions') else None
            for obj_name, score in _score_decisions_all_objectives(decisions).items():
                trial_objective_dicts[obj_name].append(score)
        for key, agg_val in aggregate_trials(trial_dicts).items():
            result[key][r] = agg_val
        result['rule-source-counts'][r] = {'values': rule_source_counts_by_trial}
        for obj_name, vals in trial_objective_dicts.items():
            result['objective'][obj_name][r] = {
                'mean': float(np.nanmean(vals)) if vals else np.nan,
                'std': float(np.nanstd(vals)) if vals else np.nan,
                'values': vals,
            }
    return result


print(f"Fitting stochastic modules across {n_trials} trials each...")
exp_results['modules']['Decision-Tree'] = fit_stochastic_varying(
    decision_tree_mod, decision_tree_params_by_r, trial_seeds, measurement_fns,
    seed_key='random_state'
)
exp_results['modules']['IDS'] = fit_stochastic_varying(
    ids_mod, ids_params_by_r, trial_seeds, measurement_fns,
    seed_key='random_state'
)
print("Stochastic modules done.")

####################################################################################################
# CN2
#
# CN2's beam-search induction doesn't depend on n_select -- only the post-hoc
# truncation to the first n_select rules does (see cn2.py's induce()/
# finalize() split). Sweeping it through Experiment's per-(module, param)
# joblib dispatch, the way ExKMC/CBA are swept, would rerun the same
# induction from scratch once per rule budget in n_rules_list for an
# otherwise-identical result. CN2 is deterministic (no seed dependence), so
# it's fit outside `Experiment.run()` here: induce once, then finalize +
# measure cheaply per budget.

def fit_cn2_varying(n_rules_list, measurement_fns):
    result = (
        {'lambda': {}, 'lambda_n_rules': {}, 'max-rule-length': {},
         'sum-rule-length': {}, 'weighted-avg-length': {},
         'objective': {obj_name: {} for obj_name in SCORING_OBJECTIVE_NAMES}} |
        {fn.name: {} for fn in measurement_fns}
    )
    cn2 = CN2()
    cn2.induce(data, kmeans_labels)
    n_unique = len(unique_labels(kmeans_labels))
    for r in n_rules_list:
        cn2.finalize(r)
        assignments = (
            cn2.get_data_to_rules_assignment(data),
            cn2.get_rules_to_clusters_assignment(n_labels=n_unique),
            labels_to_assignment(cn2.predict(data), n_labels=n_unique),
        )
        trial_result = {
            'lambda': np.nan,
            'lambda_n_rules': np.nan,
            'max-rule-length': cn2.max_rule_length,
            'sum-rule-length': cn2.get_sum_of_rule_lengths(),
            'weighted-avg-length': cn2.get_weighted_average_rule_length(data),
        } | {
            fn.name: fn(*assignments) for fn in measurement_fns
        }
        for key, val in trial_result.items():
            result[key][r] = val
        for obj_name, score in _score_decisions_all_objectives(cn2.decision_set).items():
            result['objective'][obj_name][r] = score
    return result


print("Fitting CN2 (induce once, finalize per rule budget)...")
exp_results['modules']['CN2'] = fit_cn2_varying(n_rules_list, measurement_fns)
print("CN2 done.")

exp.save_results(outfile, outfile_ref)
end = time.time()
print("Experiment time:", end - start)


####################################################################################################
