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
from experiments.profiling import stamp, stamp_reset
from experiments.cli_utils import conf_tag, parse_experiment_args
from experiments.lambda_grid import build_shared_grids, load_lambda_grids, save_lambda_grids
stamp_reset()

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

args = parse_experiment_args(confidence_default=0.75, cpu_count_default=6, grid_flags=True)
confidence_threshold = args.confidence
tag = conf_tag(confidence_threshold)
experiment_cpu_count = args.cpu_count

# REMINDER: The seed should only be initialized here. It should NOT
# within the parameters of any sub-function or class (except for select
# baseline experiments like KMeans), since these will
# reset the seed each time they are given one.
# Classes with their own internal randomness (IDS, ExplanationTree, DecisionTree,
# ShallowTree) accept an explicit random_state / kmeans_random_state instead of
# relying on this global seed -- see `trial_seeds` below, which derives one seed
# per trial so those modules can be refit across multiple trials and their results
# reported as mean/std rather than a single, arbitrarily-seeded point estimate.
seed = 342

# Number of independent random-seed trials used to evaluate stochastic modules
# (IDS, Exp-Tree, Decision-Tree, Shallow-Tree). Deterministic modules (PEC, ExKMC,
# CN2, CBA, WRA) are fit once, since repeating them would just reproduce the same
# result. `trial_seeds` is derived deterministically from `seed` so that re-running
# this script reproduces the exact same set of trials.
n_trials = 10
trial_seeds = [seed + i for i in range(n_trials)]

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
data, data_labels, feature_labels, scaler = load_preprocessed_ansio()
stamp("data loaded")
data = _memoryview_safe(data)
n,d = data.shape

fixed_parameters = {
    'n': n,
    'd': d,
    'n_clusters': 5,
    'n_select': 5,
    'max_rules': 11,
    'shallow_tree_depth_factor': 0.03,
    'n_forest': 10,
    'forest_max_depth': 4,
    'car_min_support': 0.025,
    'car_min_confidence': 0.75,
    'car_max_rule_length': 2, # (really means 4 by pyfim convention)
    'filter_confidence': confidence_threshold,
    'seed': seed,
    'n_trials': n_trials,
    'trial_seeds': trial_seeds,
}

# Unlike max_rules.py (which sweeps the rule budget n_select over n_rules_list),
# this experiment fixes the rule budget at fixed_parameters['n_select'] -- the same
# budget alphas.py used to tune alpha -- and instead sweeps PEC's lambda hyperparameter.
n_select = fixed_parameters['n_select']

np.random.seed(fixed_parameters['seed'])

# Baseline KMeans
kmeans_base = KMeansBase(n_clusters = fixed_parameters['n_clusters'], random_seed = fixed_parameters['seed'])
kmeans_assignment = kmeans_base.assign(data)
kmeans_labels = kmeans_base.labels
stamp("kmeans clustering")

# Weights for uncertainty objectives
weights = distance_ratio_score(data, kmeans_base.centers)
fixed_parameters['weights'] = weights.tolist()

# Alpha values for objectives:
with open(f"data/experiments/aniso/alphas/selected_alphas_resub_conf_{tag}.json") as f:
    selected_alpha_dict = json.load(f)
fixed_parameters['alpha'] = selected_alpha_dict

decision_info_dict_directory = 'data/experiments/aniso/rules/'

outfile = 'data/experiments/aniso/lambda/'
outfile_ref = f'_resub_conf_{tag}'

####################################################################################################
# Load pre-mined rules:


ensemble_rules = load_rules(f'data/experiments/aniso/rules/ensemble_rules_conf_{tag}.pkl')

with open(f'data/experiments/aniso/rules/ensemble_labels_conf_{tag}.pkl', 'rb') as f:
    ensemble_labels = pickle.load(f)

rule_miner_dict = {
    'ensemble': (None, ensemble_rules, None),
}

####################################################################################################
# Objectives for Decision Set Clustering:
#
# Built per confidence tag (rather than once for this run's tag) because --emit-grid probes
# lambda* for every threshold in the sweep, and each threshold has its own precomputed
# coverage/cost caches. Only the precomputed_path varies with the tag.


def objective_dict_for_tag(t):
    return {
        'coverage-mistake': {
            'objective_type': 'coverage-mistake',
            'precomputed_path': os.path.join(
                decision_info_dict_directory, f'mistake_info_dict_conf_{t}.pkl.gz'
            )
        },
        'coverage-cost': {
            'objective_type': 'coverage-cost',
            'cluster_centers': kmeans_base.centers,
            'cluster_cost_method': 'kmeans',
            'precomputed_path': os.path.join(
                decision_info_dict_directory, f'cost_info_dict_conf_{t}.pkl.gz'
            )
        },
        'coverage-pairwise-distance': {
            'objective_type': 'coverage-pairwise-distance',
            'precomputed_path': os.path.join(
                decision_info_dict_directory, f'pairwise_distance_info_dict_conf_{t}.pkl.gz'
            )
        },
        # The '-weighted' variants (coverage-mistake-weighted, coverage-cost-weighted,
        # coverage-pairwise-distance-weighted) are intentionally left out of this sweep; they
        # differ only by passing 'weights': weights alongside the same objective_type.
    }


objective_dict = objective_dict_for_tag(tag)

####################################################################################################
# Lambda grid:
#
# PEC's lambda* -- the minimum lambda for which the distorted-greedy guarantee holds -- shrinks as
# the filter confidence rises (it is a max over coverage/cost ratios across the rule pool, and a
# stricter filter can only remove candidates for that max). Sweeping each confidence over its own
# [0, 2 * lambda*] therefore gave the confidences non-overlapping x-axes: aniso's coverage-cost
# lambda* is 0.329 at confidence 0.25/0.50 but 0.116 at 0.75, so the 0.75 sweep stopped at 0.231
# while the others ran to 0.659.
#
# Instead, ONE grid per objective is shared across every confidence: it spans
# [0, 2 * max_c lambda*_c] and contains every confidence's lambda*_c as an exact grid point. It is
# built once by the --emit-grid barrier stage (which needs every threshold's selected alphas on
# disk, since lambda* depends on alpha) and read back by each per-confidence run, so all runs key
# their results off bit-identical lambda floats and line up when plotted together.
#
# Each run still fits distorted-greedy only for lambda >= its OWN lambda*, which stays a clean cut
# because that anchor is on the grid. See experiments/lambda_grid.py.

# Matches alphas.py's n_compare convention (25). Kept high because the shared grid spans
# [0, 2 * max_c lambda*_c] rather than a single confidence's [0, 2 * lambda*]: at a fixed point
# count, the wider span thins out resolution exactly around the smaller lambda*_c values. The
# other datasets' lambda.py still use 10 -- aniso is the small/fast one (~10s per run), so it can
# afford the denser sweep.
n_lambda_points = 25
lambda_grid_path = os.path.join(outfile, 'lambda_grid_resub.json')

if args.emit_grid:
    lambda_star_by_module = {}
    for c in args.confidence_thresholds:
        c_tag = conf_tag(c)
        c_rules = load_rules(f'data/experiments/aniso/rules/ensemble_rules_conf_{c_tag}.pkl')
        with open(f'data/experiments/aniso/alphas/selected_alphas_resub_conf_{c_tag}.json') as f:
            c_alphas = json.load(f)

        for obj_name, obj_params in objective_dict_for_tag(c_tag).items():
            module_name = f'dscluster; {obj_name}; ensemble'
            probe_params = {
                'n_select': n_select,
                'alpha_val': c_alphas[module_name],
            } | obj_params | {'lambda_val': None, 'selection_algorithm': 'distorted-greedy'}
            probe = PEC(rules=c_rules, **probe_params)
            # compute_lambda_star does everything fit() would, minus the selection pass.
            lambda_star = float(probe.compute_lambda_star(data, kmeans_labels))
            lambda_star_by_module.setdefault(module_name, {})[c_tag] = lambda_star
            print(f"  lambda* [conf={c}] {module_name}: {lambda_star}")

    grids = build_shared_grids(lambda_star_by_module, n_lambda_points)
    save_lambda_grids(lambda_grid_path, grids, n_lambda_points, args.confidence_thresholds)
    stamp("lambda* probe fits (all thresholds)")
    print(f"\nShared lambda grid written to {lambda_grid_path}")
    sys.exit(0)

lambda_grid_dict, lambda_star_dict, lambda_star_by_conf_dict = load_lambda_grids(
    lambda_grid_path, tag
)
fixed_parameters['lambda_star'] = lambda_star_dict
fixed_parameters['lambda_star_by_conf'] = lambda_star_by_conf_dict
fixed_parameters['lambda_grid'] = lambda_grid_dict
fixed_parameters['n_lambda_points'] = n_lambda_points

# Union of every objective's lambda grid -- used to broadcast each comparison
# model's single fit (which doesn't depend on lambda or the objective at all)
# across every lambda value any objective's plot might need to look up.
all_lambda_values = tuple(
    sorted(set().union(*(set(g) for g in lambda_grid_dict.values())))
)

####################################################################################################
# Comparison Modules:
#
# NOTE on reproducibility: Decision-Tree, Exp-Tree, Shallow-Tree, and IDS all have
# inherent randomness in their fitted solution (sklearn tree tie-breaking, heap
# tie-breaking, internal KMeans re-initialization, and randomized-greedy/SLS
# selection respectively). Rather than fit each once under the single global
# `seed`, these four are refit across `trial_seeds` further below (see
# "Stochastic module trials") and their results are recorded as mean/std/values
# instead of a single point estimate.
#
# NOTE on this experiment vs. max_rules.py: none of the comparison models below take
# a lambda parameter, and their selected rules don't change as PEC's lambda changes.
# So -- unlike max_rules.py, which refits each comparison model once per rule budget r
# -- every comparison model here is fit exactly ONCE, at the fixed `n_select` budget,
# and its result is simply broadcast across every lambda value in the sweep.

# Decision Tree
decision_tree_shared_params = {'max_leaf_nodes': n_select}
decision_tree_mod = DecisionTreeMod(
    model = DecisionTree,
    name = 'Decision-Tree'
)


# Explanation Tree
# (ExplanationTree's leaf count is fixed at num_clusters, independent of any rule
# budget, so its single fit's result is recorded under every lambda label.)
exp_tree_shared_params = {'num_clusters' : fixed_parameters['n_clusters']}
exp_tree_mod = DecisionTreeMod(
    model = ExplanationTree,
    name = 'Exp-Tree'
)


# ExKMC
exkmc_shared_params = {
    'k' : fixed_parameters['n_clusters'],
    'kmeans': kmeans_base.clustering,
    'max_leaf_nodes': n_select
}
exkmc_mod = DecisionTreeMod(
    model = ExkmcTree,
    name = 'ExKMC'
)


# Shallow Tree
# (ShallowTree's structure is controlled by depth_factor, not by a rule-count/
# max_leaf_nodes parameter, so its single fit's result is recorded under every
# lambda label.)
shallow_tree_shared_params = {
    'n_clusters' : fixed_parameters['n_clusters'],
    'depth_factor' : fixed_parameters['shallow_tree_depth_factor'],
}
shallow_tree_mod = DecisionTreeMod(
    model = ShallowTree,
    name = 'Shallow-Tree'
)

# WRA:
wra_shared_params = {'n_select': n_select}
wra_mod = DecisionSetMod(
    model=WRABaseline,
    rules=ensemble_rules,
    rule_labels=ensemble_labels,
    name='WRA'
)

wra_weighted_shared_params = {'n_select': n_select, 'weights': weights}
wra_weighted_mod = DecisionSetMod(
    model=WRABaseline,
    rules=ensemble_rules,
    rule_labels=ensemble_labels,
    name='WRA-weighted'
)


# CBA:
cba_shared_params = {'n_select': n_select}
cba_mod = DecisionSetMod(
    model=CBA,
    rules=ensemble_rules,
    rule_labels=ensemble_labels,
    name='CBA'
)


# CN2:
cn2_shared_params = {'n_select': n_select}
cn2_mod = DecisionSetMod(
    model=CN2,
    rules=None,
    name='CN2'
)


# IDS:
with open(f'data/experiments/aniso/rules/ids_lambdas_conf_{tag}.json') as f:
    ids_lambdas = json.load(f)
if isinstance(ids_lambdas, dict):
    ids_lambdas = list(ids_lambdas.values())

_ids_cache_path = f'data/experiments/aniso/rules/ids_coverage_cache_ensemble_conf_{tag}.pkl'
if os.path.exists(_ids_cache_path):
    print("Loading pre-built IDS cache...")
    with open(_ids_cache_path, 'rb') as f:
        ids_cache = pickle.load(f)
    print(f"IDS cache loaded ({len(ids_cache.decisions)} decisions).")
    stamp("IDS cache loaded from disk")
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
    stamp("IDS cache BUILT (first-time, no cache file)")

ids_shared_params = {
    'n_select': n_select,
    'lambdas': ids_lambdas,
    'cache': ids_cache,
    'optimizer': 'random_greedy',
}
ids_mod = DecisionSetMod(
    model=IDS,
    rules=ensemble_rules,
    rule_labels=ensemble_labels,
    name='IDS'
)

####################################################################################################
# Decision Set Clustering Modules:
#
# Two modules per objective: 'lazy-greedy' is valid (and recorded) across the full shared grid,
# while 'distorted-greedy' is only valid -- and thus only fit/recorded -- for lambda >= this
# confidence's own lambda*. Since the shared grid spans [0, 2 * max_c lambda*_c], a confidence
# whose lambda* is below the max gets MORE distorted-greedy points than it did under its own
# [0, 2 * lambda*] grid, and its high-lambda tail may go degenerate (the objective g - lambda*h
# turns cost-dominated, so selections empty out). That is the intended cost of a common x-axis.
#
# lambda_star here is the exact anchor the grid was built from (read from the grid JSON, not
# re-probed), so the `l >= lambda_star` cut lands on a real grid point rather than missing it by
# a last-ULP mismatch.

dscluster_module_list = []
for obj_name, obj_params in objective_dict.items():
    for rule_miner_name, (rule_miner, rules, rule_labels) in rule_miner_dict.items():
        module_name = f'dscluster; {obj_name}; {rule_miner_name}'
        alpha_val = fixed_parameters['alpha'][module_name]
        base_params = {'n_select': n_select, 'alpha_val': alpha_val} | obj_params

        lambda_star = lambda_star_dict[module_name]
        lambda_grid = lambda_grid_dict[module_name]

        lazy_params = {
            (l,): base_params | {'lambda_val': l, 'selection_algorithm': 'lazy-greedy'}
            for l in lambda_grid
        }
        lazy_mod = DecisionSetMod(
            model = PEC,
            rules = rules,
            name = f'{module_name}; lazy-greedy'
        )
        dscluster_module_list.append((lazy_mod, lazy_params))

        distorted_params = {
            (l,): base_params | {'lambda_val': l, 'selection_algorithm': 'distorted-greedy'}
            for l in lambda_grid if l >= lambda_star
        }
        distorted_mod = DecisionSetMod(
            model = PEC,
            rules = rules,
            name = f'{module_name}; distorted-greedy'
        )
        dscluster_module_list.append((distorted_mod, distorted_params))


####################################################################################################


baseline = kmeans_base
# Decision-Tree, Exp-Tree, Shallow-Tree, and IDS are handled separately below via
# `fit_stochastic_shared` (see "Stochastic module trials"), since they need to be
# refit per trial seed rather than dispatched once through `Experiment`'s
# joblib-parallel `run()` (whose worker processes do not inherit this script's
# seeded global NumPy state, which would make single-fit results irreproducible
# for exactly these randomized modules). Each is fit once per trial seed here
# (not once per lambda value) and its per-trial result is broadcast across
# `all_lambda_values`, exactly like the deterministic comparison modules below.
module_list = [
    (exkmc_mod, {all_lambda_values: exkmc_shared_params}),
    (wra_mod, {all_lambda_values: wra_shared_params}),
    (wra_weighted_mod, {all_lambda_values: wra_weighted_shared_params}),
    (cba_mod, {all_lambda_values: cba_shared_params}),
    (cn2_mod, {all_lambda_values: cn2_shared_params}),
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
stamp("setup complete -> starting exp.run")
exp_results = exp.run()
stamp("exp.run: all PEC + comparison module fits")

####################################################################################################
# Stochastic module trials
#
# Decision-Tree, Exp-Tree, Shallow-Tree, and IDS each have a fitted solution that
# depends on randomness. Rather than record one arbitrarily-seeded fit, each is
# refit once per seed in `trial_seeds` and the results across trials are aggregated
# into {'mean', 'std', 'values'} via `aggregate_trials` (see experiments/modules.py).
# This runs single-process (not through `Experiment`'s joblib dispatch) specifically
# so each trial's explicit seed is what controls its randomness.
#
# None of these four vary with lambda (only PEC does), so -- unlike max_rules.py,
# where Decision-Tree and IDS varied with the rule budget r -- all four are handled
# with `fit_stochastic_shared`: fit once per trial seed, and the trial-aggregated
# result is broadcast across every value in `all_lambda_values`.

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


def fit_stochastic_shared(mod, shared_params, r_values, trial_seeds, measurement_fns, seed_key='random_state'):
    """
    Refits `mod` once per trial seed -- for modules whose output does not vary
    with the swept lambda value -- and broadcasts the trial-aggregated result
    across every value in `r_values` (matching the pre-existing convention for
    these modules in max_rules.py).
    """
    result = (
        {'lambda': {}, 'lambda_n_rules': {}, 'max-rule-length': {},
         'sum-rule-length': {}, 'weighted-avg-length': {}} |
        {fn.name: {} for fn in measurement_fns}
    )
    trial_dicts = []
    for trial_seed in trial_seeds:
        assignments = _seed_and_fit(mod, dict(shared_params) | {seed_key: trial_seed}, trial_seed)
        trial_dicts.append(_module_trial_result(mod, assignments, measurement_fns))
    aggregated = aggregate_trials(trial_dicts)
    for r in r_values:
        for key, agg_val in aggregated.items():
            result[key][r] = agg_val
    return result


print(f"Fitting stochastic modules across {n_trials} trials each...")
exp_results['modules']['Decision-Tree'] = fit_stochastic_shared(
    decision_tree_mod, decision_tree_shared_params, all_lambda_values, trial_seeds, measurement_fns,
    seed_key='random_state'
)
exp_results['modules']['Exp-Tree'] = fit_stochastic_shared(
    exp_tree_mod, exp_tree_shared_params, all_lambda_values, trial_seeds, measurement_fns,
    seed_key='random_state'
)
exp_results['modules']['Shallow-Tree'] = fit_stochastic_shared(
    shallow_tree_mod, shallow_tree_shared_params, all_lambda_values, trial_seeds, measurement_fns,
    seed_key='kmeans_random_state'
)
exp_results['modules']['IDS'] = fit_stochastic_shared(
    ids_mod, ids_shared_params, all_lambda_values, trial_seeds, measurement_fns,
    seed_key='random_state'
)
print("Stochastic modules done.")
stamp("stochastic trials (trees + IDS x n_trials)")

exp.save_results(outfile, outfile_ref)
stamp("results saved")
end = time.time()
print("Experiment time:", end - start)


####################################################################################################
