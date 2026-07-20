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
from experiments.mnist.config import (
    SEED, N_CLUSTERS, N_SELECT_DEFAULT, MAX_RULES, SHALLOW_TREE_DEPTH_FACTOR,
    N_FOREST, FOREST_MAX_DEPTH, CAR_MIN_SUPPORT, CAR_MIN_CONFIDENCE,
    CAR_MAX_RULE_LENGTH, CONFIDENCE_DEFAULT, CPU_COUNT, OUTFILE_REF, RULES_DIR,
    ALPHAS_DIR, MAX_RULES_DIR,
)

####################################################################################################
# This is the PEC-lazy-greedy-only counterpart to max_rules.py -- a further split of the same
# per-model split max_rules_pec.py belongs to (see max_rules_combine.py, which now also merges in
# whichever of max_rules_{cba,cn2,dtree,ids,pec,pec_lazy,exkmc}.py have completed). It fits ONLY
# PEC's lazy-greedy selection algorithm, as a separate model from the distorted-greedy PEC fit in
# max_rules_pec.py, so that lazy-greedy results can be added to an already-completed run without
# refitting anything else. max_rules.py itself also fits lazy-greedy directly now (see its
# dscluster_module_list loop) -- this script exists purely so you don't have to rerun max_rules.py
# (or max_rules_pec.py) in full just to pick up lazy-greedy results retroactively.
#
# NOTE: this uses its own outfile_ref ('_pec_lazy', not '_pec' or '_dscluster') so it never
# collides with max_rules.py's or max_rules_pec.py's own output files if all happen to exist on
# disk at once.

import os
import json
import pickle
import numpy as np
from intercluster import *
from intercluster.decision_trees import *
from intercluster.decision_sets import *
from intercluster.decision_sets.objectives import *
from intercluster.decision_sets.mining import *
from intercluster.measurements import *


# Prevents memory leakage for KMeans:
os.environ["OMP_NUM_THREADS"] = "1"

experiment_cpu_count = CPU_COUNT

# REMINDER: The seed should only be initialized here. It should NOT
# within the parameters of any sub-function or class (except for select
# baseline experiments like KMeans), since these will
# reset the seed each time they are given one.
seed = SEED

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
data, data_labels, feature_labels, scaler = load_preprocessed_mnist()
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
    'car_max_rule_length': CAR_MAX_RULE_LENGTH, # (really means 4 by pyfim convention)
    'filter_confidence': CONFIDENCE_DEFAULT,
    'seed': seed,
}

n_rules_list = list(range(fixed_parameters['n_clusters'], fixed_parameters['max_rules'] + 1))

np.random.seed(fixed_parameters['seed'])

# Baseline KMeans
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
outfile_ref = '_pec_lazy' + OUTFILE_REF

####################################################################################################
# Load pre-mined rules:

ensemble_rules = load_rules(RULES_DIR + f'ensemble_rules{OUTFILE_REF}.pkl')

rule_miner_dict = {
    'ensemble': (None, ensemble_rules, None),
}

####################################################################################################
# Objectives for Decision Set Clustering:

objective_dict = {
    'coverage-mistake': {
        'objective_type': 'coverage-mistake',
        'selection_algorithm': 'distorted-greedy',
        'precomputed_path': os.path.join(
            decision_info_dict_directory, f'mistake_info_dict{OUTFILE_REF}.pkl.gz'
        )
    },
    'coverage-cost': {
        'objective_type': 'coverage-cost',
        'cluster_centers': kmeans_base.centers,
        'cluster_cost_method': 'kmeans',
        'selection_algorithm': 'distorted-greedy',
        'precomputed_path': os.path.join(
            decision_info_dict_directory, f'cost_info_dict{OUTFILE_REF}.pkl.gz'
        )
    },
    'coverage-pairwise-distance': {
        'objective_type': 'coverage-pairwise-distance',
        'selection_algorithm': 'distorted-greedy',
        'precomputed_path': os.path.join(
            decision_info_dict_directory, f'pairwise_distance_info_dict{OUTFILE_REF}.pkl.gz'
        )
    },
    'coverage-mistake-weighted': {
        'objective_type': 'coverage-mistake',
        'weights': weights,
        'selection_algorithm': 'distorted-greedy',
        'precomputed_path': os.path.join(
            decision_info_dict_directory, f'mistake_info_dict{OUTFILE_REF}.pkl.gz'
        )
    },
    'coverage-cost-weighted': {
        'objective_type': 'coverage-cost',
        'cluster_centers': kmeans_base.centers,
        'weights': weights,
        'cluster_cost_method': 'kmeans',
        'selection_algorithm': 'distorted-greedy',
        'precomputed_path': os.path.join(
            decision_info_dict_directory, f'cost_info_dict{OUTFILE_REF}.pkl.gz'
        )
    },
    'coverage-pairwise-distance-weighted': {
        'objective_type': 'coverage-pairwise-distance',
        'weights': weights,
        'selection_algorithm': 'distorted-greedy',
        'precomputed_path': os.path.join(
            decision_info_dict_directory, f'pairwise_distance_info_dict{OUTFILE_REF}.pkl.gz'
        )
    },
}

####################################################################################################
# Decision Set Clustering Modules (lazy-greedy only):
#
# lambda* is still probed ONCE per objective via a throwaway distorted-greedy PEC fit -- exactly
# as max_rules.py/max_rules_pec.py do -- purely so the lazy-greedy modules below can be fit at the
# *same* lambda the distorted-greedy modules use elsewhere. This makes lazy-greedy vs
# distorted-greedy a genuine apples-to-apples comparison of the two selection algorithms rather
# than two differently-tuned models. See max_rules.py's identical probe block for the full
# rationale on why this is cheap (reuses the precomputed coverage/cost caches).

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
        # so leave those objectives on the original per-fit path (lambda_val=None lets PEC
        # recompute the same fallback itself).
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
        # do. Fitting them across all of n_rules_list wasted most of every PEC
        # fit, so restrict weighted objectives to just that one budget.
        r_values = [fixed_parameters['n_select']] if obj_name.endswith('-weighted') else n_rules_list

        # Only the lazy-greedy module is built here -- the distorted-greedy counterpart is
        # max_rules_pec.py's job. Named with an '; lazy-greedy' suffix so it never collides with
        # max_rules_pec.py's (or max_rules.py's) distorted-greedy module of the same objective.
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
module_list = dscluster_module_list

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
