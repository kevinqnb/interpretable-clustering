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
from experiments.mnist.config import (
    SEED, N_CLUSTERS, N_SELECT_DEFAULT, OUTFILE_REF, RULES_DIR, ALPHAS_DIR,
    IDS_ALT_OBJECTIVE_TYPE,
)

####################################################################################################

import os
import json
import time
import pickle
import numpy as np
import pandas as pd
from intercluster import *
from intercluster.decision_sets import *
from intercluster.decision_sets.ids import IDSCoverageCache
from sklearn.model_selection import train_test_split

os.environ["OMP_NUM_THREADS"] = "1"

# REMINDER: The seed should only be initialized here. IDS is given this seed
# explicitly below (random_state=seed) rather than relying on this global
# np.random.seed call, so this lambda search is reproducible independent of
# global NumPy state.
seed = SEED

# Suffix distinguishing this alternate pipeline's own outputs from the AUC-based
# ids_lambda_search.py's -- inputs (ensemble rules/labels, selected alphas, the
# coverage cache) are shared and stay keyed by the plain OUTFILE_REF.
OUTFILE_REF_ALT = OUTFILE_REF + '_ids_alt'

####################################################################################################
# Data + clustering

data, data_labels, feature_labels, scaler = load_preprocessed_mnist()
n, d = data.shape

np.random.seed(seed)

kmeans_base = KMeansBase(n_clusters=N_CLUSTERS, random_seed=seed)
kmeans_base.assign(data)
kmeans_labels = kmeans_base.labels
y_flat = flatten_labels(kmeans_labels)

n_select = N_SELECT_DEFAULT

####################################################################################################
# Load pre-mined ensemble rules

ensemble_rules = load_rules(RULES_DIR + f'ensemble_rules{OUTFILE_REF}.pkl')
print(f"Ensemble rule set size: {len(ensemble_rules)}")

with open(RULES_DIR + f'ensemble_labels{OUTFILE_REF}.pkl', 'rb') as f:
    ensemble_labels = pickle.load(f)

ids_rules  = ensemble_rules
ids_labels = ensemble_labels

####################################################################################################
# Load (or build) IDSCoverageCache -- identical cache to the AUC-based search, since coverage
# doesn't depend on the scoring mode used to pick lambdas. Reused, not rebuilt.

cache_path = RULES_DIR + f'ids_coverage_cache_ensemble{OUTFILE_REF}.pkl'
if os.path.exists(cache_path):
    print("Loading pre-built IDS cache...")
    with open(cache_path, 'rb') as f:
        ids_cache = pickle.load(f)
    print(f"IDS cache loaded ({len(ids_cache.decisions)} decisions).")
else:
    print(f"Building coverage cache for {len(ids_rules)} rules...")
    t0 = time.time()
    ids_cache = IDSCoverageCache.from_rules(ids_rules, ids_labels, data, kmeans_labels)
    print(f"Cache ready: {len(ids_cache.decisions)} valid decisions in {time.time() - t0:.1f}s")
    with open(cache_path, 'wb') as f:
        pickle.dump(ids_cache, f)
    print(f"Cache saved to {cache_path}")

####################################################################################################
# Subsample points for the lambda search only. The cache built and saved above stays full-data --
# max_rules.py/confidence.py load ids_coverage_cache_ensemble*.pkl expecting it to cover every
# point, falling back to rebuilding it from the full dataset if the file is missing. Coordinate
# ascent's dominant per-candidate cost scales with N (the |S| x N matmul inside
# IDSObjective.evaluate, run inside RandomGreedyOptimizer for every candidate at every round), so
# searching on a smaller stratified subsample cuts that cost substantially without reducing the
# search's resolution (same precision/iterations). point_subset() reuses the already-computed
# antecedent/correct masks -- no rule is re-evaluated against X.
#
# Unlike ids_lambda_search.py's held-out-AUC path (which only ever reads cache masks, so a
# subsampled cache paired with the full data/labels below is harmless), the PEC-objective scorer
# re-evaluates rules against whatever X/y are passed to fit() (see score_decision_set() in
# intercluster/decision_sets/objectives/scoring.py) -- so data/labels passed to fit() below are
# subsampled to the same indices as the cache, keeping the optimizer and scorer consistent.

_LAMBDA_SEARCH_SUBSAMPLE = 0.2

all_point_idx = np.arange(ids_cache.N)
try:
    subsample_idx, _ = train_test_split(
        all_point_idx,
        train_size=_LAMBDA_SEARCH_SUBSAMPLE,
        stratify=y_flat,
        random_state=seed,
    )
except ValueError:
    # A cluster has too few points to stratify.
    subsample_idx, _ = train_test_split(
        all_point_idx, train_size=_LAMBDA_SEARCH_SUBSAMPLE, random_state=seed,
    )
search_cache = ids_cache.point_subset(subsample_idx)
data_search = data[subsample_idx]
labels_search = [kmeans_labels[i] for i in subsample_idx]
print(
    f"Lambda search will use a {_LAMBDA_SEARCH_SUBSAMPLE:.0%} stratified subsample: "
    f"{search_cache.N} of {ids_cache.N} points."
)

####################################################################################################
# PEC scoring configuration: alpha (from select_alphas.py) + lambda* (probed here, exactly the
# way max_rules.py probes it), for IDS_ALT_OBJECTIVE_TYPE.
#
# select_alphas.py only ever writes alpha -- lambda* is never persisted to disk, since max_rules.py
# always recomputes it via PEC.compute_lambda_star() (it's cheap: one compute_lambdas() call, no
# selection pass). This script does the same, rather than inventing a separate lambda* file.
#
# The probe (and lambda*) is computed on the full dataset, not the subsample above -- lambda*
# characterizes the rule pool/objective, not any particular sample of points, and max_rules.py's
# probe (whose lambda* this must match so IDS and PEC are compared under the same lambda) is
# likewise computed on the full dataset.

with open(ALPHAS_DIR + f'selected_alphas{OUTFILE_REF}.json') as f:
    selected_alpha_dict = json.load(f)

module_name = f'dscluster; {IDS_ALT_OBJECTIVE_TYPE}; ensemble'
alpha_val = selected_alpha_dict[module_name]

_pec_kwargs = {'objective_type': IDS_ALT_OBJECTIVE_TYPE}
if IDS_ALT_OBJECTIVE_TYPE == 'coverage-cost':
    _pec_kwargs['cluster_centers'] = kmeans_base.centers
    _pec_kwargs['cluster_cost_method'] = 'kmeans'

probe = PEC(
    rules=ensemble_rules,
    n_select=n_select,
    alpha_val=alpha_val,
    lambda_val=None,
    selection_algorithm='distorted-greedy',
    **_pec_kwargs,
)
lambda_star = probe.compute_lambda_star(data, kmeans_labels)

# Degenerate case: no valid lambda exists, so set_lambda fell back to lambda=0 and switched the
# probe to lazy-greedy -- mirrors max_rules.py's identical handling of this case.
if probe.objective.selection_algorithm != 'distorted-greedy':
    print(f"No valid lambda* for {module_name}; falling back to lambda_val=0.0.")
    lambda_star = 0.0
else:
    lambda_star = float(lambda_star)

print(f"alpha={alpha_val}, lambda*={lambda_star} for {module_name}")

pec_scoring = {
    'alpha_val': alpha_val,
    'lambda_val': lambda_star,
    **_pec_kwargs,
}

####################################################################################################
# Coordinate ascent, scored by the PEC objective (coverage-cost by default).
#
# For a candidate lambda array λ:
#   1. Run the inner optimizer with λ on the subsample -> selected solution S*(λ)
#   2. Score S*(λ) directly by the PEC objective value it achieves under the fixed
#      alpha/lambda* above (see score_decision_set() in
#      intercluster/decision_sets/objectives/scoring.py) -- no train/val split, since
#      the goal is to match PEC's own (training-set) objective, not estimate
#      generalization to held-out points (contrast with ids_lambda_search.py's
#      held-out-AUC scoring).

print("Starting coordinate ascent (PEC objective)...")
print(f"  7 lambdas × up to 5 iterations, precision=0.01, tol=1e-3")
t1 = time.time()

ids_model = IDS(
    rules=ids_rules,
    rule_labels=ids_labels,
    n_select=n_select,
    lambdas=None,
    lambda_search_dict=[(0.01, 1.0)] * 7,
    pec_scoring=pec_scoring,
    ternary_search_precision=0.01,
    max_iterations=5,
    tol=1e-3,
    cache=search_cache,
    optimizer='random_greedy',
    random_state=seed,
)
ids_model.fit(data_search, labels_search)
best_lambdas = ids_model.lambdas

print(f"Coordinate ascent finished in {time.time() - t1:.1f}s")
print(f"Best lambdas: {best_lambdas}")

####################################################################################################
# Save as a JSON list of 7 floats

out_path = RULES_DIR + f'ids_lambdas{OUTFILE_REF_ALT}.json'
with open(out_path, 'w') as f:
    json.dump(best_lambdas, f, indent=4)

print(f"Saved to {out_path}")

####################################################################################################
