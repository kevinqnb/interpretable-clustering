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
from experiments.yeast.config import SEED, N_CLUSTERS, N_SELECT_DEFAULT, OUTFILE_REF, RULES_DIR

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

os.environ["OMP_NUM_THREADS"] = "1"

# REMINDER: The seed should only be initialized here. IDS is given this seed
# explicitly below (random_state=seed) rather than relying on this global
# np.random.seed call, so this lambda search is reproducible independent of
# global NumPy state.
seed = SEED

####################################################################################################
# Data + clustering

data, data_labels, feature_labels, scaler = load_preprocessed_yeast()
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
# Pre-compute IDSCoverageCache (done once, reused across all lambda evaluations)

print(f"Building coverage cache for {len(ids_rules)} rules...")
t0 = time.time()

ids_cache = IDSCoverageCache.from_rules(ids_rules, ids_labels, data, kmeans_labels)

print(f"Cache ready: {len(ids_cache.decisions)} valid decisions in {time.time() - t0:.1f}s")

cache_path = RULES_DIR + f'ids_coverage_cache_ensemble{OUTFILE_REF}.pkl'
with open(cache_path, 'wb') as f:
    pickle.dump(ids_cache, f)
print(f"Cache saved to {cache_path}")

####################################################################################################
# Coordinate ascent, scored by held-out AUC.
#
# For a candidate lambda array λ:
#   1. Run the inner optimizer with λ on a train split → selected solution S*(λ)
#   2. Score S*(λ) by ROC-AUC of "was the top-confidence firing rule's prediction
#      correct" on a held-out val split (see IDS.select() in
#      intercluster/decision_sets/ids.py for the full algorithm).
#
# Unlike scoring S*(λ) with the same training objective used to select it, this
# does not reward λ purely for pushing every weight toward its search-space
# maximum -- see point_subset()/_held_out_auc() in ids.py.

print("Starting coordinate ascent (held-out AUC)...")
print(f"  7 lambdas × up to 5 iterations, precision=0.01, tol=1e-3")
t1 = time.time()

ids_model = IDS(
    rules=ids_rules,
    rule_labels=ids_labels,
    n_select=n_select,
    lambdas=None,
    lambda_search_dict=[(0.01, 1.0)] * 7,
    ternary_search_precision=0.01,
    max_iterations=5,
    tol=1e-3,
    cache=ids_cache,
    optimizer='random_greedy',
    random_state=seed,
)
ids_model.fit(data, kmeans_labels)
best_lambdas = ids_model.lambdas

print(f"Coordinate ascent finished in {time.time() - t1:.1f}s")
print(f"Best lambdas: {best_lambdas}")

####################################################################################################
# Save as a JSON list of 7 floats

out_path = RULES_DIR + f'ids_lambdas{OUTFILE_REF}.json'
with open(out_path, 'w') as f:
    json.dump(best_lambdas, f, indent=4)

print(f"Saved to {out_path}")

####################################################################################################
