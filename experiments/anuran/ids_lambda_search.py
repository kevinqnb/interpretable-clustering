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
import time
import pickle
import numpy as np
import pandas as pd
from intercluster import *
from intercluster.decision_sets import *
from intercluster.decision_sets.ids import IDSCoverageCache, IDSObjective, SLSOptimizer, RandomGreedyOptimizer, IDSCoordinateAscent

os.environ["OMP_NUM_THREADS"] = "1"

# REMINDER: The seed should only be initialized here. RandomGreedyOptimizer and
# IDSCoordinateAscent are given this seed explicitly below (random_state=seed)
# rather than relying on this global np.random.seed call, so this lambda search
# is reproducible independent of global NumPy state.
seed = 342

####################################################################################################
# Data + clustering

data, data_labels, feature_labels, scaler = load_preprocessed_anuran('data/anuran')
n, d = data.shape

np.random.seed(seed)

kmeans_base = KMeansBase(n_clusters=5, random_seed=seed)
kmeans_base.assign(data)
kmeans_labels = kmeans_base.labels
y_flat = flatten_labels(kmeans_labels)

n_select = 5

####################################################################################################
# Load pre-mined ensemble rules

ensemble_rules = load_rules('data/experiments/anuran/rules/ensemble_rules.pkl')
print(f"Ensemble rule set size: {len(ensemble_rules)}")

with open('data/experiments/anuran/rules/ensemble_labels.pkl', 'rb') as f:
    ensemble_labels = pickle.load(f)

ids_rules  = ensemble_rules
ids_labels = ensemble_labels

####################################################################################################
# Pre-compute IDSCoverageCache (done once, reused across all lambda evaluations)

print(f"Building coverage cache for {len(ids_rules)} rules...")
t0 = time.time()

ids_cache = IDSCoverageCache.from_rules(ids_rules, ids_labels, data, kmeans_labels)

print(f"Cache ready: {len(ids_cache.decisions)} valid decisions in {time.time() - t0:.1f}s")

cache_path = 'data/experiments/anuran/rules/ids_coverage_cache_ensemble.pkl'
with open(cache_path, 'wb') as f:
    pickle.dump(ids_cache, f)
print(f"Cache saved to {cache_path}")

####################################################################################################
# Coordinate ascent using the IDS objective as the scoring function.
#
# For a candidate lambda array λ:
#   1. Run SLS with λ → selected solution S*(λ)
#   2. Evaluate S*(λ) with the IDS objective under λ
#
# This finds lambdas that are self-consistent: the selected rule set maximally
# satisfies the IDS objective under the same weights used to select it.

D = len(ids_cache.decisions)
N = ids_cache.N

# A single shared Generator (rather than re-seeding with the raw `seed` int on
# every call) so successive fmax() calls during the search draw from a
# continuing, still fully reproducible, random stream instead of repeating the
# exact same draws each time.
_rng = np.random.default_rng(seed)


def fmax(lambdas):
    obj = IDSObjective(lambdas, ids_cache, N=N, M=D)
    optimizer = RandomGreedyOptimizer(obj, list(range(D)), random_state=_rng)
    selected = optimizer.optimize(n_select=n_select)
    return obj.evaluate(set(selected))


search_space = [(0.01, 1.0)] * 7

print("Starting coordinate ascent...")
print(f"  7 lambdas × up to 5 iterations, precision=0.01, tol=1e-3")
t1 = time.time()

coord_asc = IDSCoordinateAscent(
    func=fmax,
    ranges=search_space,
    precision=0.01,
    max_iterations=5,
    tol=1e-3,
    random_state=_rng,
)
best_lambdas = coord_asc.fit()

print(f"Coordinate ascent finished in {time.time() - t1:.1f}s")
print(f"Best lambdas: {best_lambdas}")

####################################################################################################
# Save as a JSON list of 7 floats

out_path = 'data/experiments/anuran/rules/ids_lambdas.json'
with open(out_path, 'w') as f:
    json.dump(best_lambdas, f, indent=4)

print(f"Saved to {out_path}")

####################################################################################################
