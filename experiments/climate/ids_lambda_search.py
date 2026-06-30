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
import numpy as np
import pandas as pd
from intercluster import *
from intercluster.decision_sets import *
from intercluster.decision_sets.ids import IDSCoverageCache, IDSObjective, SLSOptimizer, IDSCoordinateAscent

os.environ["OMP_NUM_THREADS"] = "1"

seed = 342

####################################################################################################
# Data + clustering

data, data_labels, feature_labels, scaler = load_preprocessed_climate('data/climate')
n, d = data.shape

np.random.seed(seed)

kmeans_base = KMeansBase(n_clusters=6, random_seed=seed)
kmeans_base.assign(data)
kmeans_labels = kmeans_base.labels
y_flat = flatten_labels(kmeans_labels)

n_select = 6

####################################################################################################
# Load pre-mined CAR rules

class_association_rules = load_rules('data/experiments/climate/rules/class_association_rules.pkl')

####################################################################################################
# Pre-compute IDSCoverageCache (the expensive O(N²) step, done once)
#
# We construct the full decision set (one Decision per rule × cluster label),
# then build the cache using Rule.evaluate(data) directly — no bin_df needed.

print(f"Building coverage cache for {len(class_association_rules)} rules...")
t0 = time.time()

# Temporarily construct an IDS with n_select=None to trigger cache building
_pre = IDS(
    rules=class_association_rules,
    n_select=None,
    lambdas=[1.0] * 7,  # placeholder lambdas (cache build doesn't depend on them)
)
_pre.fit(data, kmeans_labels)
ids_cache = _pre.get_cache()

print(f"Cache ready: {len(ids_cache.decisions)} valid decisions in {time.time() - t0:.1f}s")

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


def fmax(lambdas):
    obj = IDSObjective(lambdas, ids_cache, N=N, M=D)
    optimizer = SLSOptimizer(obj, list(range(D)))
    selected = optimizer.optimize(n_select=n_select)
    return obj.evaluate(set(selected))


search_space = [(0.0, 1.0)] * 7

print("Starting coordinate ascent...")
print(f"  7 lambdas × 10 iterations, ternary search precision=0.001")
t1 = time.time()

coord_asc = IDSCoordinateAscent(
    func=fmax,
    ranges=search_space,
    precision=0.001,
    max_iterations=10,
)
best_lambdas = coord_asc.fit()

print(f"Coordinate ascent finished in {time.time() - t1:.1f}s")
print(f"Best lambdas: {best_lambdas}")

####################################################################################################
# Save as a JSON list of 7 floats

out_path = 'data/experiments/climate/rules/ids_lambdas.json'
with open(out_path, 'w') as f:
    json.dump(best_lambdas, f, indent=4)

print(f"Saved to {out_path}")

####################################################################################################
