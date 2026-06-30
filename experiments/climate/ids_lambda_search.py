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
# Load CAR rules and bin_df

from pyids.data_structures.ids_rule import IDSRule
from pyids.data_structures.ids_ruleset import IDSRuleSet
from pyids.data_structures.ids_cacher import IDSCacher
from pyids.algorithms.ids import IDS as IDS_pyids
from pyids.algorithms.ids_objective_function import IDSObjectiveFunction, ObjectiveFunctionParameters
from pyids.model_selection.coordinate_ascent import CoordinateAscent
from pyarc.qcba.data_structures import QuantitativeDataFrame

class_association_rules = load_rules('data/experiments/climate/rules/class_association_rules.pkl')
bin_df_raw = pd.read_csv('data/experiments/climate/rules/bin_df.csv')

####################################################################################################
# Build pyids data structures (mirrors IDS.select() in src/intercluster/decision_sets/ids.py)

cars = decision_set_to_cars(data, kmeans_labels, class_association_rules)
cars = [car for car in cars if car.confidence > 0 and car.support > 0]
cars = [car for car in cars if int(car.consequent[1]) != -1]

if not cars:
    raise RuntimeError("No valid class association rules found after filtering.")

ids_rules = list(map(IDSRule, cars))
ids_ruleset = IDSRuleSet(ids_rules)

bin_df = bin_df_raw.assign(**{'class': y_flat})
bin_df['class'] = bin_df['class'].astype(str)
quant_df = QuantitativeDataFrame(bin_df)

####################################################################################################
# Compute IDS cacher (expensive O(N²) step, done once)

print(f"Computing IDS cacher for {len(ids_ruleset.ruleset)} rules...")
t0 = time.time()
ids_cacher = IDSCacher()
ids_cacher.calculate_overlap(ids_ruleset, quant_df)
print(f"Cacher ready in {time.time() - t0:.1f}s")

####################################################################################################
# Coordinate ascent using the IDS internal objective as the scoring function.
#
# For a candidate lambda array λ:
#   1. Fit IDS using λ → select solution set S*(λ)
#   2. Evaluate S*(λ) with the IDS objective function under λ
#
# This finds lambdas that are self-consistent: the selected rule set maximally
# satisfies the IDS objective criteria (compactness, coverage, low overlap) under
# the same weights used to select it. The ternary search is well-defined because
# the objective is not monotone in any single lambda — increasing one lambda shifts
# emphasis toward that criterion at the expense of others, creating a peak.

def fmax(lambda_dict):
    lambdas = list(lambda_dict.values())

    ids = IDS_pyids(n_select=n_select, algorithm="DLS")
    ids.ids_ruleset = ids_ruleset
    ids.cacher = ids_cacher
    ids.fit(quant_df, lambda_array=lambdas)

    solution = IDSRuleSet(ids.clf.rules)

    params = ObjectiveFunctionParameters()
    params.params["all_rules"] = ids_ruleset
    params.params["len_all_rules"] = len(ids_ruleset.ruleset)
    params.params["quant_dataframe"] = quant_df
    params.params["lambda_array"] = lambdas
    obj_fn = IDSObjectiveFunction(params, cacher=ids_cacher)

    return obj_fn.evaluate(solution)


# Search range: (0, 1) covers the heuristic lambda magnitudes. The extension
# mechanism in CoordinateAscent will expand the upper bound automatically if the
# optimum is near the boundary.
search_space = {f'l{i}': (0.0, 1.0) for i in range(7)}

print("Starting coordinate ascent...")
print(f"  7 lambdas × {10} iterations, ternary search precision=0.001")
t1 = time.time()

coord_asc = CoordinateAscent(
    func=fmax,
    func_args_ranges=search_space,
    ternary_search_precision=0.001,
    max_iterations=10,
)
best_lambdas = coord_asc.fit()

print(f"Coordinate ascent finished in {time.time() - t1:.1f}s")
print(f"Best lambdas: {best_lambdas}")

####################################################################################################
# Save

out_path = 'data/experiments/climate/rules/ids_lambdas.json'
with open(out_path, 'w') as f:
    json.dump(best_lambdas, f, indent=4)

print(f"Saved to {out_path}")

####################################################################################################
