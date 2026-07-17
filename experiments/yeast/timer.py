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
from experiments.modules import *
from experiments.timing import run_timing_sweep
from experiments.profiling import stamp, stamp_reset
from experiments.yeast.config import SEED, N_CLUSTERS, N_SELECT_DEFAULT, OUTFILE_REF, RULES_DIR, ALPHAS_DIR
stamp_reset()

####################################################################################################

import os
import json
import numpy as np
from intercluster.decision_sets import PEC
from intercluster.rules import load_rules

# Prevents memory leakage for KMeans:
os.environ["OMP_NUM_THREADS"] = "1"

# REMINDER: The seed should only be initialized here (see experiments/README.md,
# "Reproducibility"). PEC/distorted-greedy/heap-distorted-greedy have no internal
# randomness of their own -- the only randomness in this script is which rules land
# in each step's subset, which is controlled by `timing_seed` below via
# `run_timing_sweep`'s per-trial `np.random.default_rng(timing_seed + trial)`.
seed = SEED

# Number of rules added to the pool at each step, and number of independent random
# orderings of the rule ensemble to bootstrap timing statistics over. If the mined ensemble
# has fewer than n_step rules, this collapses to a single "full pool" step -- left as the
# requested starting point; tune n_step once the ensemble size here is known.
n_step = 200
n_trials = 10
timing_seed = seed

fixed_parameters = {
    'n_clusters': N_CLUSTERS,
    'n_select': N_SELECT_DEFAULT,
    'seed': seed,
    'n_step': n_step,
    'n_trials': n_trials,
}

####################################################################################################
# Read and process data, and rebuild the same baseline clustering used to mine/select rules:

data, data_labels, feature_labels, scaler = load_preprocessed_yeast()
n, d = data.shape

np.random.seed(seed)
kmeans_base = KMeansBase(n_clusters=fixed_parameters['n_clusters'], random_seed=seed)
kmeans_assignment = kmeans_base.assign(data)
kmeans_labels = kmeans_base.labels
stamp("data loaded + baseline kmeans")

rules_directory = RULES_DIR
alphas_path = ALPHAS_DIR + f'selected_alphas{OUTFILE_REF}.json'
outfile = f'data/experiments/yeast/timing/exp_timing{OUTFILE_REF}.json'

####################################################################################################
# Load pre-mined rules + selected alpha (both produced by mine_rules.py / alphas.py):

ensemble_rules = load_rules(os.path.join(rules_directory, f'ensemble_rules{OUTFILE_REF}.pkl'))
stamp(f"loaded {len(ensemble_rules)} mined rules")

with open(alphas_path) as f:
    selected_alpha_dict = json.load(f)
alpha_val = selected_alpha_dict['dscluster; coverage-cost; ensemble']

####################################################################################################
# Probe lambda* ONCE from the full ensemble (mirrors max_rules.py) and hold it fixed across every
# step below -- this experiment is about comparing selection-algorithm wall time, not re-deriving
# a per-subset lambda*, and a shared lambda_val keeps that recompute cost out of both algorithms'
# timings equally rather than adding asymmetric noise.

probe = PEC(
    rules=ensemble_rules,
    objective_type='coverage-cost',
    cluster_centers=kmeans_base.centers,
    n_select=fixed_parameters['n_select'],
    alpha_val=alpha_val,
    lambda_val=None,
    selection_algorithm='distorted-greedy',
)
lambda_star = probe.compute_lambda_star(data, kmeans_labels)
if probe.objective.selection_algorithm != 'distorted-greedy':
    raise ValueError(
        'No valid lambda* found for this rule ensemble/alpha; cannot fix a lambda_val for timing.'
    )
fixed_parameters['alpha_val'] = alpha_val
fixed_parameters['lambda_star'] = float(lambda_star)
stamp("lambda* probe (once, full ensemble)")

####################################################################################################
# Time distorted-greedy vs. heap-distorted-greedy across a growing random rule subset:

pec_kwargs = dict(
    objective_type='coverage-cost',
    cluster_centers=kmeans_base.centers,
    n_select=fixed_parameters['n_select'],
    alpha_val=alpha_val,
    lambda_val=float(lambda_star),
)

results = run_timing_sweep(
    rules=ensemble_rules,
    X=data,
    y=kmeans_labels,
    pec_kwargs=pec_kwargs,
    n_step=n_step,
    n_trials=n_trials,
    seed=timing_seed,
)
stamp("timing sweep complete")

results['fixed_parameters'] = fixed_parameters

os.makedirs(os.path.dirname(outfile), exist_ok=True)
with open(outfile, 'w') as f:
    json.dump(results, f, indent=2)
stamp("results saved")

print(f"Saved timing results to {outfile}")

####################################################################################################
