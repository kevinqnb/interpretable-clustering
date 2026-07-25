####################################################################################################
# Shared, unchanging parameters for the aniso experiment pipeline. Individual stage scripts
# import these as defaults rather than duplicating the same fixed_parameters literals -- they're
# defaults those scripts start from or fall back to, not hard invariants (e.g. max_rules.py
# sweeps n_select from N_CLUSTERS to MAX_RULES, and confidence.py sweeps its own confidence range
# independent of CONFIDENCE_DEFAULT).

import os

SEED = 342

# Clustering / mining
N_CLUSTERS = 5
N_SELECT_DEFAULT = N_CLUSTERS
MAX_RULES = 11

SHALLOW_TREE_DEPTH_FACTOR = 0.03
N_FOREST = 10
FOREST_MAX_DEPTH = 4

CAR_MIN_SUPPORT = 0.025
CAR_MIN_CONFIDENCE = 0.5
CAR_MAX_RULE_LENGTH = 2  # really means 4 by pyfim convention

# Confidence threshold used once, in mine_rules.py, to filter the pre-filter
# ensemble. This is a default, not an invariant -- confidence.py sweeps its own
# independent range (0.0-0.95) over the pre-filter ensemble and is unaffected by
# this constant. Changing CONFIDENCE_DEFAULT only takes effect by re-running
# mine_rules.py (and everything downstream that reads its output).
CONFIDENCE_DEFAULT = 0.0

# Stochastic-model trial reproducibility (max_rules.py, lambda.py, confidence.py)
N_TRIALS = 10
TRIAL_SEEDS = [SEED + i for i in range(N_TRIALS)]

# Total CPU budget for the pipeline. Defaults to the machine's detected core
# count; override via ANISO_TOTAL_CPU_COUNT (or experiment_runner.py's
# --total-cpu-count) to reserve some cores for other work.
TOTAL_CPU_COUNT = int(os.environ.get('ANISO_TOTAL_CPU_COUNT', os.cpu_count() or 1))

# Per-script joblib worker count. Defaults to the full TOTAL_CPU_COUNT (running a stage
# standalone uses the whole machine); experiment_runner.py overrides it per subprocess via
# ANISO_CPU_COUNT, splitting TOTAL_CPU_COUNT across however many scripts run concurrently at
# that stage so none of them oversubscribe the machine.
CPU_COUNT = int(os.environ.get('ANISO_CPU_COUNT', TOTAL_CPU_COUNT))

# Suffix threaded through every confidence-dependent artifact filename this pipeline saves or
# reads (rule pools, precomputed cost/mistake/pairwise-distance caches, IDS lambdas/caches,
# selected alphas, every stage's exp*.json results). Change this (together with
# CONFIDENCE_DEFAULT, if that's why you're switching) whenever a run's artifacts shouldn't
# collide with a previous run's. Pre-filter mining artifacts (bin_df.csv, the raw per-miner rule
# files, pre_filter_ensemble_rules/labels.pkl, ids_coverage_cache_prefilter.pkl) are the one
# exception: they're produced once from the unfiltered pool before any confidence filtering, so
# they stay unsuffixed and aren't rebuilt just because OUTFILE_REF changes -- entropy_bin() in
# particular is expensive to rerun.
OUTFILE_REF = '_conf_00'

# Objective type that ids_lambda_search_alt.py's coordinate ascent maximizes
# (PEC's own objective, rather than held-out AUC) -- see that file for the rationale.
IDS_ALT_OBJECTIVE_TYPE = 'coverage-cost'

# Directories (relative to repo root; every script inserts repo root onto
# sys.path and is expected to run with repo root as cwd).
RULES_DIR = 'data/experiments/aniso/rules/'
ALPHAS_DIR = 'data/experiments/aniso/alphas/'
MAX_RULES_DIR = 'data/experiments/aniso/max_rules/'
LAMBDA_DIR = 'data/experiments/aniso/lambda/'
CONFIDENCE_DIR = 'data/experiments/aniso/confidence/'
INPUT_SENSITIVITY_DIR = 'data/experiments/aniso/input_sensitivity/'

####################################################################################################
