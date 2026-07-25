####################################################################################################
# Shared, unchanging parameters for the mnist experiment pipeline.
#
# Individual stage scripts import these as defaults rather than duplicating the
# same fixed_parameters literals. Some parameters are swept within an individual
# script (e.g. max_rules.py sweeps n_select from N_CLUSTERS up to MAX_RULES, and
# confidence.py sweeps its own confidence range independent of CONFIDENCE_DEFAULT)
# -- the constants below are the defaults those scripts start from or fall back
# to, not hard invariants.

import os

SEED = 342

# Clustering / mining
N_CLUSTERS = 10
N_SELECT_DEFAULT = N_CLUSTERS
MAX_RULES = 16

SHALLOW_TREE_DEPTH_FACTOR = 0.03
N_FOREST = 100
FOREST_MAX_DEPTH = 6

CAR_MIN_SUPPORT = 0.025
CAR_MIN_CONFIDENCE = 0.75
CAR_MAX_RULE_LENGTH = 2 # really means 6 by pyfim convention

# Confidence threshold used once, in mine_rules.py, to filter the pre-filter
# ensemble. This is a default, not an invariant -- confidence.py sweeps its own
# independent range (0.0-0.95) over the pre-filter ensemble and is unaffected by
# this constant. Changing CONFIDENCE_DEFAULT only takes effect by re-running
# mine_rules.py (and everything downstream that reads its output).
CONFIDENCE_DEFAULT = 0.00

# Stochastic-model trial reproducibility (max_rules.py, lambda.py, confidence.py)
N_TRIALS = 3
TRIAL_SEEDS = [SEED + i for i in range(N_TRIALS)]

# Total CPU budget for the pipeline. Defaults to the machine's detected core
# count; override via MNIST_TOTAL_CPU_COUNT (or experiment_runner.py's
# --total-cpu-count) to reserve some cores for other work.
TOTAL_CPU_COUNT = int(os.environ.get('MNIST_TOTAL_CPU_COUNT', os.cpu_count() or 1))

# Per-script joblib worker count for scripts dispatching through Experiment's
# joblib.Parallel (alphas.py, max_rules.py, max_rules_exkmc.py, lambda.py,
# lambda_exkmc.py) or directly (confidence.py). Defaults to the full
# TOTAL_CPU_COUNT budget, i.e. running a stage script standalone uses the whole
# machine. experiment_runner.py overrides this per subprocess via
# MNIST_CPU_COUNT, dividing TOTAL_CPU_COUNT across however many scripts/families
# it launches concurrently at each stage so no stage oversubscribes.
CPU_COUNT = int(os.environ.get('MNIST_CPU_COUNT', TOTAL_CPU_COUNT))

# Suffix threaded through every confidence-dependent artifact filename this
# pipeline saves or reads (mined+filtered rule pools, precomputed cost/mistake/
# pairwise-distance caches, IDS lambdas/caches, selected alphas, and every
# stage's own exp*.json results), not just the top-level experiment JSON files.
# Change this (together with CONFIDENCE_DEFAULT, if that's why you're
# switching) whenever a run's artifacts should land somewhere that won't
# collide with a previous run's. Pre-filter mining artifacts (bin_df.csv, the
# raw per-miner rule files, pre_filter_ensemble_rules/labels.pkl,
# ids_coverage_cache_prefilter.pkl) are the exception: they're produced once
# from the unfiltered pool before any confidence filtering, so they stay
# unsuffixed and don't vary with OUTFILE_REF -- retagging them would force an
# expensive recompute (see mine_rules.py's ~24h bin_df note in
# experiments/README.md) for no benefit.
OUTFILE_REF = '_conf_00'

# Objective type that ids_lambda_search_alt.py's coordinate ascent targets when
# maximizing the PEC objective (rather than held-out AUC) -- see
# experiments/README.md and ids_lambda_search_alt.py for the full rationale.
IDS_ALT_OBJECTIVE_TYPE = 'coverage-cost'

# Directories (relative to repo root; every script inserts repo root onto
# sys.path and is expected to run with repo root as cwd).
RULES_DIR = 'data/experiments/mnist/rules/'
ALPHAS_DIR = 'data/experiments/mnist/alphas/'
MAX_RULES_DIR = 'data/experiments/mnist/max_rules/'
LAMBDA_DIR = 'data/experiments/mnist/lambda/'
CONFIDENCE_DIR = 'data/experiments/mnist/confidence/'
INPUT_SENSITIVITY_DIR = 'data/experiments/mnist/input_sensitivity/'

####################################################################################################
