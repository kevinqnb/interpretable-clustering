####################################################################################################
# Shared CLI plumbing for the confidence-threshold sweep.
#
# mine_rules.py filters the mined rule ensemble at several confidence
# thresholds and saves one tagged rule pool per threshold (see conf_tag).
# Every downstream script (alphas.py, select_alphas.py, ids_lambda_search.py,
# max_rules.py, lambda.py) accepts --confidence to pick which tagged pool to
# load and tags its own outputs the same way, so a bash runner can drive the
# whole pipeline once per threshold.

import argparse


def conf_tag(confidence: float) -> str:
    """Map a confidence threshold (e.g. 0.75) to a filename tag (e.g. "75")."""
    return str(int(round(confidence * 100)))


def parse_experiment_args(confidence_default: float, cpu_count_default: int = None):
    """Standard CLI for a per-dataset experiment script.

    Every script exposes --confidence. Scripts that dispatch fits through
    Experiment's joblib.Parallel also pass cpu_count_default, which adds
    --cpu-count so a bash runner can reduce it when running two such scripts
    concurrently (see experiments/aniso/run_confidence_sweep.sh).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--confidence', type=float, default=confidence_default)
    if cpu_count_default is not None:
        parser.add_argument('--cpu-count', type=int, default=cpu_count_default)
    return parser.parse_args()
