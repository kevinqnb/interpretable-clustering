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


def parse_experiment_args(
    confidence_default: float,
    cpu_count_default: int = None,
    grid_flags: bool = False,
    alpha_flag: bool = False,
):
    """Standard CLI for a per-dataset experiment script.

    Every script exposes --confidence. Scripts that dispatch fits through
    Experiment's joblib.Parallel also pass cpu_count_default, which adds
    --cpu-count so a bash runner can reduce it when running two such scripts
    concurrently (see experiments/aniso/run_confidence_sweep.sh).

    grid_flags adds the two flags lambda.py needs to share one lambda grid across
    the whole sweep (see experiments/lambda_grid.py):
      --emit-grid              probe lambda* for every threshold, write the shared
                               grid, and exit without running the sweep. This is the
                               barrier stage: it needs the alpha-source threshold's
                               selected alphas on disk, since lambda* depends on alpha.
      --confidence-thresholds  the thresholds to probe under --emit-grid (matching
                               mine_rules.py's flag). Ignored by a normal run, which
                               just reads the grid and uses --confidence.

    alpha_flag adds:
      --alpha-confidence       read the selected alphas from THIS threshold's
                               selected_alphas file instead of --confidence's, so every
                               confidence in a sweep fits PEC against one common alpha.
                               See experiments/aniso/run_confidence_sweep.sh for why.
                               Omitted => each run uses its own threshold's alphas (the
                               original per-confidence behaviour).

    Both flag groups are opt-in rather than always-on so that a script which does not
    honour a flag rejects it outright instead of silently ignoring it -- only aniso is
    on the common-alpha path so far.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--confidence', type=float, default=confidence_default)
    if cpu_count_default is not None:
        parser.add_argument('--cpu-count', type=int, default=cpu_count_default)
    if grid_flags:
        parser.add_argument('--emit-grid', action='store_true')
        parser.add_argument(
            '--confidence-thresholds', type=float, nargs='+', default=[0.25, 0.5, 0.75]
        )
    if alpha_flag:
        parser.add_argument('--alpha-confidence', type=float, default=None)
    return parser.parse_args()


def alpha_tag_for(args) -> str:
    """The confidence tag whose selected_alphas file a run should read.

    --alpha-confidence when given, else the run's own --confidence. Compared against
    None rather than truthiness so that --alpha-confidence 0.0 is honoured.
    """
    alpha_confidence = getattr(args, 'alpha_confidence', None)
    if alpha_confidence is None:
        alpha_confidence = args.confidence
    return conf_tag(alpha_confidence)
