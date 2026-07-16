####################################################################################################
# Shared lambda-grid construction for the confidence-threshold sweep.
#
# WHY THIS EXISTS
#
# PEC's lambda* -- the smallest lambda for which the distorted-greedy approximation guarantee
# holds -- is (essentially) a max over coverage/cost ratios across the mined rule pool, so it is
# monotone non-increasing in the filter confidence: raising the threshold removes rules, which can
# only remove candidates for that max. Concretely, on aniso's coverage-cost objective (at its
# sweep's common alpha) lambda* is 0.329 at confidence 0.25 and 0.50 but only 0.051 at 0.75.
#
# lambda* depends on alpha as well as on the pool -- alpha enters the cost term -- so these values
# move if the alpha a run fits against changes. That is why the --emit-grid probe and the sweep it
# feeds must source alpha identically (see experiments/aniso/lambda.py): the grid's lambda* anchor
# is only a valid distorted-greedy cutoff for a PEC fit with the same alpha it was probed under.
#
# When each confidence's lambda sweep builds its own grid over [0, 2 * lambda*], those grids do not
# line up: the 0.75 sweep stops at 0.231 while the 0.25 sweep runs to 0.659, so the high-confidence
# curve simply ends partway across a cross-confidence plot -- not a bug, but it makes the runs
# incomparable. `build_anchored_grid` instead builds ONE grid per objective, shared across every
# confidence, spanning [0, 2 * max_c lambda*_c] and containing every lambda*_c as an exact grid
# point (each run still fits distorted-greedy only for lambda >= its own lambda*, which stays a
# clean cut precisely because that anchor is on the grid).
#
# Grids are per-OBJECTIVE, not global: lambda* spans orders of magnitude across objectives
# (0.689 for coverage-mistake vs 0.0023 for coverage-pairwise-distance), so one global grid would
# leave the pairwise-distance objective with a single usable point.
#
# The grid is computed once, by a barrier stage between the per-confidence alpha stage and the
# per-confidence lambda stage (see any dataset's run_confidence_sweep.sh), and persisted. Every
# run then READS the same floats. That matters beyond saving probes: results are keyed by
# str(lambda), so cross-confidence plots only line up if all runs carry bit-identical grid values.

import json
from typing import Dict, Iterable, List

import numpy as np


def build_anchored_grid(anchors: Iterable[float], n_points: int) -> List[float]:
    """
    Builds an approximately evenly spaced grid that contains every anchor exactly.

    Points are allocated to each consecutive anchor gap in proportion to that gap's width, so
    spacing stays near-uniform across the whole range while every anchor lands exactly on the
    grid. `np.linspace` reproduces its endpoints bit-exactly, so an anchor fed in here comes back
    out unchanged -- which is what lets a downstream `lambda >= lambda_star` filter cut cleanly at
    the boundary point instead of dropping it to a last-ULP mismatch.

    Args:
        anchors (Iterable[float]): Values the grid must contain exactly (duplicates are collapsed).
            Typically {0.0} | {lambda*_c for each confidence c} | {2 * max_c lambda*_c}.
        n_points (int): Target total number of grid points. The result may differ by a point or
            two, since each gap is rounded to a whole number of subintervals (and every gap gets
            at least one, so near-equal anchors are both preserved).

    Returns:
        grid (List[float]): Sorted grid values, from min(anchors) to max(anchors).
    """
    anchors = sorted({float(a) for a in anchors})
    if len(anchors) == 1:
        return [anchors[0]]

    span = anchors[-1] - anchors[0]
    if span <= 0:
        return [anchors[0]]

    step = span / max(1, n_points - 1)

    grid = [anchors[0]]
    for a, b in zip(anchors[:-1], anchors[1:]):
        n_sub = max(1, int(round((b - a) / step)))
        segment = np.linspace(a, b, n_sub + 1)
        # Drop the leading endpoint: it is already on the grid as the previous gap's endpoint
        # (or as anchors[0]). Keeps each anchor exactly once.
        grid.extend(float(v) for v in segment[1:])
    return grid


def build_shared_grids(
    lambda_star_by_module: Dict[str, Dict[str, float]],
    n_points: int,
    upper_factor: float = 2.0,
) -> Dict[str, dict]:
    """
    Builds one shared grid per module from that module's per-confidence lambda* values.

    Args:
        lambda_star_by_module (dict): {module_name: {conf_tag: lambda_star}}.
        n_points (int): Target number of grid points per module.
        upper_factor (float): The grid runs up to upper_factor * max(lambda*) over confidences.

    Returns:
        dict: {module_name: {'lambda_star_by_conf': {...}, 'grid': [...]}}
    """
    out = {}
    for module_name, by_conf in lambda_star_by_module.items():
        stars = [float(v) for v in by_conf.values()]
        max_star = max(stars)
        anchors = [0.0] + stars + [upper_factor * max_star]
        out[module_name] = {
            'lambda_star_by_conf': {t: float(v) for t, v in by_conf.items()},
            'grid': build_anchored_grid(anchors, n_points),
        }
    return out


def save_lambda_grids(path: str, grids: Dict[str, dict], n_points: int, thresholds) -> None:
    payload = {
        'n_lambda_points': n_points,
        'confidence_thresholds': list(thresholds),
        'modules': grids,
    }
    with open(path, 'w') as f:
        json.dump(payload, f, indent=4)


def load_lambda_grids(path: str, tag: str):
    """
    Loads the shared grids and pulls out one confidence's lambda* anchors.

    Returns the anchors as the exact floats the grid was built from -- callers must filter with
    these rather than re-probing lambda*, so the `lambda >= lambda_star` boundary point stays on
    the grid.

    Args:
        path (str): Path to the JSON written by `save_lambda_grids`.
        tag (str): This run's confidence tag (see cli_utils.conf_tag).

    Returns:
        (lambda_grid_dict, lambda_star_dict, lambda_star_by_conf_dict)
    """
    with open(path) as f:
        payload = json.load(f)

    modules = payload['modules']
    lambda_grid_dict = {m: list(v['grid']) for m, v in modules.items()}
    lambda_star_by_conf_dict = {m: v['lambda_star_by_conf'] for m, v in modules.items()}

    missing = [m for m, v in modules.items() if tag not in v['lambda_star_by_conf']]
    if missing:
        raise KeyError(
            f"Shared lambda grid at {path} has no lambda* for confidence tag {tag!r} "
            f"(modules: {missing}). Rebuild it with --emit-grid over every threshold in the "
            f"sweep, including this one."
        )

    lambda_star_dict = {m: float(v['lambda_star_by_conf'][tag]) for m, v in modules.items()}
    return lambda_grid_dict, lambda_star_dict, lambda_star_by_conf_dict


####################################################################################################
