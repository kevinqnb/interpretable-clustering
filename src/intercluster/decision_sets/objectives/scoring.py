from typing import List, Set
from numpy.typing import NDArray
from intercluster import Decision
from .coverage_mistake import CoverageMistakeObjective
from .coverage_cost import CoverageCostObjective
from .coverage_pairwise_distance import CoveragePairwiseDistanceObjective

####################################################################################################


def _make_objective(
    objective_type: str,
    n_select: int,
    alpha_val: float,
    lambda_val: float = None,
    cluster_centers: NDArray = None,
    cluster_cost_method: str = 'kmeans',
    weights: NDArray = None,
    data_to_center_distances: NDArray = None,
):
    """Instantiate the right Objective subclass without precomputed data."""
    common = dict(n_select=n_select, alpha_val=alpha_val, lambda_val=lambda_val, weights=weights)
    if objective_type == 'coverage-mistake':
        return CoverageMistakeObjective(**common)
    elif objective_type == 'coverage-cost':
        return CoverageCostObjective(
            cluster_centers=cluster_centers,
            cluster_cost_method=cluster_cost_method,
            data_to_center_distances=data_to_center_distances,
            **common,
        )
    elif objective_type == 'coverage-pairwise-distance':
        return CoveragePairwiseDistanceObjective(**common)
    else:
        raise ValueError(f'Unknown objective type: {objective_type}')


def score_decision_set(
    decisions: List[Decision],
    X: NDArray,
    y: List[Set[int]],
    n_select: int,
    objective_type: str,
    alpha_val: float,
    lambda_val: float,
    cluster_centers: NDArray = None,
    cluster_cost_method: str = 'kmeans',
    weights: NDArray = None,
    data_to_center_distances: NDArray = None,
) -> float:
    """
    Evaluate the PEC objective on a fixed decision set using a forced lambda.

    n_select must match the budget used when lambda was computed (for cost normalization).
    The full decision set is scored as-is (no greedy selection step).
    """
    if not decisions:
        return 0.0
    dset = set(decisions)
    obj = _make_objective(
        objective_type,
        n_select=n_select,
        alpha_val=alpha_val,
        lambda_val=None,  # set explicitly below after initialization
        cluster_centers=cluster_centers,
        cluster_cost_method=cluster_cost_method,
        weights=weights,
        data_to_center_distances=data_to_center_distances,
    )
    obj.initialize_data(X, y)
    obj.initialize_decision_set(dset)
    obj.set_lambda(lambda_val)
    return float(obj.compute_objective(obj.decision_info_dict))


####################################################################################################
