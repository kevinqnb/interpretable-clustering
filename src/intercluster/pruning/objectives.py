import numpy as np
from typing import Tuple
from numpy.typing import NDArray
from intercluster.measures import (
    coverage,
    kmeans_cost
)
from intercluster.utils import (
    update_centers
)


class PruningObjective:
    """
    Base class for a pruning objective, which is used to evaluate a pruning 
    during the grid search process.
    """
    def __init__(self):
        pass

    def __call__(
        self,
        assignment : NDArray
    ) -> float:
        """
        Computes the objective value associated with a given clustering assignment.
        
        Args:
            assignment (np.ndarray: bool): n x k boolean (or binary) matrix with entry (i,j) 
                being True (1) if point i belongs to cluster j and False (0) otherwise. 
        Returns:
            float: The objective value.
        """
        pass


class CoverageObjective(PruningObjective):
    """
    Measures the coverage for a given assignment of data points to rules.
    """
    def __init__(self):
        pass

    def __call__(
        self,
        data_to_rules_assignment : NDArray
    ) -> float:
        """
        Computes the percent coverage associated with a given point to rule assignment.
        
        Args:
            data_to_rules_assignment (np.ndarray: bool): n x m boolean (or binary) matrix 
                with entry (i,j)  being True (1) if point i satisfies rule j and False (0) 
                otherwise.
        Returns:
            float: Percentage of points covered by the assignment.
        """
        cover = coverage(data_to_rules_assignment, percentage = True)
        return cover



class CoverageMistakeObjective(PruningObjective):
    """
    Measures an objective evaluated on both coverage and mistakes made with respect to some 
    ground truth labeling / clustering. 
    """
    def __init__(
        self,
        lambda_val : float = 1.0
    ):
        """
        Args:
            lambd (float): Weighting factor for the mistakes term in the objective function.
        """
        self.lambda_val = lambda_val

    def __call__(
        self,
        data_to_cluster_assignment : NDArray,
        rule_to_cluster_assignment : NDArray,
        data_to_rules_assignment : NDArray
    ) -> float:
        """
        Computes the objective value associated with a given clustering assignment.
        
        Args:
            data_to_cluster_assignment (np.ndarray: bool): n x k boolean (or binary) matrix 
                with entry (i,j) being True (1) if point i belongs to cluster j and False (0) 
                otherwise. This should correspond to a ground truth labeling of the data. 
            rule_to_cluster_assignment (np.ndarray: bool): m x k boolean (or binary) matrix 
                with entry (i,j) being True (1) if rule i belongs to cluster j and False (0) 
                otherwise. NOTE: each rule must belong to exactly one cluster.
            data_to_rules_assignment (np.ndarray: bool): n x m boolean (or binary) matrix 
                with entry (i,j)  being True (1) if point i satisfies rule j and False (0) 
                otherwise.
        Returns:
            float: The objective value.
        """
        n, k = data_to_cluster_assignment.shape
        m = rule_to_cluster_assignment.shape[0]
        assert rule_to_cluster_assignment.shape[1] == k, \
            "The number of clusters in the rule assignment must match the data assignment."
        assert np.all(np.sum(rule_to_cluster_assignment, axis=1) == 1), \
            "Each rule must belong to exactly one cluster."
        assert data_to_rules_assignment.shape == (n, m), \
            ("The data to rules assignment must have shape (n, m) where n is the number of data "
            "points and m is the number of rules.")
        
        cover = coverage(data_to_rules_assignment, percentage = False)
        
        mistakes = 0
        for i, rule_points in enumerate(data_to_rules_assignment.T):
            rule_cluster = np.where(rule_to_cluster_assignment[i])[0][0]
            cluster_points = data_to_cluster_assignment[:, rule_cluster]
            mistakes += np.sum(rule_points & ~cluster_points)

        return cover - self.lambda_val * mistakes
        
    
class KmeansObjective(PruningObjective):
    """
    Measures the cost of the clustering as the sum of distances between
    each point and its assigned center(s).
    """
    def __init__(
        self,
        X,
        centers,
        average : bool = False,
        normalize : bool = False
    ):
        super().__init__()
        self.X = X
        self.centers = centers
        self.normalize = normalize
        self.average = average
        
    def __call__(
        self,
        assignment : NDArray
    ) -> Tuple[float, float]:
        """
        Computes the objective value associated with a given clustering assignment.
        
        Args:
            assignment (np.ndarray: bool): n x k boolean (or binary) matrix with entry (i,j) 
                being True (1) if point i belongs to cluster j and False (0) otherwise. 
        Returns:
            objective_val, tiebreak-val: The computed objective value.
        """
        updated_centers = update_centers(self.X, self.centers, assignment)
        objective_cost = kmeans_cost(
            self.X,
            updated_centers,
            assignment,
            average = self.average,
            normalize = self.normalize
        )
        return objective_cost