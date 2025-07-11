import numpy as np
from typing import Tuple
from numpy.typing import NDArray
from intercluster.utils import kmeans_cost, update_centers, coverage, can_flatten, flatten_labels
from typing import List, Set

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
        cover = coverage(assignment)

'''
class CoverageMistakeObjective(PruningObjective):
    """
    Base class for a pruning objective, which is used to evaluate a pruning 
    during the grid search process.
    """
    def __init__(
        self,
        data_to_cluster_assignment : NDArray,
        rule_to_cluster_assignment : NDArray,
        data_to_rules_assignment : NDArray,
        lambd : float = 1.0
    ):
        """
        Args:
            labels (List[Set[int]]): List of sets of labels, where each set corresponds to a 
                point and cotains all labels that point belongs to. 
                Note, however, that we assume here each point has exactly one label.
            lambd (float): Weighting factor for the mistakes term in the objective function.
        """
        if not can_flatten(labels):
            raise ValueError("Each data point must have exactly one label.")
        self.labels = flatten_labels(labels)
        self.lambd = lambd

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
        cover = coverage(assignment)

        mistakes = 0
        for i, l in enumerate(self.labels):
            assigned_labels = set(np.where(assignment[i])[0])
            mistakes += len(assigned_labels - set(l))
'''    
    
    
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
    

class CoverageObjective(PruningObjective):
    """
    Measures the coverage of the pruned assignment.
    """
    def __init__(self):
        super().__init__()
        
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
            objective_val: The computed objective value.
        """
        return coverage(assignment)