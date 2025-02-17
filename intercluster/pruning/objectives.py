import numpy as np
from typing import Tuple
from numpy.typing import NDArray
from intercluster.utils import kmeans_cost, update_centers

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
    
    
class KmeansObjective(PruningObjective):
    """
    Measures the cost of the clustering as the sum of distances between
    each point and its assigned center(s). If a threshold fraction of points 
    have not been assigned to a cluster, the objective value is manually set to infinity.
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
            objective_val, tiebreak-val: The main objective value used for comparison along with a 
                tiebreak value (if needed, considered as secondary information).
        """
        updated_centers = update_centers(self.X, assignment)
        objective_cost = kmeans_cost(
            self.X,
            updated_centers,
            assignment,
            average = self.average,
            normalize = self.normalize
        )
        return objective_cost