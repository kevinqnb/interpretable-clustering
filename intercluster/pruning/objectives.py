from numpy.typing import NDArray
from intercluster.utils import kmeans_cost, num_assigned

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
    def __init__(self, X, centers, average : bool = True, normalize : bool = True, threshold : float = 1):
        super().__init__()
        self.X = X
        self.centers = centers
        self.normalize = normalize
        self.average = average
        
        if threshold < 0 or threshold > 1:
            raise ValueError('Threshold must be between 0 and 1.')
        self.threshold = threshold
        
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
        n_assigned = num_assigned(assignment)
        
        if n_assigned < self.threshold * len(self.X):
            return float('inf')
        else:
            return kmeans_cost(
                self.X,
                assignment,
                self.centers,
                average = self.average,
                normalize = self.normalize
            )