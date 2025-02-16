import numpy as np
from numpy.typing import NDArray
from intercluster.utils import kmeans_cost, update_centers, overlap, coverage, center_dists

class MeasurementFunction:
    def __init__(self, name):
        self.name = name

    def __call__(
        self,
        X : NDArray,
        assignment : NDArray,
        centers : NDArray
    ):
        """
        X (np.ndarray): (n x d) Dataset
        
        assignment (np.ndarray: bool): n x k boolean (or binary) matrix with entry (i,j) 
            being True (1) if point i belongs to cluster j and False (0) otherwise. 
            
        centers (np.ndarray): (k x d) Set of representative centers for each of the k clusters.
        """
        pass
    
    
class ClusteringCost(MeasurementFunction):
    """
    Measures the cost of the clustering as the sum of distances between
    each point and its assigned center.
    
    Args:
        average (bool, optional): Whether to average the per-point cost by the number of clusters
                that the point is assigned to. Defaults to False.
        normalize (bool): If True, the cost is normalized to adjust for 
            overlapping clusters and uncovered points. Defaults to False.
    """
    def __init__(self, average : bool = False, normalize : bool = False):
        name = 'point-average-clustering-cost' if average else 'clustering-cost'
        name = 'normalized-clustering-cost' if normalize else name
        super().__init__(name)
        self.average = average
        self.normalize = normalize
        
    def __call__(
        self,
        X : NDArray,
        assignment : NDArray,
        centers : NDArray
    ) -> float:
        """
        X (np.ndarray): (n x d) Dataset
        
        assignment (np.ndarray: bool): n x k boolean (or binary) matrix with entry (i,j) 
            being True (1) if point i belongs to cluster j and False (0) otherwise. 
            
        centers (np.ndarray): (k x d) Set of representative centers for each of the k clusters.
        """
        if (assignment is None) or (centers is None):
            return np.nan
        
        return kmeans_cost(
            X,
            assignment,
            centers,
            average = self.average,
            normalize = self.normalize
        )
    
    
class Overlap(MeasurementFunction):
    """
    Computes the average overlap between clusters. 
    """
    def __init__(self):
        super().__init__('overlap')
        
    def __call__(
        self,
        X : NDArray,
        assignment : NDArray,
        centers : NDArray
    ) -> int:
        """
        X (np.ndarray): (n x d) Dataset
        
        assignment (np.ndarray: bool): n x k boolean (or binary) matrix with entry (i,j) 
            being True (1) if point i belongs to cluster j and False (0) otherwise. 
            
        centers (np.ndarray): (k x d) Set of representative centers for each of the k clusters.
        """
        if (assignment is None) or (centers is None):
            return np.nan
        
        return overlap(assignment)
    
    
class Coverage(MeasurementFunction):
    """
    Computes the average overlap between clusters. 
    """
    def __init__(self):
        super().__init__('coverage')
        
    def __call__(
        self,
        X : NDArray,
        assignment : NDArray,
        centers : NDArray
    ) -> int:
        """
        X (np.ndarray): (n x d) Dataset
        
        assignment (np.ndarray: bool): n x k boolean (or binary) matrix with entry (i,j) 
            being True (1) if point i belongs to cluster j and False (0) otherwise. 
            
        centers (np.ndarray): (k x d) Set of representative centers for each of the k clusters.
        """
        if (assignment is None) or (centers is None):
            return np.nan
        
        return coverage(assignment)
    
    
class DistanceRatio(MeasurementFunction):
    """
    For every point computes the ratio between
        - The distance to its second closest center
        - The distance to its closest cluster center. 
    """
    def __init__(self):
        super().__init__('distance-ratio')
        
    def __call__(
        self,
        X : NDArray,
        assignment : NDArray,
        centers : NDArray
    ) -> int:
        """
        X (np.ndarray): (n x d) Dataset
        
        assignment (np.ndarray: bool): n x k boolean (or binary) matrix with entry (i,j) 
            being True (1) if point i belongs to cluster j and False (0) otherwise. 
            
        centers (np.ndarray): (k x d) Set of representative centers for each of the k clusters.
        """
        if (assignment is None) or (centers is None):
            return np.nan
        
        n,d = X.shape
        center_dist_matrix = center_dists(X, centers, norm = 2, square = False)
        sorted_dist_matrix = np.argsort(center_dist_matrix, axis = 1)
        closest_dists = np.array(
            [center_dist_matrix[i, sorted_dist_matrix[i, 0]] for i in range(n)]
        )
        second_closest_dists = np.array(
            [center_dist_matrix[i, sorted_dist_matrix[i, 1]] for i in range(n)]
        )
        return np.mean(second_closest_dists/closest_dists)
        