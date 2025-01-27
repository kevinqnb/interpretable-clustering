import numpy as np
from numpy.typing import NDArray
from intercluster.utils import kmeans_cost, overlap, coverage, center_dists

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
        normalize (bool): If True, the cost is normalized to adjust for 
            overlapping clusters and uncovered points. Defaults to False.
    """
    def __init__(self, normalize : bool = False):        
        name = 'normalized-clustering-cost' if normalize else 'clustering-cost'
        super().__init__(name)
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
        return kmeans_cost(X, assignment, centers, normalize = self.normalize)
    
    
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
        return coverage(assignment)
    
    
class OverlapDistance(MeasurementFunction):
    """
    Computes the ratio between
        - The average distance between points which belong to multiple 
          clusters to their closest cluster center.
        - The average distance between all points to their closest cluster center. 
    """
    def __init__(self):
        super().__init__('overlap-distance')
        
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
        center_dist_matrix = center_dists(X, centers)
        closest_dist_arr = np.min(center_dist_matrix, axis = 1)
        overlap_mask = np.sum(assignment, axis = 1) > 1
        single_mask = np.sum(assignment, axis = 1) == 1

        if np.sum(overlap_mask) > 0 and np.sum(single_mask) > 0:
            overlap_avg_dist = np.mean(closest_dist_arr[overlap_mask])
            single_avg_dist = np.mean(closest_dist_arr[single_mask])
            return overlap_avg_dist / single_avg_dist
        else:
            return np.nan