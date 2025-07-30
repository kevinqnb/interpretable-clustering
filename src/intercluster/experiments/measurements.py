import numpy as np
from numpy.typing import NDArray
from intercluster.measures import (
    kmeans_cost,
    overlap,
    coverage,
    center_dists,
    silhouette_score
)
from intercluster.utils import divide_with_zeros

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
        #name = 'point-average-clustering-cost' if average else 'clustering-cost'
        #name = 'normalized-clustering-cost' if normalize else name
        name = 'normalized-clustering-cost' if normalize else 'clustering-cost'
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
            centers,
            assignment,
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
    For every point which is assigned to exactly one cluster, computes the ratio between
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
        single_cover_mask = np.sum(assignment, axis = 1) == 1
        overlap_uncover_mask = ~single_cover_mask
        if np.sum(overlap_uncover_mask) == 0:
            return np.nan

        center_dist_matrix = center_dists(X, centers, norm = 2, square = False)
        sorted_dist_matrix = np.argsort(center_dist_matrix, axis = 1)
        closest_dists = np.array(
            [center_dist_matrix[i, sorted_dist_matrix[i, 0]] for i in range(n)]
        )
        second_closest_dists = np.array(
            [center_dist_matrix[i, sorted_dist_matrix[i, 1]] for i in range(n)]
        )

        # Calculate mean of distribution for all points:
        out1 = divide_with_zeros(second_closest_dists, closest_dists)
        all_points_mean = np.mean(out1)

        # Calculate mean of distribution for ONLY overlapped and uncovered points:
        overlap_uncover_closest_dists = closest_dists[overlap_uncover_mask]
        overlap_uncover_second_closest_dists = second_closest_dists[overlap_uncover_mask]
        out2 = divide_with_zeros(
            overlap_uncover_second_closest_dists,
            overlap_uncover_closest_dists
        )
        overlap_uncover_points_mean = np.mean(out2)

        return all_points_mean / overlap_uncover_points_mean
    

class Silhouette(MeasurementFunction):
    """
    Computes the silhouette score of a clustering.
    """
    def __init__(self):
        super().__init__('silhouette')
        
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
        
        return silhouette_score(X, assignment)
        