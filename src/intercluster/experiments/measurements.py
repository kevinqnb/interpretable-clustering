import numpy as np
from numpy.typing import NDArray
from intercluster.measurements import (
    kmeans_cost,
    overlap,
    coverage,
    center_dists,
    silhouette_score,
    coverage_mistake_score,
    clustering_distance
)
from intercluster.utils import (
    assignment_to_labels,
    assignment_to_dict,
    divide_with_zeros
)
from intercluster.decision_sets.objectives import Objective

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


class Coverage(MeasurementFunction):
    """
    Computes the average overlap between clusters. 
    """
    def __init__(self):
        super().__init__('coverage')
        
    def __call__(
        self,
        data_to_rule_assignment : NDArray = None,
        rule_to_cluster_assignment : NDArray = None,
        data_to_cluster_assignment : NDArray = None
    ) -> int:
        """
        Args:
            data_to_rules_assignment (NDArray): A boolean matrix where entry (i,j) is `True` if 
                    data point i is assigned to rule j and `False` otherwise.

            rule_to_cluster_assignment (np.ndarray): Size (r x k) boolean array where entry (i,j) is 
                `True` if rule i is assigned to cluster j and `False` otherwise. Each rule must 
                be assigned to a single cluster.

            data_to_cluster_assignment (np.ndarray): Size (n x k) boolean array where entry (i,j) is 
                `True` if point i is assigned to cluster j and `False` otherwise. Data points may be 
                assigned to multiple clusters. 

        Returns:
            float : Computed coverage.
        """
        if data_to_cluster_assignment is None:
            return np.nan
        
        return coverage(assignment = data_to_cluster_assignment, percentage = False)


####################################################################################################

    
class ClusteringCost(MeasurementFunction):
    """
    Measures the cost of the clustering as the sum of distances between
    each point and its assigned center.
    
    Args:
        data (np.ndarray): (n x d) Dataset.
        average (bool, optional): Whether to average the per-point cost by the number of clusters
                that the point is assigned to. Defaults to False.
        normalize (bool): If True, the cost is normalized to adjust for 
            overlapping clusters and uncovered points. Defaults to False.
    """
    def __init__(
            self,
            data : NDArray,
            method : str = 'kmeans',
            average : bool = False,
            normalize : bool = False,
            name : str = 'clustering-cost'
        ):
        super().__init__(name)
        self.data = data
        if method not in ['kmeans', 'kmedians']:
            raise ValueError("Method must be one of 'kmeans' or 'kmedians'.")
        self.method = method
        self.average = average
        self.normalize = normalize
        
    def __call__(
        self,
        data_to_rule_assignment : NDArray = None,
        rule_to_cluster_assignment : NDArray = None,
        data_to_cluster_assignment : NDArray = None
    ) -> int:
        """
        Args:
            data_to_rules_assignment (NDArray): A boolean matrix where entry (i,j) is `True` if 
                    data point i is assigned to rule j and `False` otherwise.

            rule_to_cluster_assignment (np.ndarray): Size (r x k) boolean array where entry (i,j) is 
                `True` if rule i is assigned to cluster j and `False` otherwise. Each rule must 
                be assigned to a single cluster.

            data_to_cluster_assignment (np.ndarray): Size (n x k) boolean array where entry (i,j) is 
                `True` if point i is assigned to cluster j and `False` otherwise. Data points may be 
                assigned to multiple clusters. 

        Returns:
            float : Computed coverage.
        """
        if data_to_cluster_assignment is None:
            return np.nan
        n,k = data_to_cluster_assignment.shape
        
        cost = 0.0
        for j in range(k):
            cluster_points_idx = np.where(data_to_cluster_assignment[:,j])[0]
            if len(cluster_points_idx) == 0:
                continue
            cluster_points = self.data[cluster_points_idx, :]
            center = np.mean(cluster_points, axis = 0)
            if self.method == 'kmeans':
                dists = np.linalg.norm(cluster_points - center, ord = 2, axis = 1)
            else:  # kmedians
                dists = np.linalg.norm(cluster_points - center, ord = 1, axis = 1)

            if self.average:
                avg_dists = np.array(
                    [dists[i] / np.sum(data_to_cluster_assignment[c, :]) for i,c in enumerate(cluster_points_idx)]
                )
                cost += np.sum(avg_dists)
            else:
                cost += np.sum(dists)

        if self.normalize:
            covered = coverage(data_to_cluster_assignment, percentage = False)
            cost /= covered if covered > 0 else 1.0

        return cost
        
        
    
    
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


####################################################################################################
    
    
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
    

####################################################################################################
    

class Silhouette(MeasurementFunction):
    """
    Computes the silhouette score of a clustering.
    """
    def __init__(self, distances : NDArray, name : str = 'silhouette'):
        """
        Args:
            distances (np.ndarray): n x n array of pairwise distances between points in the dataset.
        """
        super().__init__(name = name)
        self.distances = distances
        
    def __call__(
        self,
        data_to_rule_assignment : NDArray = None,
        rule_to_cluster_assignment : NDArray = None,
        data_to_cluster_assignment : NDArray = None
    ) -> int:
        """
        Args:
            data_to_rules_assignment (NDArray): A boolean matrix where entry (i,j) is `True` if 
                    data point i is assigned to rule j and `False` otherwise.

            rule_to_cluster_assignment (np.ndarray): Size (r x k) boolean array where entry (i,j) is 
                `True` if rule i is assigned to cluster j and `False` otherwise. Each rule must 
                be assigned to a single cluster.

            data_to_cluster_assignment (np.ndarray): Size (n x k) boolean array where entry (i,j) is 
                `True` if point i is assigned to cluster j and `False` otherwise. Data points may be 
                assigned to multiple clusters. 

        Returns:
            float : Computed silhouette score.
        """
        if data_to_cluster_assignment is None:
            return np.nan
        
        return silhouette_score(self.distances, data_to_cluster_assignment)
    

####################################################################################################


class ClusteringDistance(MeasurementFunction):
    """
    Computes the distance between a reference clustering and a new, interpretable clustering.
    """
    def __init__(self, ground_truth_assignment : NDArray, name : str = 'clustering-distance'):
        """
        Args:
            ground_truth_assignment (np.ndarray: bool): n x k boolean (or binary) matrix 
                with entry (i,j) being True (1) if point i belongs to cluster j and False (0) 
                otherwise. This should correspond to a ground truth labeling of the data. 
        """
        super().__init__(name = name)
        self.ground_truth_assignment = ground_truth_assignment
        self.ground_truth_labels = assignment_to_labels(ground_truth_assignment)
        
    def __call__(
        self,
        data_to_rule_assignment : NDArray = None,
        rule_to_cluster_assignment : NDArray = None,
        data_to_cluster_assignment : NDArray = None
    ) -> int:
        """
        Args:
            data_to_rules_assignment (NDArray): A boolean matrix where entry (i,j) is `True` if 
                    data point i is assigned to rule j and `False` otherwise.

            rule_to_cluster_assignment (np.ndarray): Size (r x k) boolean array where entry (i,j) is 
                `True` if rule i is assigned to cluster j and `False` otherwise. Each rule must 
                be assigned to a single cluster.

            data_to_cluster_assignment (np.ndarray): Size (n x k) boolean array where entry (i,j) is 
                `True` if point i is assigned to cluster j and `False` otherwise. Data points may be 
                assigned to multiple clusters. 

        Returns:
            float : Computed coverage mistake score.
        """
        if data_to_cluster_assignment is None:
            return np.nan
        
        new_labels = assignment_to_labels(data_to_cluster_assignment)
        return clustering_distance(
            self.ground_truth_labels,
            new_labels,
            percentage = True,
            ignore = {-1}
        )
    

####################################################################################################


class ObjectiveValue(MeasurementFunction):
    """
    Records the value of the objective function used to fit the decision set.
    """
    def __init__(
        self,
        objective : Objective,
        baseline_assignment : NDArray,
        name : str = 'objective-value'
    ):
        super().__init__(name = name)
        self.objective = objective
        self.baseline_assignment = baseline_assignment
        
    def __call__(
        self,
        data_to_rule_assignment : NDArray = None,
        rule_to_cluster_assignment : NDArray = None,
        data_to_cluster_assignment : NDArray = None
    ) -> int:
        """
        Args:
            data_to_rules_assignment (NDArray): A boolean matrix where entry (i,j) is `True` if 
                    data point i is assigned to rule j and `False` otherwise.

            rule_to_cluster_assignment (np.ndarray): Size (r x k) boolean array where entry (i,j) is 
                `True` if rule i is assigned to cluster j and `False` otherwise. Each rule must 
                be assigned to a single cluster.

            data_to_cluster_assignment (np.ndarray): Size (n x k) boolean array where entry (i,j) is 
                `True` if point i is assigned to cluster j and `False` otherwise. Data points may be 
                assigned to multiple clusters. 

        Returns:
            float : Computed objective value.
        """
        if data_to_rule_assignment is None or rule_to_cluster_assignment is None:
            return np.nan
        n,d = data_to_rule_assignment.shape
        r,k = rule_to_cluster_assignment.shape
        
        selected_rule_labels = {i: rule_to_cluster_assignment[i,:].nonzero()[0][0] for i in range(r)}
        selected_rule_points = assignment_to_dict(data_to_rule_assignment)
        cluster_points = assignment_to_dict(self.baseline_assignment)
        selected_rule_coverage = {
            i : r_points.intersection(
                cluster_points[selected_rule_labels[i]]
            )
            for i, r_points in selected_rule_points.items()
        }
        return self.objective.compute_objective(
            selected_rule_labels,
            selected_rule_points,
            selected_rule_coverage
        )
    

####################################################################################################


class ObjectiveGain(MeasurementFunction):
    """
    Records the gain portion of the objective function used to fit the decision set.
    """
    def __init__(
        self,
        objective : Objective,
        baseline_assignment : NDArray,
        name : str = 'objective-value'
    ):
        super().__init__(name = name)
        self.objective = objective
        self.baseline_assignment = baseline_assignment
        
    def __call__(
        self,
        data_to_rule_assignment : NDArray = None,
        rule_to_cluster_assignment : NDArray = None,
        data_to_cluster_assignment : NDArray = None
    ) -> int:
        """
        Args:
            data_to_rules_assignment (NDArray): A boolean matrix where entry (i,j) is `True` if 
                    data point i is assigned to rule j and `False` otherwise.

            rule_to_cluster_assignment (np.ndarray): Size (r x k) boolean array where entry (i,j) is 
                `True` if rule i is assigned to cluster j and `False` otherwise. Each rule must 
                be assigned to a single cluster.

            data_to_cluster_assignment (np.ndarray): Size (n x k) boolean array where entry (i,j) is 
                `True` if point i is assigned to cluster j and `False` otherwise. Data points may be 
                assigned to multiple clusters. 

        Returns:
            float : Computed objective value.
        """
        if data_to_rule_assignment is None or rule_to_cluster_assignment is None:
            return np.nan
        n,d = data_to_rule_assignment.shape
        r,k = rule_to_cluster_assignment.shape
        
        selected_rule_labels = {i: rule_to_cluster_assignment[i,:].nonzero()[0][0] for i in range(r)}
        selected_rule_points = assignment_to_dict(data_to_rule_assignment)
        cluster_points = assignment_to_dict(self.baseline_assignment)
        selected_rule_coverage = {
            i : r_points.intersection(
                cluster_points[selected_rule_labels[i]]
            )
            for i, r_points in selected_rule_points.items()
        }
        return self.objective.gain(
            selected_rule_labels,
            selected_rule_points,
            selected_rule_coverage
        )
    

####################################################################################################


class ObjectiveCost(MeasurementFunction):
    """
    Records the value of the objective function used to fit the decision set.
    """
    def __init__(
        self,
        objective : Objective,
        baseline_assignment : NDArray,
        name : str = 'objective-value'
    ):
        super().__init__(name = name)
        self.objective = objective
        self.baseline_assignment = baseline_assignment
        
    def __call__(
        self,
        data_to_rule_assignment : NDArray = None,
        rule_to_cluster_assignment : NDArray = None,
        data_to_cluster_assignment : NDArray = None
    ) -> int:
        """
        Args:
            data_to_rules_assignment (NDArray): A boolean matrix where entry (i,j) is `True` if 
                    data point i is assigned to rule j and `False` otherwise.

            rule_to_cluster_assignment (np.ndarray): Size (r x k) boolean array where entry (i,j) is 
                `True` if rule i is assigned to cluster j and `False` otherwise. Each rule must 
                be assigned to a single cluster.

            data_to_cluster_assignment (np.ndarray): Size (n x k) boolean array where entry (i,j) is 
                `True` if point i is assigned to cluster j and `False` otherwise. Data points may be 
                assigned to multiple clusters. 

        Returns:
            float : Computed objective value.
        """
        if data_to_rule_assignment is None or rule_to_cluster_assignment is None:
            return np.nan
        n,d = data_to_rule_assignment.shape
        r,k = rule_to_cluster_assignment.shape
        
        selected_rule_labels = {i: rule_to_cluster_assignment[i,:].nonzero()[0][0] for i in range(r)}
        selected_rule_points = assignment_to_dict(data_to_rule_assignment)
        cluster_points = assignment_to_dict(self.baseline_assignment)
        selected_rule_coverage = {
            i : r_points.intersection(
                cluster_points[selected_rule_labels[i]]
            )
            for i, r_points in selected_rule_points.items()
        }
        return self.objective.cost(
            selected_rule_labels,
            selected_rule_points,
            selected_rule_coverage
        )
    

####################################################################################################





    
        