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


####################################################################################################


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


####################################################################################################


class TotalCoverage(MeasurementFunction):
    """
    Computes the total coverage of all selected rules. 
    """
    def __init__(self, weights = None, name : str = 'total-coverage'):
        super().__init__(name)
        self.weights = weights
        
    def __call__(
        self,
        data_to_rule_assignment : NDArray = None,
        rule_to_cluster_assignment : NDArray = None,
        data_to_cluster_assignment : NDArray = None,
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

            weights (np.ndarray): Size (n,) array of weights for each data point.

        Returns:
            float : Computed coverage.
        """
        if data_to_cluster_assignment is None:
            return np.nan
        
        return coverage(
            assignment = data_to_cluster_assignment,
            weights = self.weights,
            percentage = False
        )


####################################################################################################


class ClusterCoverage(MeasurementFunction):
    """
    Computes the coverage of rules within their assigned clusters. 

    Args:
        baseline_assignment (np.ndarray: bool): n x k boolean (or binary) matrix with entry (i,j) 
            being True (1) if point i belongs to cluster j and False (0) otherwise. This should correspond 
            to the clustering being approximated by the decision set.
    """
    def __init__(self, baseline_assignment : NDArray, weights = None, name : str = 'cluster-coverage'):
        super().__init__(name)    
        self.baseline_assignment = baseline_assignment
        self.weights = weights

    def __call__(
        self,
        data_to_rule_assignment : NDArray = None,
        rule_to_cluster_assignment : NDArray = None,
        data_to_cluster_assignment : NDArray = None,
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

            weights (np.ndarray): Size (n,) array of weights for each data point.

        Returns:
            float : Computed coverage.
        """
        if data_to_cluster_assignment is None:
            return np.nan
        
        within_cluster_assignment = data_to_cluster_assignment & self.baseline_assignment
        
        return coverage(
            assignment = within_cluster_assignment,
            weights = self.weights,
            percentage = False
        )


####################################################################################################


class Overlap(MeasurementFunction):
    """
    Computes the average overlap between clusters. 
    """
    def __init__(self, name = 'overlap'):
        super().__init__(name)

    def __call__(
        self,
        data_to_rule_assignment : NDArray = None,
        rule_to_cluster_assignment : NDArray = None,
        data_to_cluster_assignment : NDArray = None,
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

            weights (np.ndarray): Size (n,) array of weights for each data point.

        Returns:
            float : Computed overlap.
        """
        if data_to_cluster_assignment is None:
            return np.nan
        
        return overlap(data_to_cluster_assignment)


####################################################################################################


class Mistakes(MeasurementFunction):
    """
    Computes the sum of mistakes made by each rule. 

    Args:
        baseline_assignment (np.ndarray: bool): n x k boolean (or binary) matrix with entry (i,j) 
            being True (1) if point i belongs to cluster j and False (0) otherwise. This should correspond 
            to the clustering being approximated by the decision set.
    """
    def __init__(self, baseline_assignment : NDArray, name : str = 'mistakes'):
        super().__init__(name)
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
            float : Computed coverage.
        """
        if data_to_rule_assignment is None or rule_to_cluster_assignment is None:
            return np.nan
        
        n,r = data_to_rule_assignment.shape
        r2,k = rule_to_cluster_assignment.shape
        assert r == r2, "Number of rules in data_to_rule_assignment and rule_to_cluster_assignment must match."
        assert np.all(np.sum(rule_to_cluster_assignment, axis = 1) == 1), ("Each rule must be "
                                                                "assigned to exactly one cluster.")
        
        total_mistakes = 0.0
        for i in range(r):
            assigned_cluster = np.where(rule_to_cluster_assignment[i,:])[0][0]
            covered = np.sum(data_to_rule_assignment[:,i])
            if covered == 0:
                continue

            correctly_covered = np.sum(
                data_to_rule_assignment[:,i] & self.baseline_assignment[:,assigned_cluster]
            )
            mistakes = covered - correctly_covered
            total_mistakes += mistakes
        
        return total_mistakes


####################################################################################################

    
class ClusteringCost(MeasurementFunction):
    """
    Measures the cost of the clustering as the sum of distances between
    each point in a cluster, and its assigned center.
    
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
    

####################################################################################################
    

class RuleClusteringCost(MeasurementFunction):
    """
    Measures the cost of the clustering as the sum over RULES of distances between
    its covered points and a set of given cluster centers.
    
    Args:
        data (np.ndarray): (n x d) Dataset.
        method (str): 'kmeans' or 'kmedians' distance.
    """
    def __init__(
        self,
        data : NDArray,
        cluster_centers : NDArray = None,
        method : str = 'kmeans',
        name : str = 'rule-clustering-cost'
    ):
        super().__init__(name)
        self.data = data
        self.cluster_centers = cluster_centers
        if method not in ['kmeans', 'kmedians']:
            raise ValueError("Method must be one of 'kmeans' or 'kmedians'.")
        self.method = method
        
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
        if data_to_rule_assignment is None or rule_to_cluster_assignment is None:
            return np.nan
        n,r = data_to_rule_assignment.shape
        r2,k = rule_to_cluster_assignment.shape
        assert r == r2, "Number of rules in data_to_rule_assignment and rule_to_cluster_assignment must match."
        assert np.all(np.sum(rule_to_cluster_assignment, axis = 1) == 1), ("Each rule must be "
                                                                "assigned to exactly one cluster.")
        
        
        if self.cluster_centers is None:
            cluster_centers = np.zeros((k, self.data.shape[1]))
            for j in range(k):
                cluster_points_idx = np.where(data_to_cluster_assignment[:,j])[0]
                if len(cluster_points_idx) == 0:
                    continue
                cluster_points = self.data[cluster_points_idx, :]
                center = np.mean(cluster_points, axis = 0)
                cluster_centers[j,:] = center
        else:
            cluster_centers = self.cluster_centers
        
        
        cost = 0.0
        for i in range(r):
            rule_points_idx = np.where(data_to_rule_assignment[:,i])[0]
            if len(rule_points_idx) == 0:
                continue
            rule_points = self.data[rule_points_idx, :]
            assigned_cluster = np.where(rule_to_cluster_assignment[i,:])[0][0]
            center = cluster_centers[assigned_cluster, :]

            if self.method == 'kmeans':
                dists = np.linalg.norm(rule_points - center, ord = 2, axis = 1)
            else:  # kmedians
                dists = np.linalg.norm(rule_points - center, ord = 1, axis = 1)

            cost += np.sum(dists)

        return cost
        

####################################################################################################


class PairwiseDistance(MeasurementFunction):
    """
    Computes the clustering
    distance between a reference clustering and a new, interpretable clustering.
    """
    def __init__(self, baseline_assignment : NDArray, name : str = 'pairwise-distance'):
        """
        Args:
            ground_truth_assignment (np.ndarray: bool): n x k boolean (or binary) matrix 
                with entry (i,j) being True (1) if point i belongs to cluster j and False (0) 
                otherwise. This should correspond to a ground truth labeling of the data. 
        """
        super().__init__(name = name)
        self.baseline_assignment = baseline_assignment
        self.baseline_labels = assignment_to_labels(baseline_assignment)
        
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
            self.baseline_labels,
            new_labels,
            percentage = True,
            ignore = {-1}
        )
    

####################################################################################################


class RulePairwiseDistance(MeasurementFunction):
    """
    Computes the clustering
    distance between a reference clustering and a new, interpretable clustering.
    """
    def __init__(self, baseline_assignment : NDArray, name : str = 'rule-pairwise-distance'):
        """
        Args:
            ground_truth_assignment (np.ndarray: bool): n x k boolean (or binary) matrix 
                with entry (i,j) being True (1) if point i belongs to cluster j and False (0) 
                otherwise. This should correspond to a ground truth labeling of the data. 
        """
        super().__init__(name = name)
        self.baseline_assignment = baseline_assignment
        self.baseline_labels = assignment_to_labels(baseline_assignment)
        
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
        if data_to_rule_assignment is None:
            return np.nan
        
        n,r = data_to_rule_assignment.shape
        
        total_pairwise_distance = 0.0
        for i in range(r):
            rule_points_idx = np.where(data_to_rule_assignment[:,i])[0]
            if len(rule_points_idx) <= 1:
                continue
            rule_labels = [{0} for _ in range(len(rule_points_idx))]
            baseline_labels = [self.baseline_labels[idx] for idx in rule_points_idx]

            total_pairwise_distance += clustering_distance(
                baseline_labels,
                rule_labels,
                percentage = False,
                ignore = {-1}
            )
    

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





    
        