import numpy as np
from numpy.typing import NDArray
from intercluster import Decision
from intercluster.utils import (
    assignment_to_dict, labels_to_assignment, unique_labels
)
from .objectives import Objective


####################################################################################################


class CoverageCostObjective(Objective):
    """
    Objective that selects rules based on a coverage and cluster cost objective.

    Args:
        data (NDArray): (n x d) array.
        cluster_centers (NDArray): (k x d) array where each row i is the given 
                representative for cluster i.
        n_select (int): The *maximum* number of rules to select.
        lambda_val (float): A hyperparameter that controls tradeoff between coverage and overlap.
            Defaults to 1.0.
        alpha_val (float): A hyperparameter for tuning the size of the selected rules.
            Larger values penalize longer rules more heavily. Defaults to 1.0.
        cluster_cost_method (str): The cluster_cost_method used to compute cluster costs. 
            Currently only "kmeans" or "kmedians" are supported.
    """
    def __init__(
            self,
            n_select : int,
            alpha_val : float = 0.0,
            lambda_val : float = None,
            cluster_centers : NDArray = None,
            weights : NDArray = None,
            cluster_cost_method : str = "kmeans"
        ):
        super().__init__(
            n_select = n_select,
            alpha_val = alpha_val,
            lambda_val = lambda_val,
            cluster_centers = cluster_centers,
            weights = weights
        )
        if self.cluster_centers is None:
            raise ValueError("Cluster_centers must be provided for this objective.")
        if cluster_cost_method not in ["kmeans", "kmedians"]:
            raise ValueError(f"Method {cluster_cost_method} not supported. Supported cluster_cost_methods are 'kmeans' and 'kmedians'.")
        self.cluster_cost_method = cluster_cost_method


    def initialize_data(
        self, 
        X : NDArray,
        y : list[set[int]],
    ):
        """
        Sets the data for the objective.
        """
        assert isinstance(X, np.ndarray), 'X must be a numpy array.'
        assert len(X.shape) == 2, 'X must be a 2D array.'
        assert len(y) == X.shape[0], 'y must have the same number of elements as X has rows.'
        assert all(isinstance(label_set, set) for label_set in y), \
            'Each element of y must be a set of labels.'
        
        if self.weights is None:
            self.weights = np.ones(X.shape[0], dtype = float)
        else:
            assert len(self.weights) == X.shape[0], \
                'weights must have the same length as the number of samples in X.'

        self.X = X
        self.y = y
            
        self.label_set = unique_labels(y)
        self.n_labels = len(self.label_set)
        data_to_cluster_assignment = labels_to_assignment(
            y, n_labels = self.n_labels, ignore = {-1}
        )
        self.cluster_coverage_dict = assignment_to_dict(data_to_cluster_assignment)

        # Compute n x k distance matrix between data points and cluster centers.
        self.data_to_center_distances = np.zeros((self.X.shape[0], self.cluster_centers.shape[0]))
        for i in range(self.cluster_centers.shape[0]):
            if self.cluster_cost_method == "kmeans":
                self.data_to_center_distances[:, i] = np.sum((self.X - self.cluster_centers[i])**2, axis=1)
            else:  # self.cluster_cost_cluster_cost_method == "kmedians"
                self.data_to_center_distances[:, i] = np.sum(np.abs(self.X - self.cluster_centers[i]), axis=1)

        self.data_initialized = True


    def reward(
        self,
        selected_decisions_info: dict[Decision, dict[str, any]],
    ) -> float:
        """
        Computes the reward from the selected decisions.

        Args:
            selected_decisions_info (dict[Decision, dict[str, any]]): A dictionary mapping each 
                selected decision to its information (points, coverage, length, label).
        Returns:
            reward (float): The reward from the selected decisions.
        """
        total_cluster_coverage = {}
        for decision, info in selected_decisions_info.items():
            label = info['label']
            if label not in total_cluster_coverage:
                total_cluster_coverage[label] = set()
            total_cluster_coverage[label].update(info['cluster_coverage'])
        
        total_weighted_coverage = 0
        for covered in total_cluster_coverage.values():
            if covered:  # Only process non-empty sets
                covered_array = np.fromiter(covered, dtype=np.int64)
                total_weighted_coverage += np.sum(self.weights[covered_array])
        return total_weighted_coverage


    def cost(
        self,
        selected_decisions_info: dict[Decision, dict[str, any]]
    ) -> float:
        """
        Computes the cost of the selected decisions.

        Args:
            selected_decisions_info (dict[Decision, dict[str, any]]): A dictionary mapping each
                selected decision to its information (points, coverage, length, label).
        Returns:
            cost (float): The cost of the selected decisions.
        """
        total_cost = 0.0
        length_penalty = 0.0
        for decision, info in selected_decisions_info.items():
            r_coverage = info['coverage']
            r_center = info['label']
            if r_coverage:  # Only process non-empty sets
                coverage_array = info['coverage_array']
                cluster_cost = np.sum(self.data_to_center_distances[coverage_array, r_center])
                total_cost += cluster_cost
            length_penalty += self.alpha_val * info['length']

        return total_cost + length_penalty


    def marginal_reward(
        self,
        decision_info: dict[str, any],
        total_coverage : set[int],
        total_cluster_coverage : dict[int, set[int]]
    ) -> float:
        """
        Computes the marginal reward as new coverage from selected decision.

        Args:
            decision_info (dict[str, any]): A dictionary containing information about the decision 
                being considered.
            cluster_coverage (dict[int, set[int]]): A dictionary mapping each cluster label to the 
                set of data points already covered by selected decisions.

        Returns:
            coverage (float): The coverage of the selected decisions.
        """
        r_cluster_coverage = decision_info['cluster_coverage']
        s_coverage = total_cluster_coverage[decision_info['label']]
        new_coverage = r_cluster_coverage.difference(s_coverage)
        new_coverage_array = np.fromiter(new_coverage, dtype=np.int64)
        new_coverage_weighted = np.sum(self.weights[new_coverage_array])
        return new_coverage_weighted


####################################################################################################


class TotalCoverageCostObjective(Objective):
    """
    Objective that selects rules based on a coverage and cluster cost objective. The difference
    with CoverageCostObjective is that the coverage is computed across all clusters,
    rather than within each cluster.

    Args:
        data (NDArray): (n x d) data array.
        cluster_centers (NDArray): (k x d) array where each row i is the given 
            representative for cluster i.
        n_select (int): The *maximum* number of rules to select.
        lambda_val (float): A hyperparameter that controls tradeoff between coverage and overlap.
            Defaults to 1.0.
        alpha_val (float): A hyperparameter that controls the length penalty. Defaults to 1.0.
        cluster_cost_method (str): The cluster_cost_method used to compute cluster costs. 
            Currently only "kmeans" or "kmedians" are supported.
    """
    def __init__(
            self,
            n_select : int,
            alpha_val : float = 0.0,
            lambda_val : float = None,
            cluster_centers : NDArray = None,
            weights : NDArray = None,
            cluster_cost_method : str = "kmeans"
        ):
        super().__init__(
            n_select = n_select,
            alpha_val = alpha_val,
            lambda_val = lambda_val,
            cluster_centers = cluster_centers,
            weights = weights
        )
        if self.cluster_centers is None:
            raise ValueError("Cluster_centers must be provided for this objective.")
        if cluster_cost_method not in ["kmeans", "kmedians"]:
            raise ValueError(f"Method {cluster_cost_method} not supported. Supported cluster_cost_methods are 'kmeans' and 'kmedians'.")
        self.cluster_cost_method = cluster_cost_method


    def initialize_data(
        self, 
        X : NDArray,
        y : list[set[int]],
    ):
        """
        Sets the data for the objective.
        """
        assert isinstance(X, np.ndarray), 'X must be a numpy array.'
        assert len(X.shape) == 2, 'X must be a 2D array.'
        assert len(y) == X.shape[0], 'y must have the same number of elements as X has rows.'
        assert all(isinstance(label_set, set) for label_set in y), \
            'Each element of y must be a set of labels.'
        
        if self.weights is None:
            self.weights = np.ones(X.shape[0], dtype = float)
        else:
            assert len(self.weights) == X.shape[0], \
                'weights must have the same length as the number of samples in X.'
            
        self.X = X
        self.y = y

        self.label_set = unique_labels(y)
        self.n_labels = len(self.label_set)
        data_to_cluster_assignment = labels_to_assignment(
            y, n_labels = self.n_labels, ignore = {-1}
        )
        self.cluster_coverage_dict = assignment_to_dict(data_to_cluster_assignment)

        # Compute n x k distance matrix between data points and cluster centers.
        self.data_to_center_distances = np.zeros((self.X.shape[0], self.cluster_centers.shape[0]))
        for i in range(self.cluster_centers.shape[0]):
            if self.cluster_cost_method == "kmeans":
                self.data_to_center_distances[:, i] = np.sum((self.X - self.cluster_centers[i])**2, axis=1)
            else:  # self.cluster_cost_cluster_cost_method == "kmedians"
                self.data_to_center_distances[:, i] = np.sum(np.abs(self.X - self.cluster_centers[i]), axis=1)

        self.data_initialized = True


    def reward(
        self,
        selected_decisions_info: dict[Decision, dict[str, any]],
    ) -> float:
        """
        Computes the reward from the selected decisions.

        Args:
            selected_decisions_info (dict[Decision, dict[str, any]]): A dictionary mapping each
                selected decision to its information (points, coverage, length, label).
        Returns:
            reward (float): The reward from the selected decisions.
        """
        total_coverage = set()
        for decision, info in selected_decisions_info.items():
            r_coverage = info['coverage']
            total_coverage = total_coverage.union(r_coverage)
        return np.sum(self.weights[list(total_coverage)])


    def cost(
        self,
        selected_decisions_info: dict[Decision, dict[str, any]]
    ) -> float:
        """
        Computes the cost of the selected decisions.

        Args:
            selected_decisions_info (dict[Decision, dict[str, any]]): A dictionary mapping each
                selected decision to its information (points, coverage, length, label).
        Returns:
            cost (float): The cost of the selected decisions.
        """
        total_cost = 0.0
        length_penalty = 0.0
        for decision, info in selected_decisions_info.items():
            r_coverage = info['coverage']
            r_center = info['label']
            if r_coverage:  # Only process non-empty sets
                coverage_array = info['coverage_array']
                cluster_cost = np.sum(self.data_to_center_distances[coverage_array, r_center])
                total_cost += cluster_cost
            length_penalty += self.alpha_val * info['length']

        return total_cost + length_penalty


    def marginal_reward(
        self,
        decision_info: dict[str, any],
        total_coverage : set[int],
        total_cluster_coverage : dict[int, set[int]]
    ) -> float:
        """
        Computes the marginal reward as new coverage from selected decision.

        Args:
            decision_info (dict[str, any]): A dictionary containing information about the decision 
                being considered.
            cluster_coverage (dict[int, set[int]]): A dictionary mapping each cluster label to the 
                set of data points already covered by selected decisions.
        Returns:
            coverage (float): The coverage of the selected decisions.
        """
        r_coverage = decision_info['coverage']
        new_coverage = r_coverage.difference(total_coverage)
        new_coverage_array = np.fromiter(new_coverage, dtype=np.int64)
        new_points_weighted = np.sum(self.weights[new_coverage_array])
        return new_points_weighted


####################################################################################################