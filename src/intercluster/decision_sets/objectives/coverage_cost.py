import numpy as np
from numpy.typing import NDArray
from intercluster import Decision
from intercluster.utils import (
    assignment_to_dict,
    labels_to_assignment,
    unique_labels, 
    _pack_bool_matrix,
    _unpack_bool_matrix,
)
from .objectives import Objective
from pathlib import Path
from typing import Any, Union


####################################################################################################


class CoverageCostObjective(Objective):
    """
    Objective that selects rules based on a coverage and cluster cost objective.

    Args:
        n_select (int): The *maximum* number of rules to select.
        lambda_val (float): A hyperparameter that controls tradeoff between coverage and overlap.
            Defaults to 1.0.
        alpha_val (float): A hyperparameter for tuning the size of the selected rules.
            Larger values penalize longer rules more heavily. Defaults to 1.0.
        cluster_centers (NDArray): (k x d) array where each row i is the given 
                representative for cluster i.
        cluster_cost_method (str): The cluster_cost_method used to compute cluster costs. 
            Currently only "kmeans" or "kmedians" are supported.
        weights (NDArray): (n,) Array of weights for each data point. Defaults to None,
        selection_algorithm (str): The selection algorithm to use. Options are
            'distorted-greedy' and 'lazy-greedy'. Defaults to 'distorted-greedy'.
        precomputed_path (Union[str, Path]): Path to precomputed data for the objective. Defaults to None.
        output_path (Union[str, Path]): Path to save output data. Defaults to None.
        pack_bits (bool): Whether to pack boolean matrices as bit vectors for memory efficiency. Defaults to True.
    """
    def __init__(
            self,
            n_select : int = 0,
            alpha_val : float = 0.0,
            lambda_val : float = None,
            cluster_centers : NDArray = None,
            cluster_cost_method : str = "kmeans",
            weights : NDArray = None,
            selection_algorithm : str = 'distorted-greedy',
            precomputed_path: Union[str, Path] = None,
            output_path: Union[str, Path] = None,
            pack_bits: bool = True,
        ):
        super().__init__(
            n_select = n_select,
            alpha_val = alpha_val,
            lambda_val = lambda_val,
            cluster_centers = cluster_centers,
            weights = weights,
            selection_algorithm = selection_algorithm,
            precomputed_path = precomputed_path,
            output_path = output_path,
            pack_bits = pack_bits,
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
        self.n_samples = X.shape[0]
            
        self.label_set = unique_labels(y)
        self.n_labels = len(self.label_set)

        if self.precomputed:
            self.data_initialized = True
            return
        
        cluster_membership = labels_to_assignment(
            y, n_labels = self.n_labels
        ).T
        self.cluster_membership_packed = _pack_bool_matrix(
            cluster_membership
        ) if self.pack_bits else cluster_membership

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
        if self.pack_bits:
            total_by_cluster = np.zeros((self.n_labels, self.cluster_membership_packed.shape[1]), dtype=np.uint8)
            for _, info in selected_decisions_info.items():
                lbl = int(info['label'])
                ridx = int(info['coverage_idx'])
                rule_bits = self.rule_coverage_packed[ridx:ridx + 1]
                cluster_bits = self.cluster_membership_packed[lbl:lbl + 1]
                total_by_cluster[lbl:lbl + 1] = np.bitwise_or(
                    total_by_cluster[lbl:lbl + 1], np.bitwise_and(rule_bits, cluster_bits)
                )

            total_weighted_coverage = 0.0
            for lbl in range(self.n_labels):
                bits = np.unpackbits(total_by_cluster[lbl:lbl + 1], axis=-1)[0][: self.n_samples]
                idxs = np.flatnonzero(bits)
                if idxs.size:
                    total_weighted_coverage += float(np.sum(self.weights[idxs]))
            return total_weighted_coverage

        total_by_cluster = np.zeros((self.n_labels, self.n_samples), dtype=np.bool_)
        for _, info in selected_decisions_info.items():
            lbl = int(info['label'])
            ridx = int(info['coverage_idx'])
            total_by_cluster[lbl] |= (self.rule_coverage_packed[ridx] & self.cluster_membership_packed[lbl])

        total_weighted_coverage = 0.0
        for lbl in range(self.n_labels):
            idxs = np.flatnonzero(total_by_cluster[lbl])
            if idxs.size:
                total_weighted_coverage += float(np.sum(self.weights[idxs]))
        return total_weighted_coverage


    def cost(
        self,
        selected_decisions_info: dict[Decision, dict[str, any]],
        alpha_val : float = None,
    ) -> float:
        if alpha_val is None:
            alpha_val = self.alpha_val

        total_cost = 0.0
        length_penalty = 0.0

        for _, info in selected_decisions_info.items():
            ridx = int(info['coverage_idx'])
            center = int(info['label'])

            if self.pack_bits:
                bits = np.unpackbits(self.rule_coverage_packed[ridx:ridx + 1], axis=-1)[0][: self.n_samples]
                idxs = np.flatnonzero(bits)
            else:
                idxs = np.flatnonzero(self.rule_coverage_packed[ridx])

            if idxs.size:
                total_cost += float(np.sum(self.data_to_center_distances[idxs, center]))

            length_penalty += float(alpha_val) * float(info['length'])

        return float(total_cost + length_penalty)


    def marginal_reward(
        self,
        decision_info: dict[str, any],
        total_coverage : NDArray,
        total_cluster_coverage : NDArray
    ) -> float:
        lbl = int(decision_info['label'])
        ridx = int(decision_info['coverage_idx'])

        if self.pack_bits:
            rule_bits = self.rule_coverage_packed[ridx:ridx + 1]
            cluster_bits = self.cluster_membership_packed[lbl:lbl + 1]
            r_cluster_bits = np.bitwise_and(rule_bits, cluster_bits)
            new_bits = np.bitwise_and(
                r_cluster_bits, np.bitwise_not(total_cluster_coverage[lbl:lbl + 1])
            )
            new_mask = np.unpackbits(new_bits, axis=-1)[0][: self.n_samples].astype(np.bool_, copy=False)
            return float(np.sum(self.weights[new_mask]))

        r_cluster_mask = self.rule_coverage_packed[ridx] & self.cluster_membership_packed[lbl]
        new_mask = r_cluster_mask & ~total_cluster_coverage[lbl]
        return float(np.sum(self.weights[new_mask]))


####################################################################################################


class TotalCoverageCostObjective(Objective):
    """
    Objective that selects rules based on a coverage and cluster cost objective. The difference
    with CoverageCostObjective is that the coverage is computed across all clusters,
    rather than within each cluster.

    Args:
        n_select (int): The *maximum* number of rules to select.
        lambda_val (float): A hyperparameter that controls tradeoff between coverage and overlap.
            Defaults to 1.0.
        alpha_val (float): A hyperparameter for tuning the size of the selected rules.
            Larger values penalize longer rules more heavily. Defaults to 1.0.
        cluster_centers (NDArray): (k x d) array where each row i is the given 
                representative for cluster i.
        cluster_cost_method (str): The cluster_cost_method used to compute cluster costs. 
            Currently only "kmeans" or "kmedians" are supported.
        weights (NDArray): (n,) Array of weights for each data point. Defaults to None,
        selection_algorithm (str): The selection algorithm to use. Options are
            'distorted-greedy' and 'lazy-greedy'. Defaults to 'distorted-greedy'.
        precomputed_path (Union[str, Path]): Path to precomputed data for the objective. Defaults to None.
        output_path (Union[str, Path]): Path to save output data. Defaults to None.
        pack_bits (bool): Whether to pack boolean matrices as bit vectors for memory efficiency. Defaults to True.
    """
    def __init__(
            self,
            n_select : int = 0,
            alpha_val : float = 0.0,
            lambda_val : float = None,
            cluster_centers : NDArray = None,
            cluster_cost_method : str = "kmeans",
            weights : NDArray = None,
            selection_algorithm : str = "distorted-greedy",
            precomputed_path : Union[str, Path] = None,
            output_path : Union[str, Path] = None,
            pack_bits : bool = True
        ):
        super().__init__(
            n_select = n_select,
            alpha_val = alpha_val,
            lambda_val = lambda_val,
            cluster_centers = cluster_centers,
            weights = weights,
            selection_algorithm = selection_algorithm,
            precomputed_path = precomputed_path,
            output_path = output_path,
            pack_bits = pack_bits,
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
        self.n_samples = X.shape[0]

        self.label_set = unique_labels(y)
        self.n_labels = len(self.label_set)
        
        if self.precomputed:
            self.data_initialized = True
            return
        
        cluster_membership = labels_to_assignment(
            y, n_labels = self.n_labels
        ).T
        self.cluster_membership_packed = _pack_bool_matrix(
            cluster_membership
        ) if self.pack_bits else cluster_membership

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
        if self.pack_bits:
            total_bits = np.zeros((1, self.rule_coverage_packed.shape[1]), dtype=np.uint8)
            for _, info in selected_decisions_info.items():
                total_bits = np.bitwise_or(
                    total_bits, self.rule_coverage_packed[int(info['coverage_idx']):int(info['coverage_idx']) + 1]
                )
            mask = np.unpackbits(total_bits, axis=-1)[0][: self.n_samples].astype(np.bool_, copy=False)
            return float(np.sum(self.weights[mask]))

        total_mask = np.zeros((self.n_samples,), dtype=np.bool_)
        for _, info in selected_decisions_info.items():
            total_mask |= self.rule_coverage_packed[int(info['coverage_idx'])]
        return float(np.sum(self.weights[total_mask]))


    def cost(
        self,
        selected_decisions_info: dict[Decision, dict[str, any]],
        alpha_val: float = None,
    ) -> float:
        if alpha_val is None:
            alpha_val = self.alpha_val

        total_cost = 0.0
        length_penalty = 0.0

        for _, info in selected_decisions_info.items():
            ridx = int(info['coverage_idx'])
            center = int(info['label'])

            if self.pack_bits:
                bits = np.unpackbits(self.rule_coverage_packed[ridx:ridx + 1], axis=-1)[0][: self.n_samples]
                idxs = np.flatnonzero(bits)
            else:
                idxs = np.flatnonzero(self.rule_coverage_packed[ridx])

            if idxs.size:
                total_cost += float(np.sum(self.data_to_center_distances[idxs, center]))

            length_penalty += float(alpha_val) * float(info['length'])

        return float(total_cost + length_penalty)


    def marginal_reward(
        self,
        decision_info: dict[str, any],
        total_coverage : NDArray,
        total_cluster_coverage : NDArray,
    ) -> float:
        ridx = int(decision_info['coverage_idx'])

        if self.pack_bits:
            rule_bits = self.rule_coverage_packed[ridx:ridx + 1]
            new_bits = np.bitwise_and(rule_bits, np.bitwise_not(total_coverage))
            new_mask = np.unpackbits(new_bits, axis=-1)[0][: self.n_samples].astype(np.bool_, copy=False)
            return float(np.sum(self.weights[new_mask]))

        new_mask = self.rule_coverage_packed[ridx] & ~total_coverage
        return float(np.sum(self.weights[new_mask]))


####################################################################################################