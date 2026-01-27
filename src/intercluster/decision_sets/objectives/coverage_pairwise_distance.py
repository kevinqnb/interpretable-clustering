import numpy as np
from numpy.typing import NDArray
from intercluster import Decision
from intercluster.measurements import rule_pairwise_difference
from .objectives import Objective
from pathlib import Path
from typing import Any, Union



####################################################################################################


class CoveragePairwiseDistanceObjective(Objective):
    """
    Objective that selects rules based on a coverage and mistake objective.

    Args:
         n_select (int): The *maximum* number of rules to select.
        lambda_val (float): A hyperparameter that controls tradeoff between coverage and overlap.
            Defaults to 1.0.
        alpha_val (float): A hyperparameter for tuning the size of the selected rules.
            Larger values penalize longer rules more heavily. Defaults to 1.0.
        weights (NDArray): (n,) Array of weights for each data point. Defaults to None,
        selection_algorithm (str): The selection algorithm to use. Options are
            'distorted-greedy' and 'lazy-greedy'. Defaults to 'distorted-greedy'.
        precomputed_path (Union[str, Path]): Path to precomputed data for the objective. 
            Defaults to None.
        output_path (Union[str, Path]): Path to save output data. Defaults to None.
        pack_bits (bool): Whether to pack boolean matrices as bit vectors for memory efficiency.
            Defaults to True.
    """
    def __init__(
        self,
        n_select : int,
        alpha_val : float = 0.0,
        lambda_val : float = None, 
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
            weights = weights,
            selection_algorithm = selection_algorithm,
            precomputed_path = precomputed_path,
            output_path = output_path,
            pack_bits = pack_bits,
        )
        self.cluster_sizes = None


    def reward(
        self,
        selected_decision_info: dict[Decision, dict[str, any]],
    ) -> float:
        """
        Computes the reward from the selected rules.

        Args:
            selected_decision_info (dict[Decision, dict[str, any]]): A dictionary mapping each 
                selected decision to its information (coverage, cluster coverage, length, label).
        Returns:
            reward (float): The reward from the selected rules.
        """
        if self.pack_bits:
            total_by_cluster = np.zeros((self.n_labels, self.cluster_membership_packed.shape[1]), dtype=np.uint8)
            for _, info in selected_decision_info.items():
                lbl = int(info['label'])
                ridx = int(info['coverage_idx'])
                rule_bits = self.rule_coverage_packed[ridx:ridx + 1]
                cluster_bits = self.cluster_membership_packed[lbl:lbl + 1]
                total_by_cluster[lbl:lbl + 1] = np.bitwise_or(total_by_cluster[lbl:lbl + 1], np.bitwise_and(rule_bits, cluster_bits))

            total_weighted_coverage = 0.0
            for lbl in range(self.n_labels):
                bits = np.unpackbits(total_by_cluster[lbl:lbl + 1], axis=-1)[0][: self.n_samples]
                idxs = np.flatnonzero(bits)
                if idxs.size:
                    total_weighted_coverage += float(np.sum(self.weights[idxs]))
            return total_weighted_coverage

        total_by_cluster = np.zeros((self.n_labels, self.n_samples), dtype=np.bool_)
        for _, info in selected_decision_info.items():
            lbl = int(info['label'])
            ridx = int(info['coverage_idx'])
            total_by_cluster[lbl] |= (self.rule_coverage_packed[ridx] & self.cluster_membership_packed[lbl])

        total_weighted_coverage = 0.0
        for lbl in range(self.n_labels):
            idxs = np.flatnonzero(total_by_cluster[lbl])
            if idxs.size:
                total_weighted_coverage += float(np.sum(self.weights[idxs]))
        return total_weighted_coverage

    '''
    def cost(
        self,
        selected_decisions_info: dict[Decision, dict[str, any]],
        alpha_val : float = None,
    ) -> float:
        """
        Computes the cost of the selected rules.

        Args:
            selected_decisions_info (dict[Decision, dict[str, any]]): A dictionary mapping each 
                selected decision to its information (coverage, cluster coverage, length, label).
        Returns:
            cost (float): The cost of the selected rules.
        """
        if alpha_val is None:
            alpha_val = self.alpha_val

        total_pairwise_distance = 0.0
        length_penalty = 0.0

        # NOTE: this objective needs baseline_labels (coverage_labels). We now reconstruct them
        # on-demand from coverage_idx + self.y.
        for decision, info in selected_decisions_info.items():
            #baseline_labels = self.get_coverage_labels(decision)
            ridx = int(info['coverage_idx'])
            idxs = self._iter_covered_indices_from_rule_idx(ridx)
            baseline_labels = [self.y[int(i)] for i in idxs]
            total_pairwise_distance += float(
                rule_pairwise_difference(
                    baseline_labels,
                    percentage=False,
                )
            )
            length_penalty += float(alpha_val) * float(info['length'])

        return float(total_pairwise_distance + length_penalty)

    '''

    def cost(
        self,
        selected_decisions_info: dict[Decision, dict[str, any]],
        alpha_val : float = None,
    ) -> float:
        """
        Computes the cost of the selected rules.

        Args:
            selected_decisions_info (dict[Decision, dict[str, any]]): A dictionary mapping each 
                selected decision to its information (coverage, cluster coverage, length, label).
        Returns:
            cost (float): The cost of the selected rules.
        """
        if alpha_val is None:
            alpha_val = self.alpha_val

        # Cache cluster sizes as *counts of items* per label.
        # NOTE: when `self.pack_bits` is True, `cluster_membership_packed` is packed (uint8 bytes)
        # so a plain sum() is NOT a count; we need a popcount.
        if self.cluster_sizes is None:
            if self.pack_bits:
                self.cluster_sizes = np.array(
                    [
                        int(
                            np.unpackbits(self.cluster_membership_packed[l:l+1], axis=-1)[0][: self.n_samples].sum()
                        )
                        for l in range(self.n_labels)
                    ],
                    dtype=np.int64,
                )
            else:
                self.cluster_sizes = np.sum(self.cluster_membership_packed, axis=1).astype(np.int64, copy=False)

        # Precompute per-sample weight = size of the cluster it was originally assigned to.
        # `self.y` is a list of singleton sets: [{i}, {j}, ...]
        y_labels = np.fromiter((next(iter(s)) for s in self.y), dtype=np.int64, count=self.n_samples)
        weight_by_sample = self.cluster_sizes[y_labels]

        total_pairwise = 0
        length_penalty = 0.0

        if self.pack_bits:
            # mistakes_bits = rule_bits & ~cluster_bits
            # weighted_mistakes = sum(weight_by_sample[i] for i in mistakes)
            for _, info in selected_decisions_info.items():
                lbl = int(info['label'])
                ridx = int(info['coverage_idx'])

                rule_bits = self.rule_coverage_packed[ridx:ridx + 1]
                cluster_bits = self.cluster_membership_packed[lbl:lbl + 1]

                assigned_cluster_size = int(self.cluster_sizes[lbl])

                mistakes_bits = np.bitwise_and(rule_bits, np.bitwise_not(cluster_bits))
                mistakes_mask = np.unpackbits(mistakes_bits, axis=-1)[0][: self.n_samples].astype(np.bool_, copy=False)

                weighted_mistakes = int(np.sum(weight_by_sample[mistakes_mask]))

                total_pairwise += weighted_mistakes * assigned_cluster_size
                length_penalty += float(alpha_val) * float(info['length'])

            return float(total_pairwise) + float(length_penalty)

        # Unpacked/boolean path
        for _, info in selected_decisions_info.items():
            lbl = int(info['label'])
            ridx = int(info['coverage_idx'])

            rule_mask = self.rule_coverage_packed[ridx]
            assigned_cluster_size = int(self.cluster_sizes[lbl])

            mistakes_mask = rule_mask & ~self.cluster_membership_packed[lbl]
            weighted_mistakes = int(np.sum(weight_by_sample[mistakes_mask]))

            total_pairwise += weighted_mistakes * assigned_cluster_size
            length_penalty += float(alpha_val) * float(info['length'])

        return float(total_pairwise) + float(length_penalty)


    def marginal_reward(
        self,
        decision_info: dict[str, any],
        total_coverage,
        total_cluster_coverage,
    ) -> float:
        """
        Computes the marginal reward as new coverage from selected rule.

        Args:
            decision_info (dict): A dictionary containing information 
                for a given decision (coverage, cluster coverage, length, label).
            cluster_coverage (dict[int, set[int]]): A dictionary mapping each cluster
                label to the set of data points already covered by selected rules.
        
        Returns:
            coverage (float): The coverage of the selected rules.
        """
        lbl = int(decision_info['label'])
        ridx = int(decision_info['coverage_idx'])

        if self.pack_bits:
            rule_bits = self.rule_coverage_packed[ridx:ridx + 1]
            cluster_bits = self.cluster_membership_packed[lbl:lbl + 1]
            r_cluster_bits = np.bitwise_and(rule_bits, cluster_bits)
            new_bits = np.bitwise_and(r_cluster_bits, np.bitwise_not(total_cluster_coverage[lbl:lbl + 1]))
            new_mask = np.unpackbits(new_bits, axis=-1)[0][: self.n_samples].astype(np.bool_, copy=False)
            return float(np.sum(self.weights[new_mask]))

        r_cluster_mask = self.rule_coverage_packed[ridx] & self.cluster_membership_packed[lbl]
        new_mask = r_cluster_mask & ~total_cluster_coverage[lbl]
        return float(np.sum(self.weights[new_mask]))


####################################################################################################


class TotalCoveragePairwiseDistanceObjective(Objective):
    """
    Objective that selects rules based on a coverage and mistake objective. The difference 
    with CoverageMistakeObjective is that the coverage is computed across all clusters,
    rather than within each cluster.

    Args:
        n_select (int): The *maximum* number of rules to select.
        lambda_val (float): A hyperparameter that controls tradeoff between coverage and overlap.
            Defaults to 1.0.
        alpha_val (float): A hyperparameter for tuning the size of the selected rules.
            Larger values penalize longer rules more heavily. Defaults to 1.0.
        weights (NDArray): (n,) Array of weights for each data point. Defaults to None,
        selection_algorithm (str): The selection algorithm to use. Options are
            'distorted-greedy' and 'lazy-greedy'. Defaults to 'distorted-greedy'.
        precomputed_path (Union[str, Path]): Path to precomputed data for the objective. Defaults to None.
        output_path (Union[str, Path]): Path to save output data. Defaults to None.
        pack_bits (bool): Whether to pack boolean matrices as bit vectors for memory efficiency. Defaults to True.
    """
    def __init__(
        self,
        n_select : int,
        alpha_val : float = 0.0,
        lambda_val : float = None,
        weights : NDArray = None,
        selection_algorithm : str = 'distorted-greedy',
        precomputed_path : Union[str, Path] = None,
        output_path : Union[str, Path] = None,
        pack_bits : bool = True,
    ):
        super().__init__(
            n_select = n_select,
            alpha_val = alpha_val,
            lambda_val = lambda_val,
            weights = weights,
            selection_algorithm = selection_algorithm,
            precomputed_path = precomputed_path,
            output_path = output_path,
            pack_bits = pack_bits,
        )


    def reward(
        self,
        selected_decisions_info: dict[Decision, dict[str, any]],
    ) -> float:
        """
        Computes the reward from the selected decisions.

        Args:
            selected_decisions_info (dict[Decision, dict[str, any]]): A dictionary mapping 
                each selected decision to its information (points, coverage, length, label).
        Returns:
            reward (float): The reward from the selected decisions.
        """
        if self.pack_bits:
            total_bits = np.zeros((1, self.rule_coverage_packed.shape[1]), dtype=np.uint8)
            for _, info in selected_decisions_info.items():
                ridx = int(info['coverage_idx'])
                total_bits = np.bitwise_or(total_bits, self.rule_coverage_packed[ridx:ridx + 1])
            mask = np.unpackbits(total_bits, axis=-1)[0][: self.n_samples].astype(np.bool_, copy=False)
            return float(np.sum(self.weights[mask]))

        total_mask = np.zeros((self.n_samples,), dtype=np.bool_)
        for _, info in selected_decisions_info.items():
            ridx = int(info['coverage_idx'])
            total_mask |= self.rule_coverage_packed[ridx]
        return float(np.sum(self.weights[total_mask]))


    def cost(
        self,
        selected_decisions_info: dict[Decision, dict[str, any]],
        alpha_val : float = None,
    ) -> float:
        """
        Computes the cost of the selected rules.

        Args:
            selected_decisions_info (dict[Decision, dict[str, any]]): A dictionary mapping each 
                selected decision to its information (coverage, cluster coverage, length, label).
        Returns:
            cost (float): The cost of the selected rules.
        """
        if alpha_val is None:
            alpha_val = self.alpha_val

        # Cache cluster sizes as *counts of items* per label.
        # NOTE: when `self.pack_bits` is True, `cluster_membership_packed` is packed (uint8 bytes)
        # so a plain sum() is NOT a count; we need a popcount.
        if self.cluster_sizes is None:
            if self.pack_bits:
                self.cluster_sizes = np.array(
                    [
                        int(
                            np.unpackbits(self.cluster_membership_packed[l:l+1], axis=-1)[0][: self.n_samples].sum()
                        )
                        for l in range(self.n_labels)
                    ],
                    dtype=np.int64,
                )
            else:
                self.cluster_sizes = np.sum(self.cluster_membership_packed, axis=1).astype(np.int64, copy=False)

        # Precompute per-sample weight = size of the cluster it was originally assigned to.
        # `self.y` is a list of singleton sets: [{i}, {j}, ...]
        y_labels = np.fromiter((next(iter(s)) for s in self.y), dtype=np.int64, count=self.n_samples)
        weight_by_sample = self.cluster_sizes[y_labels]

        total_pairwise = 0
        length_penalty = 0.0

        if self.pack_bits:
            # mistakes_bits = rule_bits & ~cluster_bits
            # weighted_mistakes = sum(weight_by_sample[i] for i in mistakes)
            for _, info in selected_decisions_info.items():
                lbl = int(info['label'])
                ridx = int(info['coverage_idx'])

                rule_bits = self.rule_coverage_packed[ridx:ridx + 1]
                cluster_bits = self.cluster_membership_packed[lbl:lbl + 1]

                assigned_cluster_size = int(self.cluster_sizes[lbl])

                mistakes_bits = np.bitwise_and(rule_bits, np.bitwise_not(cluster_bits))
                mistakes_mask = np.unpackbits(mistakes_bits, axis=-1)[0][: self.n_samples].astype(np.bool_, copy=False)

                weighted_mistakes = int(np.sum(weight_by_sample[mistakes_mask]))

                total_pairwise += weighted_mistakes * assigned_cluster_size
                length_penalty += float(alpha_val) * float(info['length'])

            return float(total_pairwise) + float(length_penalty)

        # Unpacked/boolean path
        for _, info in selected_decisions_info.items():
            lbl = int(info['label'])
            ridx = int(info['coverage_idx'])

            rule_mask = self.rule_coverage_packed[ridx]
            assigned_cluster_size = int(self.cluster_sizes[lbl])

            mistakes_mask = rule_mask & ~self.cluster_membership_packed[lbl]
            weighted_mistakes = int(np.sum(weight_by_sample[mistakes_mask]))

            total_pairwise += weighted_mistakes * assigned_cluster_size
            length_penalty += float(alpha_val) * float(info['length'])

        return float(total_pairwise) + float(length_penalty)


    def marginal_reward(
        self,
        decision_info: dict[str, any],
        total_coverage,
        total_cluster_coverage,
    ) -> float:
        """
        Computes the marginal reward as new coverage from selected decision.

        Args:
            decision_info (dict): A dictionary containing information about the decision being considered.
            cluster_coverage (dict[int, set[int]]): A dictionary mapping each cluster
                label to the set of data points already covered by selected decisions.        
        Returns:
            coverage (float): The coverage of the selected decisions.
        """
        ridx = int(decision_info['coverage_idx'])

        if self.pack_bits:
            rule_bits = self.rule_coverage_packed[ridx:ridx + 1]
            new_bits = np.bitwise_and(rule_bits, np.bitwise_not(total_coverage))
            new_mask = np.unpackbits(new_bits, axis=-1)[0][: self.n_samples].astype(np.bool_, copy=False)
            return float(np.sum(self.weights[new_mask]))

        new_mask = self.rule_coverage_packed[ridx] & ~total_coverage
        return float(np.sum(self.weights[new_mask]))


####################################################################################################