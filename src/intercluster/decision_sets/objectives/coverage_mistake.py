import numpy as np
from numpy.typing import NDArray
from intercluster import Condition, Rule, Decision
from intercluster.utils import (
    assignment_to_dict, labels_to_assignment, unique_labels, satisfies_rule, map_rules_to_decisions
)
from .objectives import Objective


####################################################################################################


class CoverageMistakeObjective(Objective):
    """
    Objective that selects rules based on a coverage and mistake objective.

    Args:
        n_select (int): The *maximum* number of rules to select.
        lambda_val (float): A hyperparameter that controls tradeoff between coverage and overlap.
            Defaults to 1.0.
    """
    def __init__(
        self,
        n_select : int,
        alpha_val : float = 0.0,
        lambda_val : float = None, 
        weights : NDArray = None
    ):
        """
        Args:
            n_select (int): The *maximum* number of rules to select.
            lambda_val (float): A hyperparameter that controls tradeoff between coverage and overlap.
            alpha_val (float): A hyperparameter for tuning the size of the selected rules.
                Larger values penalize longer rules more heavily. Defaults to 1.0.
        """
        super().__init__(
            n_select = n_select,
            alpha_val = alpha_val,
            lambda_val = lambda_val,
            weights = weights
        )


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
        total_cluster_coverage = {}
        for decision, info in selected_decision_info.items():
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
        selected_decisions_info: dict[Decision, dict[str, any]],
    ) -> float:
        """
        Computes the cost of the selected rules.

        Args:
            selected_decisions_info (dict[Decision, dict[str, any]]): A dictionary mapping each 
                selected decision to its information (coverage, cluster coverage, length, label).
        Returns:
            cost (float): The cost of the selected rules.
        """
        total_mistakes = 0
        length_penalty = 0
        for decision, info in selected_decisions_info.items():
            r_coverage = info['coverage']
            r_cluster_coverage = info['cluster_coverage']
            mistakes = r_coverage.difference(r_cluster_coverage)
            total_mistakes += len(mistakes)
            length_penalty += self.alpha_val * info['length']

        return total_mistakes + length_penalty


    def marginal_reward(
        self,
        decision_info: dict[str, any],
        total_coverage : set[int],
        total_cluster_coverage : dict[int, set[int]]
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
        r_cluster_coverage = decision_info['cluster_coverage']
        s_coverage = total_cluster_coverage[decision_info['label']]
        new_coverage = r_cluster_coverage.difference(s_coverage)
        new_coverage_array = np.fromiter(new_coverage, dtype=np.int64)
        new_coverage_weighted = np.sum(self.weights[new_coverage_array])
        return new_coverage_weighted


####################################################################################################


class TotalCoverageMistakeObjective(Objective):
    """
    Objective that selects rules based on a coverage and mistake objective. The difference 
    with CoverageMistakeObjective is that the coverage is computed across all clusters,
    rather than within each cluster.

    Args:
        n_select (int): The *maximum* number of rules to select.
        lambda_val (float): A hyperparameter that controls tradeoff between coverage and overlap.
            Defaults to 1.0.
    """
    def __init__(
        self,
        n_select : int,
        alpha_val : float = 0.0,
        lambda_val : float = None,
        weights : NDArray = None
    ):
        """
        Args:
            n_select (int): The *maximum* number of rules to select.
            lambda_val (float): A hyperparameter that controls tradeoff between coverage and overlap.
            alpha_val (float): A hyperparameter for tuning the size of the selected rules.
                Larger values penalize longer rules more heavily. Defaults to 1.0.
        """
        super().__init__(
            n_select = n_select,
            alpha_val = alpha_val,
            lambda_val = lambda_val,
            weights = weights
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
        total_mistakes = 0
        length_penalty = 0
        for decision, info in selected_decisions_info.items():
            r_coverage = info['coverage']
            r_cluster_coverage = info['cluster_coverage']
            mistakes = r_coverage.difference(r_cluster_coverage)
            total_mistakes += len(mistakes)
            length_penalty += self.alpha_val * info['length']

        return total_mistakes + length_penalty


    def marginal_reward(
        self,
        decision_info: dict[str, any],
        total_coverage : set[int],
        total_cluster_coverage : dict[int, set[int]]
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
        r_coverage = decision_info['coverage']
        new_coverage = r_coverage.difference(total_coverage)
        new_coverage_array = np.fromiter(new_coverage, dtype=np.int64)
        new_points_weighted = np.sum(self.weights[new_coverage_array])
        return new_points_weighted


####################################################################################################