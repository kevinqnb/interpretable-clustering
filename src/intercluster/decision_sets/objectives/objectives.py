import numpy as np
from numpy.typing import NDArray
from intercluster import Condition, Rule, Decision
from intercluster.utils import (
    assignment_to_dict, labels_to_assignment, unique_labels, satisfies_rule, map_rules_to_decisions
)

####################################################################################################


class Objective:
    """
    Base class for a selector, which is used to select rules based on a given objective.

    Args:
        n_select (int): The *maximum* number of rules to select.
        alpha_val (float): A hyperparameter for tuning the size of the selected rules.
            Larger values penalize longer rules more heavily. Defaults to 0.0.
        lambda_val (float): A hyperparameter that controls tradeoff between reward and cost.
            Defaults to None, in which case it may be selected automatically.

    Attrs:
        name (str): Name of the objective.
        data_initialized (bool): Whether the data has been initialized.
        decision_set_initialized (bool): Whether the decision set has been initialized.
    """
    def __init__(
        self,
        n_select : int,
        alpha_val : float = 0.0,
        lambda_val : float = None,
        cluster_centers : NDArray = None,
        weights : NDArray = None,
    ):
        assert n_select > 0, 'n_select must be positive.'
        self.n_select = n_select

        assert alpha_val >= 0.0, 'alpha_val must be non-negative.'
        self.alpha_val = alpha_val

        assert lambda_val is None or lambda_val >= 0.0, 'lambda_val must be non-negative.'
        self.lambda_val = lambda_val

        assert cluster_centers is None or isinstance(cluster_centers, np.ndarray), \
            'cluster_centers must be a numpy array.'
        if cluster_centers is not None:
            assert len(cluster_centers.shape) == 2, 'cluster_centers must be a 2D array.'
        self.cluster_centers = cluster_centers

        assert weights is None or isinstance(weights, np.ndarray), \
            'weights must be a numpy array.'
        if weights is not None:
            assert len(weights.shape) == 1, 'weights must be a 1D array.'
        self.weights = weights

        self.data_initialized = False
        self.decision_set_initialized = False

        self.X = None
        self.y = None
        self.label_set = None
        self.n_labels = 0
        self.cluster_coverage_dict = None
        self.rule_to_decision_dict = None
        self.decision_info_dict = None


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
        self.data_initialized = True


    def initialize_decision_set(
        self,
        decision_set : set[Decision],
    ):
        """
        Sets the decisions for the objective to select from.
        """
        if self.data_initialized is False:
            raise ValueError('Data must be initialized before decisions.')

        assert isinstance(decision_set, set), 'decision_set must be a set.'
        assert all(isinstance(decision, Decision) for decision in decision_set), \
            'Each element of decision_set must be a Decision.'
        
        decision_labels = unique_labels([{d.label} for d in decision_set])
        if not decision_labels.issubset(self.label_set):
            raise ValueError(
                'Decisions must cover the same labels as the input data.'
            )

        # Isolate unique rules and map them to their decisions.
        self.rule_to_decision_dict = map_rules_to_decisions(decision_set)

        # Track info for each decision.
        self.decision_info_dict = {}
        for decision in decision_set:
            rule_coverage = set(list(satisfies_rule(self.X, decision.rule)))
            rule_cluster_coverage = self.cluster_coverage_dict[decision.label].intersection(
                rule_coverage
            )
            rule_length = len(decision.rule)
            self.decision_info_dict[decision] = {
                'coverage': rule_coverage,
                'coverage_array': np.fromiter(rule_coverage, dtype=np.int64),
                'cluster_coverage': rule_cluster_coverage,
                'length': rule_length,
                'label': decision.label
            }
        
        self.decision_set_initialized = True


    def set_lambda(self, lambda_val : float = None):
        """
        Sets the lambda value for the objective.

        Args:
            lambda_val (float): The new lambda value.
        """
        if lambda_val is None and not (self.data_initialized and self.decision_set_initialized):
            raise ValueError('Data and decision set must be initialized before setting lambda.')
        elif lambda_val is None:
            lambda_vals = self.compute_lambdas()
            if len(lambda_vals) == 0:
                lambda_val = 0.0
            else:
                lambda_val = lambda_vals[0]
    
        self.lambda_val = lambda_val


    def reward(
        self,
        selected_decisions_info: dict[Decision, dict[str, any]],
    ) -> float:
        """
        Computes the reward from the selected rules.

        Args:
            selected_decisions_info (dict[Decision, dict[str, any]]): A dictionary mapping each selected
                decision to its information, including labels, points, coverage, and lengths.
        Returns:
            reward (float): The reward from the selected decisions.
        """
        pass


    def cost(
        self,
        selected_decisions_info: dict[Decision, dict[str, any]],
    ) -> float:
        """
        Computes the cost of the selected rules.

        Args:
            selected_decisions_info (dict[Decision, dict[str, any]]): A dictionary mapping each selected
                decision to its information, including labels, points, coverage, and lengths.
        Returns:
            cost (float): The cost of the selected decisions.
        """
        pass


    def compute_objective(
        self,
        selected_decisions_info: dict[Decision, dict[str, any]],
    ) -> float:
        """
        Computes the objective value for the selected decisions.

        Args:
            selected_decisions_info (dict[Decision, dict[str, any]]): A dictionary mapping each selected
                decision to its information, including labels, points, coverage, and lengths.
        Returns:
            objective (float): The objective value for the selected decisions.
        """
        g = self.reward(selected_decisions_info)
        h = self.cost(selected_decisions_info)
        return g - self.lambda_val * h
    

    def compute_lambdas(self) -> NDArray:
        """
        Computes minimum value of lambda necessary for an approximation algorithm.

        Args:
            data (NDArray): (n x d) Data array.
            data_to_cluster_assignment (np.ndarray): Size (n x k) boolean array where entry (i,j) is 
                `True` if point i is assigned to cluster j and `False` otherwise. Data points may be 
                assigned to multiple clusters. 
            data_to_rules_assignment (NDArray): A boolean matrix where entry (i,j) is `True` if 
                data point i is assigned to rule j and `False` otherwise.
            rule_lengths (list[int]): A list of lengths for each rule.
                
        Returns:
            lambda_vals (NDArray): A sorted array of lambda values, starting from the minimum 
                most value for which the approximation guarantee holds, and increasing
                until reaching the maximum coverage/cost ratio seen
                for any (rule, cluster) assignment pair. 
        """
        ratios = []
        second_max_ratio = 0.0
        for rule in self.rule_to_decision_dict.keys():
            max_rule_ratio = 0.0
            second_max_rule_ratio = 0.0
            for decision in self.rule_to_decision_dict[rule]:
                decision_info = self.decision_info_dict[decision]

                r_coverage = decision_info['coverage']
                r_cluster_coverage = decision_info['cluster_coverage']
                r_length = decision_info['length']
                d_label = decision_info['label']
                d_info = {
                    decision: {
                        'coverage': r_coverage,
                        'coverage_array': np.fromiter(r_coverage, dtype=np.int64),
                        'cluster_coverage': r_cluster_coverage,
                        'length': r_length,
                        'label': d_label
                    }
                }
                g = self.reward(d_info)
                h = self.cost(d_info)

                if h > 0:
                    ratio = g / h
                else:
                    ratio = np.inf

                if ratio > max_rule_ratio:
                    second_max_rule_ratio = max_rule_ratio
                    max_rule_ratio = ratio
                elif ratio > second_max_rule_ratio:
                    second_max_rule_ratio = ratio

            ratios.append(max_rule_ratio)
            if second_max_rule_ratio > second_max_ratio:
                second_max_ratio = second_max_rule_ratio
                    
        ratios = [r for r in ratios if r >= second_max_ratio]
        if second_max_ratio == 0.0:
            return np.sort(ratios)
        return np.sort(ratios + [second_max_ratio])


    def marginal_reward(
        self,
        decision_info: dict[str, any],
        total_coverage : set[int],
        cluster_coverage : dict[int, set[int]]
    ) -> float:
        """
        Computes the marginal reward from selected decision.

        Args:
            decision_info (dict): A dictionary containing information about the decision being considered.
            cluster_coverage (dict[int, set[int]]): A dictionary mapping each cluster
                label to the set of data points already covered by selected decisions.
        
        Returns:
            coverage (float): The coverage of the selected decisions.
        """
        pass


    def marginal_cost(
        self,
        decision_info: dict[str, any],
        total_coverage : set[int],
        cluster_coverage : dict[int, set[int]]
    ) -> float:
        """
        Computes the marginal cost of the selected decision.

        Args:
            decision_info (dict): A dictionary containing information about the decision being considered.
            cluster_coverage (dict[int, set[int]]): A dictionary mapping each cluster
                label to the set of data points already covered by selected decisions.
        
        Returns:
            cost (float): The cost of the selected decisions.
        """
        pass        


    def select(
        self,
    ) -> NDArray[np.int64]:
        """
        Selects a subset rules using a distorted greedy algorithm. For more information 
        on the algorithm, see the following paper:
        "Submodular Maximization Beyond Non-Negativity: Guarantees, Fast Algorithms, and Applications"
        by Harshaw el al., ICML 2019.
        Args:
            
        Returns:
            decision_set (Set[Decision]): The selected set of decisions.
        """
        if not (self.data_initialized and self.decision_set_initialized):
            raise ValueError('Data and decisions must be initialized before selection.')

        total_coverage = set()
        total_cluster_coverage = {l: set() for l in range(self.n_labels)}
        selected_decisions = set()
        discarded_decisions = set()
        for i in range(self.n_select):
            best_decision = None
            best_decision_score = 0.0

            # NOTE: Should this iterate over decisions in a sorted order?
            for decision, decision_info in self.decision_info_dict.items():
                if (decision not in selected_decisions) and (decision not in discarded_decisions):
                    g = self.marginal_reward(
                        decision_info,
                        total_coverage,
                        total_cluster_coverage
                    )

                    h = self.marginal_cost(
                        decision_info,
                        total_coverage,
                        total_cluster_coverage
                    )

                    # Early discard since the marginal reward will only decrease from here on out, 
                    # and its score coefficient will be at most 1.
                    # Therefore if g - lambda * c <= 0, the score will never be positive, 
                    # and it can never be selected.
                    if g - self.lambda_val * h <= 0:
                        discarded_decisions.add(decision)
                    
                    score = (1 - 1/self.n_select)**(self.n_select - (i + 1)) * g - self.lambda_val * h

                    if score > best_decision_score:
                        best_decision = decision
                        best_decision_score = score

            if best_decision_score > 0:
                selected_decisions.add(best_decision)
                best_decision_label = self.decision_info_dict[best_decision]['label']
                best_decision_coverage = self.decision_info_dict[best_decision]['coverage']
                best_decision_cluster_coverage = self.decision_info_dict[best_decision]['cluster_coverage']
                total_cluster_coverage[best_decision_label] = total_cluster_coverage[
                    best_decision_label
                ].union(
                    best_decision_cluster_coverage
                )
                total_coverage = total_coverage.union(best_decision_coverage)

        # Compute final objective value
        self.reward_value = self.reward(
            {decision: self.decision_info_dict[decision] for decision in selected_decisions},
        )
        self.cost_value = self.cost(
            {decision: self.decision_info_dict[decision] for decision in selected_decisions},
        )
        self.objective_value = self.compute_objective(
            {decision: self.decision_info_dict[decision] for decision in selected_decisions},
        )
        return selected_decisions


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


    def marginal_cost(
        self,
        decision_info: dict[str, any],
        total_coverage : set[int],
        total_cluster_coverage: dict[int, set[int]],
    ) -> float:
        """
        Computes the marginal cost as the number of mistakes made by the selected decision.

        Args:
            decision_info (dict): A dictionary containing information about the decision
                being considered.
            cluster_coverage (dict[int, set[int]]): A dictionary mapping each cluster
                label to the set of data points already covered by selected rules.
        
        Returns:
            cost (float): The cost of the selected rules.
        """
        r_coverage = decision_info['coverage']
        r_cluster_coverage = decision_info['cluster_coverage']
        mistakes = r_coverage.difference(r_cluster_coverage)
        return len(mistakes) + self.alpha_val * decision_info['length']


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


    def marginal_cost(
        self,
        decision_info: dict[str, any],
        total_coverage : set[int],
        total_cluster_coverage: dict[int, set[int]]
    ) -> float:
        """
        Computes the marginal cost as the number of mistakes made by the selected decision.

        Args:
            decision_info (dict): A dictionary containing information about the decision being considered.
            cluster_coverage (dict[int, set[int]]): A dictionary mapping each cluster
                label to the set of data points already covered by selected decisions.        
        Returns:
            cost (float): The cost of the selected decisions.
        """
        r_coverage = decision_info['coverage']
        r_cluster_coverage = decision_info['cluster_coverage']
        mistakes = r_coverage.difference(r_cluster_coverage)
        return len(mistakes) + self.alpha_val * decision_info['length']


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


    def marginal_cost(
        self,
        decision_info: dict[str, any],
        total_coverage : set[int],
        total_cluster_coverage: dict[int, set[int]]
    ) -> float:
        """
        Computes the marginal cost as the number of mistakes made by the selected decision.

        Args:
            decision_info (dict[str, any]): A dictionary containing information about the decision
                being considered.
            cluster_coverage (dict[int, set[int]]): A dictionary mapping each cluster label to the 
                set of data points already covered by selected decisions.
        Returns:
            cost (float): The cost of the selected decisions.
        """
        r_coverage = decision_info['coverage']
        r_center = decision_info['label']
        if r_coverage:  # Only process non-empty sets
            coverage_array = decision_info['coverage_array']
            cluster_cost = np.sum(self.data_to_center_distances[coverage_array, r_center])
            return cluster_cost + self.alpha_val * decision_info['length']
        return self.alpha_val * decision_info['length']


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


    def marginal_cost(
        self,
        decision_info: dict[str, any],
        total_coverage : set[int],
        total_cluster_coverage: dict[int, set[int]]
    ) -> float:
        """
        Computes the marginal cost as the number of mistakes made by the selected decision.

        Args:
            decision_info (dict[str, any]): A dictionary containing information about the decision
                being considered.
            cluster_coverage (dict[int, set[int]]): A dictionary mapping each cluster label to the 
                set of data points already covered by selected decisions.
        Returns:
            cost (float): The cost of the selected decisions.
        """
        r_coverage = decision_info['coverage']
        r_center = decision_info['label']
        if r_coverage:  # Only process non-empty sets
            coverage_array = decision_info['coverage_array']
            cluster_cost = np.sum(self.data_to_center_distances[coverage_array, r_center])
            return cluster_cost + self.alpha_val * decision_info['length']
        return self.alpha_val * decision_info['length']


####################################################################################################