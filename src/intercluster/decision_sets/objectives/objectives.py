import numpy as np
from numpy.typing import NDArray
from intercluster import Condition
from intercluster.utils import (
    assignment_to_dict, labels_to_assignment, unique_labels, satisfies_conditions
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
        X (NDArray): The data points.
        y (list[set[int]]): The labels for each data point.
        data_to_cluster_assignment (NDArray): The cluster assignment for each data point.
        rules (list[list[Condition]]): The rules for the objective.
        rule_labels (list[set[int]]): The labels for each rule.
        data_to_rules_assignment (NDArray): The rule assignment for each data point.
        rules_to_clusters_assignment (NDArray): The cluster assignment for each rule.
        value (float): The value of the objective function for the selected rules.
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

        self.data_set = False
        self.rules_set = False

        self.X = None
        self.y = None
        self.data_to_cluster_assignment = None
        self.rules = None
        self.rule_labels = None
        self.data_to_rules_assignment = None
        self.rule_to_cluster_assignment = None


    def set_data(
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
        n_labels = len(unique_labels(y))
        self.data_to_cluster_assignment = labels_to_assignment(
            y, n_labels = n_labels, ignore = {-1}
        )
        self.data_set = True


    def set_rules(
        self,
        rules : list[list[Condition]],
        rule_labels : list[set[int]],
    ):
        """
        Sets the rules for the objective.
        """
        if self.data_to_cluster_assignment is None:
            raise ValueError('Data must be set before setting rules.')
        self.rules = rules
        self.rule_labels = rule_labels
        n_labels = len(unique_labels(rule_labels))
        if n_labels != self.data_to_cluster_assignment.shape[1]:
            raise ValueError("Number of labels in rule_labels must match number of clusters "
                             "seen in y.")

        self.data_to_rules_assignment = np.zeros((self.X.shape[0], len(rules)), dtype = bool)
        for i, condition_list in enumerate(rules):
            data_points_satisfied = satisfies_conditions(self.X, condition_list)
            self.data_to_rules_assignment[data_points_satisfied, i] = True

        self.rule_to_cluster_assignment = labels_to_assignment(
            rule_labels, n_labels = n_labels, ignore = {-1}
        )

        self.rule_lengths = [len(rule) for rule in rules]
        self.rules_set = True
        self.create_rule_info_dict()


    def set_lambda(self, lambda_val : float = None):
        """
        Sets the lambda value for the objective.

        Args:
            lambda_val (float): The new lambda value.
        """
        if lambda_val is None and not (self.data_set and self.rules_set):
            raise ValueError('Data and rules must be set before setting lambda automatically.')
        elif lambda_val is None:
            lambda_vals = self.compute_lambdas()
            if len(lambda_vals) == 0:
                lambda_val = 0.0
            else:
                lambda_val = lambda_vals[0]
    
        self.lambda_val = lambda_val


    def create_rule_info_dict(self) -> dict[int, dict[str, any]]:
        """
        Creates a dictionary containing information about each rule.

        Args:
            data_to_cluster_assignment (NDArray): The cluster assignment for each data point.
            rule_to_cluster_assignment (NDArray): The cluster assignment for each rule.
            data_to_rules_assignment (NDArray): The rule assignment for each data point.
            rule_lengths (list[int]): The lengths of each rule.

        Returns:
            dict[int, dict[str, any]]: A dictionary mapping each rule index to its information.
                This includes the rule's points, coverage, length, and label.
        """
        if not (self.data_set and self.rules_set):
            raise ValueError('Data and rules must be set before creating rule info dictionary.')
        
        r,_ = self.rule_to_cluster_assignment.shape
        cluster_coverage = assignment_to_dict(self.data_to_cluster_assignment)
        rule_labels = {i: self.rule_to_cluster_assignment[i,:].nonzero()[0][0] for i in range(r)}
        rule_coverage = assignment_to_dict(self.data_to_rules_assignment)
        rule_cluster_coverage = {}
        for r, r_coverage in rule_coverage.items():
            rule_label = rule_labels[r]
            c_coverage = cluster_coverage[rule_label]
            rule_cluster_coverage[r] = r_coverage.intersection(c_coverage)

        # Storing all relevant information about each rule:
        rules_info = {
            r: {
                'coverage': rule_coverage[r],
                'cluster_coverage': rule_cluster_coverage[r],
                'length': self.rule_lengths[r],
                'label': rule_labels[r]
            } for r in range(r)
        }
        return rules_info


    def reward(
        self,
        selected_rules_info: dict[int, dict[str, any]],
    ) -> float:
        """
        Computes the reward from the selected rules.

        Args:
            selected_rules_info (dict[int, dict[str, any]]): A dictionary mapping each selected
                rule index to its information, including labels, points, coverage, and lengths.
        Returns:
            reward (float): The reward from the selected rules.
        """
        pass


    def cost(
        self,
        selected_rules_info: dict[int, dict[str, any]],
    ) -> float:
        """
        Computes the cost of the selected rules.

        Args:
            selected_rules_info (dict[int, dict[str, any]]): A dictionary mapping each selected
                rule index to its information, including labels, points, coverage, and lengths.
        Returns:
            cost (float): The cost of the selected rules.
        """
        pass


    def compute_objective(
        self,
        selected_rules_info: dict[int, dict[str, any]],
    ) -> float:
        """
        Computes the objective value for the selected rules.

        Args:
            selected_rules_info (dict[int, dict[str, any]]): A dictionary mapping each selected
                rule index to its information, including labels, points, coverage, and lengths.
        Returns:
            objective (float): The objective value for the selected rules.
        """
        g = self.reward(selected_rules_info)
        h = self.cost(selected_rules_info)
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
        n,r = self.data_to_rules_assignment.shape
        _,k = self.data_to_cluster_assignment.shape
        rule_list = list(np.arange(r))
        
        #rule_points = assignment_to_dict(self.data_to_rules_assignment)
        #cluster_points = assignment_to_dict(self.data_to_cluster_assignment)
        #rule_length_dict = {i: self.rule_lengths[i] for i in range(r)}

        ratios = []
        second_max_ratio = 0.0
        for rule in rule_list:
            r_coverage = rule_points[rule]
            r_length = rule_length_dict[rule]
            c_ratios = []
            for cluster in range(k):
                r_cluster_coverage = r_coverage.intersection(cluster_points[cluster])
                r_info = {rule: {
                        'coverage': r_coverage,
                        'cluster_coverage': r_cluster_coverage,
                        'length': r_length,
                        'label': cluster
                    }
                }
                g = self.reward(r_info)
                h = self.cost(r_info)

                if h > 0:
                    ratio = g / h
                    c_ratios.append(ratio)
                else:
                    ratio = np.inf
                    c_ratios.append(ratio)


            c_ratios_sorted = np.sort(c_ratios)
            ratios.append(c_ratios_sorted[-1])

            # Does this need to be here?
            if len(c_ratios) >= 2:
                second_largest = c_ratios_sorted[-2]
                if second_largest > second_max_ratio:
                    second_max_ratio = second_largest
                    
        ratios = [r for r in ratios if r >= second_max_ratio]
        if second_max_ratio == 0.0:
            return np.sort(ratios)
        return np.sort(ratios + [second_max_ratio])


    def marginal_reward(
        self,
        rule_info: dict[str, any],
        total_coverage : set[int],
        cluster_coverage : dict[int, set[int]]
    ) -> float:
        """
        Computes the marginal reward from selected rule.

        Args:
            rule_info (dict): A dictionary containing information about the rule being considered.
            cluster_coverage (dict[int, set[int]]): A dictionary mapping each cluster
                label to the set of data points already covered by selected rules.
        
        Returns:
            coverage (float): The coverage of the selected rules.
        """
        pass


    def marginal_cost(
        self,
        rule_info: dict[str, any],
        total_coverage : set[int],
        cluster_coverage : dict[int, set[int]]
    ) -> float:
        """
        Computes the marginal cost of the selected rule.

        Args:
            rule_info (dict): A dictionary containing information about the rule being considered.
            cluster_coverage (dict[int, set[int]]): A dictionary mapping each cluster
                label to the set of data points already covered by selected rules.
        
        Returns:
            cost (float): The cost of the selected rules.
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
            NDArray: An array of integers representing the indices of the selected rules.
        """
        if not (self.data_set and self.rules_set):
            raise ValueError('Data and rules must be set before selecting rules.')
        n,k = self.data_to_cluster_assignment.shape
        r,_ = self.rule_to_cluster_assignment.shape

        rules_info = self.create_rule_info_dict()

        total_coverage = set()
        total_cluster_coverage = {l: set() for l in range(k)}
        selected_rules = set()
        discarded_rules = set()
        for i in range(self.n_select):
            best_rule = None
            best_rule_score = 0.0
            
            # NOTE: Iterating over rules in a sorted order to ensure deterministic behavior.
            # Effectively, this means that when there are ties, the rule with the lowest index
            # will be selected. This is consistent with our preference for lexicographic ordering
            # in optimal solution sets.
            for rule in range(r):
                if (rule not in selected_rules) and (rule not in discarded_rules):
                    g = self.marginal_reward(
                        rules_info[rule],
                        total_coverage,
                        total_cluster_coverage
                    )

                    h = self.marginal_cost(
                        rules_info[rule],
                        total_coverage,
                        total_cluster_coverage
                    )

                    # Early discard since the marginal reward will only decrease from here on out, 
                    # and its score coefficient will be at most 1.
                    # Therefore if g - lambda * c <= 0, the score will never be positive, 
                    # and it can never be selected.
                    if g - self.lambda_val * h <= 0:
                        discarded_rules.add(rule)
                    
                    score = (1 - 1/self.n_select)**(self.n_select - (i + 1)) * g - self.lambda_val * h
                    
                    if score > best_rule_score:
                        best_rule = rule
                        best_rule_score = score
                        
            if best_rule_score > 0:
                selected_rules.add(best_rule)
                best_rule_label = rules_info[best_rule]['label']
                best_rule_coverage = rules_info[best_rule]['coverage']
                best_rule_cluster_coverage = rules_info[best_rule]['cluster_coverage']
                total_cluster_coverage[best_rule_label] = total_cluster_coverage[
                    best_rule_label
                ].union(
                    best_rule_cluster_coverage
                )
                total_coverage = total_coverage.union(best_rule_coverage)

        # Compute final objective value
        self.reward_value = self.reward(
            {rule: rules_info[rule] for rule in selected_rules},
        )
        self.cost_value = self.cost(
            {rule: rules_info[rule] for rule in selected_rules},
        )
        self.objective_value = self.compute_objective(
            {rule: rules_info[rule] for rule in selected_rules},
        )
        return np.array(list(selected_rules))


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
        selected_rules_info: dict[int, dict[str, any]],
    ) -> float:
        """
        Computes the reward from the selected rules.

        Args:
            selected_rules_info (dict[int, dict[str, any]]): A dictionary mapping each selected 
                rule index to its information (points, coverage, length, label).
        Returns:
            reward (float): The reward from the selected rules.
        """
        total_cluster_coverage = {}
        for rule, info in selected_rules_info.items():
            label = info['label']
            if label not in total_cluster_coverage:
                total_cluster_coverage[label] = set()
            total_cluster_coverage[label] = total_cluster_coverage[label].union(
                info['cluster_coverage']
            )
        total_weighted_coverage = 0
        for l, covered in total_cluster_coverage.items():
            total_weighted_coverage += np.sum(self.weights[list(covered)])
        return total_weighted_coverage


    def cost(
        self,
        selected_rules_info: dict[int, dict[str, any]],
    ) -> float:
        """
        Computes the cost of the selected rules.

        Args:
            selected_rules_info (dict[int, dict[str, any]]): A dictionary mapping each selected rule index
                to its information (points, coverage, length, label).
        Returns:
            cost (float): The cost of the selected rules.
        """
        total_mistakes = 0
        for rule, info in selected_rules_info.items():
            r_coverage = info['coverage']
            r_cluster_coverage = info['cluster_coverage']
            mistakes = r_coverage.difference(r_cluster_coverage)
            total_mistakes += len(mistakes)

        length_penalty = sum(
            self.alpha_val * info['length'] for rule, info in selected_rules_info.items()
        )
        return total_mistakes + length_penalty


    def marginal_reward(
        self,
        rule_info: dict[str, any],
        total_coverage : set[int],
        total_cluster_coverage : dict[int, set[int]]
    ) -> float:
        """
        Computes the marginal reward as new coverage from selected rule.

        Args:
            rule_info (dict): A dictionary containing information about the rule being considered.
            cluster_coverage (dict[int, set[int]]): A dictionary mapping each cluster
                label to the set of data points already covered by selected rules.
        
        Returns:
            coverage (float): The coverage of the selected rules.
        """
        r_cluster_coverage = rule_info['cluster_coverage']
        s_coverage = total_cluster_coverage[rule_info['label']]
        new_coverage = r_cluster_coverage.difference(s_coverage)
        new_coverage_weighted = np.sum(self.weights[list(new_coverage)])
        return new_coverage_weighted


    def marginal_cost(
        self,
        rule_info: dict[str, any],
        total_coverage : set[int],
        total_cluster_coverage: dict[int, set[int]],
    ) -> float:
        """
        Computes the marginal cost as the number of mistakes made by the selected rule.

        Args:
            rule_info (dict): A dictionary containing information about the rule being considered.
            cluster_coverage (dict[int, set[int]]): A dictionary mapping each cluster
                label to the set of data points already covered by selected rules.
        
        Returns:
            cost (float): The cost of the selected rules.
        """
        r_coverage = rule_info['coverage']
        r_cluster_coverage = rule_info['cluster_coverage']
        mistakes = r_coverage.difference(r_cluster_coverage)
        return len(mistakes) + self.alpha_val * rule_info['length']
        

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
        selected_rules_info: dict[int, dict[str, any]],
    ) -> float:
        """
        Computes the reward from the selected rules.

        Args:
            selected_rules_info (dict[int, dict[str, any]]): A dictionary mapping each selected 
                rule index to its information (points, coverage, length, label).
        Returns:
            reward (float): The reward from the selected rules.
        """
        total_coverage = set()
        for rule, info in selected_rules_info.items():
            r_coverage = info['coverage']
            total_coverage = total_coverage.union(r_coverage)

        return np.sum(self.weights[list(total_coverage)])


    def cost(
        self,
        selected_rules_info: dict[int, dict[str, any]]
    ) -> float:
        """
        Computes the cost of the selected rules.

        Args:
            selected_rules_info (dict[int, dict[str, any]]): A dictionary mapping each selected rule index
                to its information (points, coverage, length, label).
        Returns:
            cost (float): The cost of the selected rules.
        """
        total_mistakes = 0
        for rule, info in selected_rules_info.items():
            r_coverage = info['coverage']
            r_cluster_coverage = info['cluster_coverage']
            mistakes = r_coverage.difference(r_cluster_coverage)
            total_mistakes += len(mistakes)

        length_penalty = sum(
            [self.alpha_val * info['length'] for rule, info in selected_rules_info.items()]
        )
        return total_mistakes + length_penalty

    
    def marginal_reward(
        self,
        rule_info: dict[str, any],
        total_coverage : set[int],
        total_cluster_coverage : dict[int, set[int]]
    ) -> float:
        """
        Computes the marginal reward as new coverage from selected rule.

        Args:
            rule_info (dict): A dictionary containing information about the rule being considered.
            cluster_coverage (dict[int, set[int]]): A dictionary mapping each cluster
                label to the set of data points already covered by selected rules.        
        Returns:
            coverage (float): The coverage of the selected rules.
        """
        r_coverage = rule_info['coverage']
        new_points = r_coverage.difference(total_coverage)
        new_points_weighted = np.sum(self.weights[list(new_points)])
        return new_points_weighted


    def marginal_cost(
        self,
        rule_info: dict[str, any],
        total_coverage : set[int],
        total_cluster_coverage: dict[int, set[int]]
    ) -> float:
        """
        Computes the marginal cost as the number of mistakes made by the selected rule.

        Args:
            rule_info (dict): A dictionary containing information about the rule being considered.
            cluster_coverage (dict[int, set[int]]): A dictionary mapping each cluster
                label to the set of data points already covered by selected rules.        
        Returns:
            cost (float): The cost of the selected rules.
        """
        r_coverage = rule_info['coverage']
        r_cluster_coverage = rule_info['cluster_coverage']
        mistakes = r_coverage.difference(r_cluster_coverage)
        return len(mistakes) + self.alpha_val * rule_info['length']
        

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


    def set_data(
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
        n_labels = len(unique_labels(y))
        self.data_to_cluster_assignment = labels_to_assignment(
            y, n_labels = n_labels, ignore = {-1}
        )
        self.data_set = True

        # Compute n x k distance matrix between data points and cluster centers.
        self.data_to_center_distances = np.zeros((self.X.shape[0], self.cluster_centers.shape[0]))
        for i in range(self.cluster_centers.shape[0]):
            if self.cluster_cost_method == "kmeans":
                self.data_to_center_distances[:, i] = np.sum((self.X - self.cluster_centers[i])**2, axis=1)
            else:  # self.cluster_cost_cluster_cost_method == "kmedians"
                self.data_to_center_distances[:, i] = np.sum(np.abs(self.X - self.cluster_centers[i]), axis=1)


    def reward(
        self,
        selected_rules_info: dict[int, dict[str, any]],
    ) -> float:
        """
        Computes the reward from the selected rules.

        Args:
            selected_rules_info (dict[int, dict[str, any]]): A dictionary mapping each selected 
                rule index to its information (points, coverage, length, label).
        Returns:
            reward (float): The reward from the selected rules.
        """
        total_cluster_coverage = {}
        for rule, info in selected_rules_info.items():
            label = info['label']
            if label not in total_cluster_coverage:
                total_cluster_coverage[label] = set()
            total_cluster_coverage[label] = total_cluster_coverage[label].union(
                info['cluster_coverage']
            )
        total_weighted_coverage = 0
        for l, covered in total_cluster_coverage.items():
            total_weighted_coverage += np.sum(self.weights[list(covered)])
        return total_weighted_coverage


    def cost(
        self,
        selected_rules_info: dict[int, dict[str, any]]
    ) -> float:
        """
        Computes the cost of the selected rules.

        Args:
            selected_rules_info (dict[int, dict[str, any]]): A dictionary mapping each selected rule index
                to its information (points, coverage, length, label).
        Returns:
            cost (float): The cost of the selected rules.
        """
        total_cost = 0.0
        for rule, info in selected_rules_info.items():
            r_coverage = info['coverage']
            r_center = info['label']
            cluster_cost = np.sum(self.data_to_center_distances[list(r_coverage), r_center])
            total_cost += cluster_cost

        length_penalty = sum(
            self.alpha_val * info['length'] for rule, info in selected_rules_info.items()
        )
        return total_cost + length_penalty


    def marginal_reward(
        self,
        rule_info: dict[str, any],
        total_coverage : set[int],
        total_cluster_coverage : dict[int, set[int]]
    ) -> float:
        """
        Computes the marginal reward as new coverage from selected rule.

        Args:
            rule_info (dict[str, any]): A dictionary containing information about the rule being considered.
            cluster_coverage (dict[int, set[int]]): A dictionary mapping each cluster label to the set of data points
                already covered by selected rules.

        Returns:
            coverage (float): The coverage of the selected rules.
        """
        r_cluster_coverage = rule_info['cluster_coverage']
        s_coverage = total_cluster_coverage[rule_info['label']]
        new_coverage = r_cluster_coverage.difference(s_coverage)
        new_coverage_weighted = np.sum(self.weights[list(new_coverage)])
        return new_coverage_weighted


    def marginal_cost(
        self,
        rule_info: dict[str, any],
        total_coverage : set[int],
        total_cluster_coverage: dict[int, set[int]]
    ) -> float:
        """
        Computes the marginal cost as the number of mistakes made by the selected rule.

        Args:
            rule_info (dict[str, any]): A dictionary containing information about the rule being considered.
            cluster_coverage (dict[int, set[int]]): A dictionary mapping each cluster label to the set of data points
                already covered by selected rules.
        Returns:
            cost (float): The cost of the selected rules.
        """
        r_coverage = rule_info['coverage']
        r_center = rule_info['label']
        cluster_cost = np.sum(self.data_to_center_distances[list(r_coverage), r_center])
        return cluster_cost + self.alpha_val * rule_info['length']
    

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


    def set_data(
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
        n_labels = len(unique_labels(y))
        self.data_to_cluster_assignment = labels_to_assignment(
            y, n_labels = n_labels, ignore = {-1}
        )
        self.data_set = True

        # Compute n x k distance matrix between data points and cluster centers.
        self.data_to_center_distances = np.zeros((self.X.shape[0], self.cluster_centers.shape[0]))
        for i in range(self.cluster_centers.shape[0]):
            if self.cluster_cost_method == "kmeans":
                self.data_to_center_distances[:, i] = np.sum((self.X - self.cluster_centers[i])**2, axis=1)
            else:  # self.cluster_cost_cluster_cost_method == "kmedians"
                self.data_to_center_distances[:, i] = np.sum(np.abs(self.X - self.cluster_centers[i]), axis=1)


    def reward(
        self,
        selected_rules_info: dict[int, dict[str, any]],
    ) -> float:
        """
        Computes the reward from the selected rules.

        Args:
            selected_rules_info (dict[int, dict[str, any]]): A dictionary mapping each selected 
                rule index to its information (points, coverage, length, label).
        Returns:
            reward (float): The reward from the selected rules.
        """
        total_coverage = set()
        for rule, info in selected_rules_info.items():
            r_coverage = info['coverage']
            total_coverage = total_coverage.union(r_coverage)
        return np.sum(self.weights[list(total_coverage)])


    def cost(
        self,
        selected_rules_info: dict[int, dict[str, any]]
    ) -> float:
        """
        Computes the cost of the selected rules.

        Args:
            selected_rules_info (dict[int, dict[str, any]]): A dictionary mapping each selected rule index
                to its information (points, coverage, length, label).
        Returns:
            cost (float): The cost of the selected rules.
        """
        total_cost = 0.0
        for rule, info in selected_rules_info.items():
            r_coverage = info['coverage']
            r_center = info['label']
            cluster_cost = np.sum(self.data_to_center_distances[list(r_coverage), r_center])
            total_cost += cluster_cost

        length_penalty = sum(
            self.alpha_val * info['length'] for rule, info in selected_rules_info.items()
        )
        return total_cost + length_penalty


    def marginal_reward(
        self,
        rule_info: dict[str, any],
        total_coverage : set[int],
        total_cluster_coverage : dict[int, set[int]]
    ) -> float:
        """
        Computes the marginal reward as new coverage from selected rule.

        Args:
            rule_info (dict[str, any]): A dictionary containing information about the rule being considered.
            cluster_coverage (dict[int, set[int]]): A dictionary mapping each cluster label to the set of data points
                already covered by selected rules.
        Returns:
            coverage (float): The coverage of the selected rules.
        """
        r_coverage = rule_info['coverage']
        new_points = r_coverage.difference(total_coverage)
        new_points_weighted = np.sum(self.weights[list(new_points)])
        return new_points_weighted


    def marginal_cost(
        self,
        rule_info: dict[str, any],
        total_coverage : set[int],
        total_cluster_coverage: dict[int, set[int]]
    ) -> float:
        """
        Computes the marginal cost as the number of mistakes made by the selected rule.

        Args:
            rule_info (dict[str, any]): A dictionary containing information about the rule being considered.
            cluster_coverage (dict[int, set[int]]): A dictionary mapping each cluster label to the set of data points
                already covered by selected rules.
        Returns:
            cost (float): The cost of the selected rules.
        """
        r_coverage = rule_info['coverage']
        r_center = rule_info['label']
        cluster_cost = np.sum(self.data_to_center_distances[list(r_coverage), r_center])
        return cluster_cost + self.alpha_val * rule_info['length']


####################################################################################################