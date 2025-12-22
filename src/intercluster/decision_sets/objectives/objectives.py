import numpy as np
from numpy.typing import NDArray
from intercluster.utils import (
    assignment_to_dict,
)

####################################################################################################


class Objective:
    """
    Base class for a selector, which is used to select rules based on a given objective.

    Args:
        n_rules (int): The *maximum* number of rules to select.
        lambda_val (float): A hyperparameter that controls tradeoff between reward and cost.
            Defaults to 1.0.
        alpha_val (float): A hyperparameter for tuning the size of the selected rules.
            Larger values penalize longer rules more heavily. Defaults to 1.0.
        cluster_centers (NDArray): (k x d) array where each row i is the given 
            representative for cluster i.

    Attrs:
        name (str): Name of the objective.
        value (float): The value of the objective function for the selected rules.
    """
    def __init__(
        self,
        n_rules : int,
        lambda_val : float = 1.0,
        alpha_val : float = 1.0
    ):
        self.n_rules = n_rules
        self.lambda_val = lambda_val
        self.alpha_val = alpha_val


    def set_lambda(self, lambda_val : float):
        """
        Sets the lambda value for the objective.

        Args:
            lambda_val (float): The new lambda value.
        """
        self.lambda_val = lambda_val


    def create_rule_info_dict(
        self,
        data_to_cluster_assignment : NDArray,
        rule_to_cluster_assignment : NDArray,
        data_to_rules_assignment : NDArray,
        rule_lengths : list[int]
    ) -> dict[int, dict[str, any]]:
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
        r,_ = rule_to_cluster_assignment.shape
        cluster_points = assignment_to_dict(data_to_cluster_assignment)
        rule_labels = {i: rule_to_cluster_assignment[i,:].nonzero()[0][0] for i in range(r)}
        rule_points = assignment_to_dict(data_to_rules_assignment)
        rule_cluster_coverage = {}
        for rule, points in rule_points.items():
            rule_label = rule_labels[rule]
            c_points = cluster_points[rule_label]
            rule_cluster_coverage[rule] = points.intersection(c_points)

        # Storing all relevant information about each rule:
        rules_info = {
            r: {
                'points': rule_points[r],
                'coverage': rule_cluster_coverage[r],
                'length': rule_lengths[r],
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
    

    def compute_lambdas(
        self,
        data : NDArray,
        data_to_cluster_assignment : NDArray,
        data_to_rules_assignment : NDArray,
        rule_lengths : list[int]
    ) -> NDArray:
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
        n,d = data.shape
        n2,k = data_to_cluster_assignment.shape
        assert n == n2, "Data and Data to Cluster assignment arrays do not match in shape along axis 0."
        assert np.all(np.sum(data_to_cluster_assignment, axis = 1) <= 1), ("Each data point must be "
                                                                        "assigned to at most one cluster.")

        n3,r = data_to_rules_assignment.shape
        assert n == n3, "Data and Data to Rule assignment arrays do not match in shape along axis 0."
        assert len(rule_lengths) == r, "Rule lengths must match number of rules."

        #if self.cluster_centers is not None:
        #    assert self.cluster_centers.shape[0] == k, "Number of cluster centers must match number of clusters."
        #    assert self.cluster_centers.shape[1] == d, "Cluster center dimension must match data dimension."

        rule_list = list(np.arange(r))
        
        rule_points = assignment_to_dict(data_to_rules_assignment)
        cluster_points = assignment_to_dict(data_to_cluster_assignment)
        rule_length_dict = {i: rule_lengths[i] for i in range(r)}

        ratios = []
        second_max_ratio = 0.0
        for rule in rule_list:
            r_points = rule_points[rule]
            r_length = rule_length_dict[rule]
            c_ratios = []
            for cluster in range(k):
                r_coverage = rule_points[rule].intersection(cluster_points[cluster])
                r_info = {rule: {
                        'points': r_points,
                        'coverage': r_coverage,
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
        data : NDArray,
        data_to_cluster_assignment : NDArray,
        rule_to_cluster_assignment : NDArray,
        data_to_rules_assignment : NDArray,
        rule_lengths : list[int]
    ) -> NDArray[np.int64]:
        """
        Selects a subset rules using a distorted greedy algorithm. For more information 
        on the algorithm, see the following paper:
        "Submodular Maximization Beyond Non-Negativity: Guarantees, Fast Algorithms, and Applications"
        by Harshaw el al., ICML 2019.
        
        Args:
            data (NDArray): (n x d) Data array.
            data_to_cluster_assignment (np.ndarray): Size (n x k) boolean array where entry (i,j) is 
                `True` if point i is assigned to cluster j and `False` otherwise. Data points may be 
                assigned to multiple clusters. 
            rule_to_cluster_assignment (np.ndarray): Size (r x k) boolean array where entry (i,j) is 
                `True` if rule i is assigned to cluster j and `False` otherwise. Each rule must 
                be assigned to a single cluster.
            data_to_rules_assignment (NDArray): A boolean matrix where entry (i,j) is `True` if 
                data point i is assigned to rule j and `False` otherwise.
            rule_lengths (list[int]): A list of lengths for each rule.
                
        Returns:
            NDArray: An array of integers representing the indices of the selected rules.
        """
        n,d = data.shape
        n2,k = data_to_cluster_assignment.shape
        assert n == n2, "Data and Data to Cluster assignment arrays do not match in shape along axis 0."
        assert np.all(np.sum(data_to_cluster_assignment, axis = 1) <= 1), ("Each data point must be "
                                                                        "assigned to at most one cluster.")
        r,k2 = rule_to_cluster_assignment.shape
        assert k == k2, "Data and Rule assignment arrays do not match in shape along axis 1."
        assert np.all(np.sum(rule_to_cluster_assignment, axis = 1) == 1), ("Rules must be assigned "
                                                                        "to exactly one cluster.")
        n3,r2 = data_to_rules_assignment.shape
        assert n == n3, "Data and Data to Rule assignment arrays do not match in shape along axis 0."
        assert r == r2, "Number of rules in Rule assignment must match number of rules in Data to Rule assignment."
        
        #if self.cluster_centers is not None:
        #    assert self.cluster_centers.shape[0] == k, "Number of cluster centers must match number of clusters."
        #    assert self.cluster_centers.shape[1] == d, "Cluster center dimension must match data dimension."


        rules_info = self.create_rule_info_dict(
            data_to_cluster_assignment,
            rule_to_cluster_assignment,
            data_to_rules_assignment,
            rule_lengths
        )

        total_coverage = set()
        cluster_coverage = {l: set() for l in range(k)}
        selected_rules = set()
        discarded_rules = set()
        for i in range(self.n_rules):
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
                        cluster_coverage
                    )

                    h = self.marginal_cost(
                        rules_info[rule],
                        total_coverage,
                        cluster_coverage
                    )

                    # Early discard since the marginal reward will only decrease from here on out, 
                    # and its score coefficient will be at most 1.
                    # Therefore if g - lambda * c <= 0, the score will never be positive, 
                    # and it can never be selected.
                    if g - self.lambda_val * h <= 0:
                        discarded_rules.add(rule)
                    
                    score = (1 - 1/self.n_rules)**(self.n_rules - (i + 1)) * g - self.lambda_val * h
                    
                    if score > best_rule_score:
                        best_rule = rule
                        best_rule_score = score
                        
            if best_rule_score > 0:
                selected_rules.add(best_rule)
                best_rule_label = rules_info[best_rule]['label']
                best_rule_points = rules_info[best_rule]['points']
                best_rule_coverage = rules_info[best_rule]['coverage']
                cluster_coverage[best_rule_label] = cluster_coverage[
                    best_rule_label
                ].union(
                    best_rule_coverage
                )
                total_coverage = total_coverage.union(best_rule_points)

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
        n_rules (int): The *maximum* number of rules to select.
        lambda_val (float): A hyperparameter that controls tradeoff between coverage and overlap.
            Defaults to 1.0.
    """
    def __init__(self, n_rules : int, lambda_val : float = 1.0, alpha_val : float = 1.0):
        """
        Args:
            n_rules (int): The *maximum* number of rules to select.
            lambda_val (float): A hyperparameter that controls tradeoff between coverage and overlap.
            alpha_val (float): A hyperparameter for tuning the size of the selected rules.
                Larger values penalize longer rules more heavily. Defaults to 1.0.
        """
        super().__init__(
            n_rules = n_rules,
            lambda_val = lambda_val,
            alpha_val = alpha_val
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
        per_cluster_coverage = {}
        for rule, info in selected_rules_info.items():
            label = info['label']
            if label not in per_cluster_coverage:
                per_cluster_coverage[label] = set()
            per_cluster_coverage[label] = per_cluster_coverage[label].union(
                info['coverage']
            )
        total_coverage = 0
        for l, covered in per_cluster_coverage.items():
            total_coverage += len(covered)
        return total_coverage


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
            r_points = info['points']
            r_coverage = info['coverage']
            mistakes = r_points.difference(r_coverage)
            total_mistakes += len(mistakes)

        length_penalty = sum(
            self.alpha_val * info['length'] for rule, info in selected_rules_info.items()
        )
        return total_mistakes + length_penalty


    def marginal_reward(
        self,
        rule_info: dict[str, any],
        total_coverage : set[int],
        cluster_coverage : dict[int, set[int]]
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
        s_coverage = cluster_coverage[rule_info['label']]
        new_coverage = r_coverage.difference(s_coverage)
        return len(new_coverage)


    def marginal_cost(
        self,
        rule_info: dict[str, any],
        total_coverage : set[int],
        cluster_coverage: dict[int, set[int]],
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
        r_points = rule_info['points']
        r_coverage = rule_info['coverage']
        mistakes = r_points.difference(r_coverage)
        return len(mistakes) + self.alpha_val * rule_info['length']
        

####################################################################################################


class TotalCoverageMistakeObjective(Objective):
    """
    Objective that selects rules based on a coverage and mistake objective. The difference 
    with CoverageMistakeObjective is that the coverage is computed across all clusters,
    rather than within each cluster.

    Args:
        n_rules (int): The *maximum* number of rules to select.
        lambda_val (float): A hyperparameter that controls tradeoff between coverage and overlap.
            Defaults to 1.0.
    """
    def __init__(self, n_rules : int, lambda_val : float = 1.0, alpha_val : float = 1.0):
        """
        Args:
            n_rules (int): The *maximum* number of rules to select.
            lambda_val (float): A hyperparameter that controls tradeoff between coverage and overlap.
            alpha_val (float): A hyperparameter for tuning the size of the selected rules.
                Larger values penalize longer rules more heavily. Defaults to 1.0.
        """
        super().__init__(
            n_rules = n_rules,
            lambda_val = lambda_val,
            alpha_val = alpha_val
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
            r_coverage = info['points']
            total_coverage = total_coverage.union(r_coverage)
        return len(total_coverage)


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
            r_points = info['points']
            r_coverage = info['coverage']
            mistakes = r_points.difference(r_coverage)
            total_mistakes += len(mistakes)

        length_penalty = sum(
            [self.alpha_val * info['length'] for rule, info in selected_rules_info.items()]
        )
        return total_mistakes + length_penalty

    
    def marginal_reward(
        self,
        rule_info: dict[str, any],
        total_coverage : set[int],
        cluster_coverage : dict[int, set[int]]
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
        r_points = rule_info['points']
        new_points = r_points.difference(total_coverage)
        return len(new_points)


    def marginal_cost(
        self,
        rule_info: dict[str, any],
        total_coverage : set[int],
        cluster_coverage: dict[int, set[int]]
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
        r_points = rule_info['points']
        r_coverage = rule_info['coverage']
        mistakes = r_points.difference(r_coverage)
        return len(mistakes) + self.alpha_val * rule_info['length']
        

####################################################################################################


class CoverageCostObjective(Objective):
    """
    Objective that selects rules based on a coverage and cluster cost objective.

    Args:
        data (NDArray): (n x d) array.
        cluster_centers (NDArray): (k x d) array where each row i is the given 
                representative for cluster i.
        n_rules (int): The *maximum* number of rules to select.
        lambda_val (float): A hyperparameter that controls tradeoff between coverage and overlap.
            Defaults to 1.0.
        alpha_val (float): A hyperparameter for tuning the size of the selected rules.
            Larger values penalize longer rules more heavily. Defaults to 1.0.
        method (str): The method used to compute cluster costs. 
            Currently only "kmeans" or "kmedians" are supported.
    """
    def __init__(
            self,
            data : NDArray,
            cluster_centers : NDArray,
            n_rules : int,
            lambda_val : float = 1.0,
            alpha_val : float = 1.0,
            method : str = "kmeans"
        ):
        super().__init__(
            n_rules = n_rules,
            lambda_val = lambda_val,
            alpha_val = alpha_val
        )

        self.data = data
        self.cluster_centers = cluster_centers
        if method not in ["kmeans", "kmedians"]:
            raise ValueError(f"Method {method} not supported. Supported methods are 'kmeans' and 'kmedians'.")
        self.method = method


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
        per_cluster_coverage = {}
        for rule, info in selected_rules_info.items():
            label = info['label']
            if label not in per_cluster_coverage:
                per_cluster_coverage[label] = set()
            per_cluster_coverage[label] = per_cluster_coverage[label].union(
                info['coverage']
            )
        total_coverage = 0
        for l, covered in per_cluster_coverage.items():
            total_coverage += len(covered)
        return total_coverage


    def cost(
        self,
        selected_rules_info : dict[int, dict[str, any]],
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
            r_points = info['points']
            r_center = self.cluster_centers[info['label']]
            if self.method == "kmeans":
                cluster_cost = np.sum((self.data[list(r_points)] - r_center)**2)
            else:  # self.method == "kmedians"
                cluster_cost = np.sum(np.abs(self.data[list(r_points)] - r_center))
            total_cost += cluster_cost

        length_penalty = sum(
            self.alpha_val * info['length'] for rule, info in selected_rules_info.items()
        )
        return total_cost + length_penalty


    def marginal_reward(
        self,
        rule_info: dict[str, any],
        total_coverage : set[int],
        cluster_coverage : dict[int, set[int]]
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
        s_coverage = cluster_coverage[rule_info['label']]
        new_coverage = r_coverage.difference(s_coverage)
        return len(new_coverage)


    def marginal_cost(
        self,
        rule_info: dict[str, any],
        total_coverage : set[int],
        cluster_coverage: dict[int, set[int]]
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
        r_points = rule_info['points']
        r_center = self.cluster_centers[rule_info['label']]
        if self.method == "kmeans":
            cluster_cost = np.sum((self.data[list(r_points)] - r_center)**2)
        else:  # self.method == "kmedians"
            cluster_cost = np.sum(np.abs(self.data[list(r_points)] - r_center))
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
        n_rules (int): The *maximum* number of rules to select.
        lambda_val (float): A hyperparameter that controls tradeoff between coverage and overlap.
            Defaults to 1.0.
        alpha_val (float): A hyperparameter that controls the length penalty. Defaults to 1.0.
        method (str): The method used to compute cluster costs. 
            Currently only "kmeans" or "kmedians" are supported.
    """
    def __init__(
            self,
            data : NDArray,
            cluster_centers : NDArray,
            n_rules : int,
            lambda_val : float = 1.0,
            alpha_val : float = 1.0,
            method : str = "kmeans"
        ):
        super().__init__(
            n_rules = n_rules,
            lambda_val = lambda_val,
            alpha_val = alpha_val
        )

        self.data = data
        self.cluster_centers = cluster_centers
        if method not in ["kmeans", "kmedians"]:
            raise ValueError(f"Method {method} not supported. Supported methods are 'kmeans' and 'kmedians'.")
        self.method = method


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
            r_coverage = info['points']
            total_coverage = total_coverage.union(r_coverage)
        return len(total_coverage)


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
            r_points = info['points']
            r_center = self.cluster_centers[info['label']]
            if self.method == "kmeans":
                cluster_cost = np.sum((self.data[list(r_points)] - r_center)**2)
            else:  # self.method == "kmedians"
                cluster_cost = np.sum(np.abs(self.data[list(r_points)] - r_center))
            total_cost += cluster_cost

        length_penalty = sum(
            self.alpha_val * info['length'] for rule, info in selected_rules_info.items()
        )
        return total_cost + length_penalty


    def marginal_reward(
        self,
        rule_info: dict[str, any],
        total_coverage : set[int],
        cluster_coverage : dict[int, set[int]]
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
        r_points = rule_info['points']
        new_points = r_points.difference(total_coverage)
        return len(new_points)


    def marginal_cost(
        self,
        rule_info: dict[str, any],
        total_coverage : set[int],
        cluster_coverage: dict[int, set[int]]
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
        r_points = rule_info['points']
        r_center = self.cluster_centers[rule_info['label']]
        if self.method == "kmeans":
            cluster_cost = np.sum((self.data[list(r_points)] - r_center)**2)
        else:  # self.method == "kmedians"
            cluster_cost = np.sum(np.abs(self.data[list(r_points)] - r_center))
        return cluster_cost + self.alpha_val * rule_info['length']


####################################################################################################


class TotalCoverageRuleCost(Objective):
    """
    Objective that selects rules based on a coverage and rule cost objective.

    Args:
        data (NDArray): (n x d) data array.
        n_rules (int): The *maximum* number of rules to select.
        lambda_val (float): A hyperparameter that controls tradeoff between coverage and overlap.
            Defaults to 1.0.
        alpha_val (float): A hyperparameter that controls the length penalty. Defaults to 1.0.
        method (str): The method used to compute cluster costs. 
            Currently only "kmeans" or "kmedians" are supported.
    """
    def __init__(
            self,
            data : NDArray,
            n_rules : int,
            lambda_val : float = 1.0,
            alpha_val : float = 1.0,
            method : str = "kmeans"
        ):
        super().__init__(
            n_rules = n_rules,
            lambda_val = lambda_val,
            alpha_val = alpha_val
        )

        self.data = data
        if method not in ["kmeans", "kmedians"]:
            raise ValueError(f"Method {method} not supported. Supported methods are 'kmeans' and 'kmedians'.")
        self.method = method


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
            r_coverage = info['points']
            total_coverage = total_coverage.union(r_coverage)
        return len(total_coverage)


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
            r_points = info['points']
            if self.method == "kmeans":
                r_center = np.mean(self.data[list(r_points)], axis=0)
                cluster_cost = np.sum((self.data[list(r_points)] - r_center)**2)
            else:  # self.method == "kmedians"
                r_center = np.median(self.data[list(r_points)], axis=0)
                cluster_cost = np.sum(np.abs(self.data[list(r_points)] - r_center))
            total_cost += cluster_cost

        length_penalty = sum(
            self.alpha_val * info['length'] for rule, info in selected_rules_info.items()
        )
        return total_cost + length_penalty


    def marginal_reward(
        self,
        rule_info: dict[str, any],
        total_coverage : set[int],
        cluster_coverage : dict[int, set[int]]
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
        r_points = rule_info['points']
        new_points = r_points.difference(total_coverage)
        return len(new_points)


    def marginal_cost(
        self,
        rule_info: dict[str, any],
        total_coverage : set[int],
        cluster_coverage: dict[int, set[int]]
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
        r_points = rule_info['points']
        if self.method == "kmeans":
            r_center = np.mean(self.data[list(r_points)], axis=0)
            cluster_cost = np.sum((self.data[list(r_points)] - r_center)**2)
        else:  # self.method == "kmedians"
            r_center = np.median(self.data[list(r_points)], axis=0)
            cluster_cost = np.sum(np.abs(self.data[list(r_points)] - r_center))
        return cluster_cost + self.alpha_val * rule_info['length']


####################################################################################################