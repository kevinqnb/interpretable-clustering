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
        lambda_val (float): A hyperparameter that controls tradeoff between gain and cost.
            Defaults to 1.0.
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
        cluster_centers : NDArray = None,
    ):
        self.n_rules = n_rules
        self.lambda_val = lambda_val
        self.cluster_centers = cluster_centers


    def set_data(self, data : NDArray):
        """
        Sets the data for the objective.

        Args:
            data (NDArray): (n x d) Data array.
        """
        self.data = data


    def set_lambda(self, lambda_val : float):
        """
        Sets the lambda value for the objective.

        Args:
            lambda_val (float): The new lambda value.
        """
        self.lambda_val = lambda_val


    def gain(
        self,
        selected_rule_labels : dict[int, set[int]],
        selected_rule_points : dict[int, set[int]],
        selected_rule_coverage: dict[int, set[int]]
    ) -> float:
        """
        Computes the gain from the selected rules.

        Args:
            selected_rule_labels (dict[int, set[int]]): A dictionary mapping each selected rule index
            selected_rule_points (dict[int, set[int]]): A dictionary mapping each selected rule index 
                to the full set of data points it covers.
            selected_rule_coverage (dict[int, set[int]]): A dictionary mapping each selected rule index
                to the set of data points it covers within its assigned cluster.
        Returns:
            gain (float): The gain from the selected rules.
        """
        pass


    def cost(
        self,
        selected_rule_labels : dict[int, set[int]],
        selected_rule_points : dict[int, set[int]],
        selected_rule_coverage: dict[int, set[int]]
    ) -> float:
        """
        Computes the cost of the selected rules.

        Args:
            selected_rule_labels (dict[int, set[int]]): A dictionary mapping each selected rule index
            selected_rule_points (dict[int, set[int]]): A dictionary mapping each selected rule index 
                to the full set of data points it covers.
            selected_rule_coverage (dict[int, set[int]]): A dictionary mapping each selected rule index
                to the set of data points it covers within its assigned cluster.
        Returns:
            cost (float): The cost of the selected rules.
        """
        pass


    def compute_objective(
        self,
        selected_rule_labels : dict[int, set[int]],
        selected_rule_points : dict[int, set[int]],
        selected_rule_coverage: dict[int, set[int]]
    ) -> float:
        """
        Computes the objective value for the selected rules.

        Args:
            selected_rule_labels (dict[int, set[int]]): A dictionary mapping each selected rule index
            selected_rule_points (dict[int, set[int]]): A dictionary mapping each selected rule index
                to the full set of data points it covers.
            selected_rule_coverage (dict[int, set[int]]): A dictionary mapping each selected rule index
                to the set of data points it covers within its assigned cluster.
        Returns:
            objective (float): The objective value for the selected rules.
        """
        g = self.gain(selected_rule_labels, selected_rule_points, selected_rule_coverage)
        h = self.cost(selected_rule_labels, selected_rule_points, selected_rule_coverage)
        return g - self.lambda_val * h
    

    def compute_lambdas(
        self,
        data : NDArray,
        data_to_cluster_assignment : NDArray,
        data_to_rules_assignment : NDArray
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
                
        Returns:
            lambda_vals (NDArray): A sorted array of lambda values, starting from the minimum 
                most value for which the approximation guarantee holds, and increasing
                until reaching the maximum coverage/cost ratio seen
                for any (rule, cluster) assignment pair. 
        """
        n,d = data.shape
        self.data = data
        n2,k = data_to_cluster_assignment.shape
        assert n == n2, "Data and Data to Cluster assignment arrays do not match in shape along axis 0."
        assert np.all(np.sum(data_to_cluster_assignment, axis = 1) <= 1), ("Each data point must be "
                                                                        "assigned to at most one cluster.")

        n3,r = data_to_rules_assignment.shape
        assert n == n3, "Data and Data to Rule assignment arrays do not match in shape along axis 0."
        
        if self.cluster_centers is not None:
            assert self.cluster_centers.shape[0] == k, "Number of cluster centers must match number of clusters."
            assert self.cluster_centers.shape[1] == d, "Cluster center dimension must match data dimension."

        rule_list = list(np.arange(r))
        
        rule_points = assignment_to_dict(data_to_rules_assignment)
        cluster_points = assignment_to_dict(data_to_cluster_assignment)

        largest = []
        second_max_ratio = 0.0
        for rule in rule_list:
            r_points = {rule: rule_points[rule]}
            c_ratios = []
            for cluster in range(k):
                r_labels = {rule: cluster}
                r_coverage = {rule: rule_points[rule].intersection(cluster_points[cluster])}
                g = self.gain(
                    r_labels,
                    r_points,
                    r_coverage
                )

                h = self.cost(
                    r_labels,
                    r_points,
                    r_coverage
                )

                if h > 0:
                    ratio = g / h
                    c_ratios.append(ratio)
                else:
                    ratio = np.inf
                    c_ratios.append(ratio)

            if len(c_ratios) >= 2:
                c_ratios_sorted = np.sort(c_ratios)
                largest.append(c_ratios_sorted[-1])
                second_largest = c_ratios_sorted[-2]
                if second_largest > second_max_ratio:
                    second_max_ratio = second_largest

        return np.sort(largest + [second_max_ratio])
    
    
    def marginal_gain(
        self,
        rule : int,
        rule_label : int,
        rule_points : dict[int, set[int]],
        rule_cluster_coverage : dict[int, set[int]],
        selected_cluster_coverage : dict[int, set[int]]
    ) -> float:
        """
        Computes the marginal gain from selected rule.

        Args:
            rule (int): The index of the rule being considered.
            rule_label (int): The cluster label of the rule being considered.
            rule_points (dict[int, set[int]]): A dictionary mapping each rule index to the full 
                set of data points it covers.
            rule_cluster_coverage (dict[int, set[int]]): A dictionary mapping each rule index 
                to the set of data points it covers within its assigned cluster.
            selected_cluster_coverage (dict[int, set[int]]): A dictionary mapping each cluster
                label to the set of data points already covered by selected rules.
        
        Returns:
            coverage (float): The coverage of the selected rules.
        """
        pass


    def marginal_cost(
        self,
        rule : int,
        rule_label : int,
        rule_points : dict[int, set[int]],
        rule_cluster_coverage : dict[int, set[int]],
        selected_cluster_coverage : dict[int, set[int]]
    ) -> float:
        """
        Computes the marginal cost of the selected rule.

        Args:
            rule (int): The index of the rule being considered.
            rule_label (int): The cluster label of the rule being considered.
            rule_points (dict[int, set[int]]): A dictionary mapping each rule index to the full 
                set of data points it covers.
            rule_cluster_coverage (dict[int, set[int]]): A dictionary mapping each rule index 
                to the set of data points it covers within its assigned cluster.
            selected_cluster_coverage (dict[int, set[int]]): A dictionary mapping each cluster
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
        data_to_rules_assignment : NDArray
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
                
        Returns:
            NDArray: An array of integers representing the indices of the selected rules.
        """
        n,d = data.shape
        self.data = data
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
        
        if self.cluster_centers is not None:
            assert self.cluster_centers.shape[0] == k, "Number of cluster centers must match number of clusters."
            assert self.cluster_centers.shape[1] == d, "Cluster center dimension must match data dimension."

        rule_list = list(np.arange(r))
        #rule_labels = np.array(
        #    [np.where(rule_to_cluster_assignment[i,:])[0][0] for i in range(r)]
        #)
        rule_labels = {i: rule_to_cluster_assignment[i,:].nonzero()[0][0] for i in range(r)}

        cluster_points = assignment_to_dict(data_to_cluster_assignment)
        selected_cluster_coverage = {l: set() for l in range(k)}
        
        rule_points = assignment_to_dict(data_to_rules_assignment)
        rule_cluster_coverage = {}

        for rule, points in rule_points.items():
            rule_label = rule_labels[rule]
            c_points = cluster_points[rule_label]
            rule_cluster_coverage[rule] = points.intersection(c_points)
            
        
        selected_rules = set()
        discarded_rules = set()
        for i in range(self.n_rules):
            best_rule = None
            best_rule_label = None
            best_rule_score = -np.inf
            
            # NOTE: Iterating over rules in a sorted order to ensure deterministic behavior.
            # Effectively, this means that when there are ties, the rule with the lowest index
            # will be selected. This is consistent with our preference for lexicographic ordering
            # in optimal solution sets.
            for rule in rule_list:
                if (rule not in selected_rules) and (rule not in discarded_rules):
                    rule_label = rule_labels[rule]

                    g = self.marginal_gain(
                        rule,
                        rule_label,
                        rule_points,
                        rule_cluster_coverage,
                        selected_cluster_coverage
                    )

                    h = self.marginal_cost(
                        rule,
                        rule_label,
                        rule_points,
                        rule_cluster_coverage,
                        selected_cluster_coverage
                    )

                    # Early discard since the marginal gain will only decrease from here on out, 
                    # and its score coefficient will be at most 1.
                    # Therefore if g - lambda * c <= 0, the score will never be positive, 
                    # and it can never be selected.
                    if g - self.lambda_val * h <= 0:
                        discarded_rules.add(rule)
                    
                    score = (1 - 1/self.n_rules)**(self.n_rules - (i + 1)) * g - self.lambda_val * h
                    
                    if score > best_rule_score:
                        best_rule = rule
                        best_rule_label = rule_label
                        best_rule_score = score
                        
            if best_rule_score > 0:
                selected_rules.add(best_rule)
                best_rule_coverage = rule_cluster_coverage[best_rule]
                selected_cluster_coverage[best_rule_label] = selected_cluster_coverage[
                    best_rule_label
                ].union(
                    best_rule_coverage
                )

        
        # Compute final objective value
        selected_rule_labels = {rule: rule_labels[rule] for rule in selected_rules}
        selected_rule_points = {rule: rule_points[rule] for rule in selected_rules}
        selected_rule_coverage = {rule: rule_cluster_coverage[rule] for rule in selected_rules}
        self.gain_value = self.gain(
            selected_rule_labels,
            selected_rule_points,
            selected_rule_coverage
        )
        self.cost_value = self.cost(
            selected_rule_labels,
            selected_rule_points,
            selected_rule_coverage
        )
        self.objective_value = self.compute_objective(
            selected_rule_labels,
            selected_rule_points,
            selected_rule_coverage
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
    def __init__(self, n_rules : int, lambda_val : float = 1.0):
        """
        Args:
            n_rules (int): The *maximum* number of rules to select.
            
            lambda_val (float): A hyperparameter that controls tradeoff between coverage and overlap.
        """
        super().__init__(
            n_rules = n_rules,
            lambda_val = lambda_val
        )


    def gain(
        self,
        selected_rule_labels : dict[int, set[int]],
        selected_rule_points : dict[int, set[int]],
        selected_rule_coverage: dict[int, set[int]]
    ) -> float:
        """
        Computes the gain from the selected rules.

        Args:
            selected_rule_labels (dict[int, set[int]]): A dictionary mapping each selected rule index
            selected_rule_points (dict[int, set[int]]): A dictionary mapping each selected rule index 
                to the full set of data points it covers.
            selected_rule_coverage (dict[int, set[int]]): A dictionary mapping each selected rule index
                to the set of data points it covers within its assigned cluster.
        Returns:
            gain (float): The gain from the selected rules.
        """
        per_cluster_coverage = {}
        for rule, label in selected_rule_labels.items():
            if label not in per_cluster_coverage:
                per_cluster_coverage[label] = set()
            per_cluster_coverage[label] = per_cluster_coverage[label].union(
                selected_rule_coverage[rule]
            )
        total_coverage = 0
        for l, covered in per_cluster_coverage.items():
            total_coverage += len(covered)
        return total_coverage


    def cost(
        self,
        selected_rule_labels : dict[int, set[int]],
        selected_rule_points : dict[int, set[int]],
        selected_rule_coverage: dict[int, set[int]]
    ) -> float:
        """
        Computes the cost of the selected rules.

        Args:
            selected_rule_labels (dict[int, set[int]]): A dictionary mapping each selected rule index
            selected_rule_points (dict[int, set[int]]): A dictionary mapping each selected rule index 
                to the full set of data points it covers.
            selected_rule_coverage (dict[int, set[int]]): A dictionary mapping each selected rule index
                to the set of data points it covers within its assigned cluster.
        Returns:
            cost (float): The cost of the selected rules.
        """
        total_mistakes = 0
        for rule, label in selected_rule_labels.items():
            r_points = selected_rule_points[rule]
            r_coverage = selected_rule_coverage[rule]
            mistakes = r_points.difference(r_coverage)
            total_mistakes += len(mistakes)
        return total_mistakes


    def marginal_gain(
        self,
        rule : int,
        rule_label : int,
        rule_points : dict[int, set[int]],
        rule_cluster_coverage : dict[int, set[int]],
        selected_cluster_coverage : dict[int, set[int]]
    ) -> float:
        """
        Computes the marginal gain as new coverage from selected rule.

        Args:
            rule (int): The index of the rule being considered.
            rule_label (int): The cluster label of the rule being considered.
            rule_points (dict[int, set[int]]): A dictionary mapping each rule index to the full 
                set of data points it covers.
            rule_cluster_coverage (dict[int, set[int]]): A dictionary mapping each rule index 
                to the set of data points it covers within its assigned cluster.
            selected_cluster_coverage (dict[int, set[int]]): A dictionary mapping each cluster
                label to the set of data points already covered by selected rules.
        
        Returns:
            coverage (float): The coverage of the selected rules.
        """
        r_coverage = rule_cluster_coverage[rule]
        s_coverage = selected_cluster_coverage[rule_label]
        new_coverage = r_coverage.difference(s_coverage)
        return len(new_coverage)


    def marginal_cost(
        self,
        rule : int,
        rule_label : int,
        rule_points : dict[int, set[int]],
        rule_cluster_coverage : dict[int, set[int]],
        selected_cluster_coverage : dict[int, set[int]]
    ) -> float:
        """
        Computes the marginal cost as the number of mistakes made by the selected rule.

        Args:
            rule (int): The index of the rule being considered.
            rule_label (int): The cluster label of the rule being considered.
            rule_points (dict[int, set[int]]): A dictionary mapping each rule index to the full 
                set of data points it covers.
            rule_cluster_coverage (dict[int, set[int]]): A dictionary mapping each rule index 
                to the set of data points it covers within its assigned cluster.
            selected_cluster_coverage (dict[int, set[int]]): A dictionary mapping each cluster
                label to the set of data points already covered by selected rules.
        
        Returns:
            cost (float): The cost of the selected rules.
        """
        r_points = rule_points[rule]
        r_coverage = rule_cluster_coverage[rule]
        mistakes = r_points.difference(r_coverage)
        return len(mistakes)
        

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
    def __init__(self, n_rules : int, lambda_val : float = 1.0):
        """
        Args:
            n_rules (int): The *maximum* number of rules to select.
            
            lambda_val (float): A hyperparameter that controls tradeoff between coverage and overlap.
        """
        super().__init__(
            n_rules = n_rules,
            lambda_val = lambda_val
        )


    def gain(
        self,
        selected_rule_labels : dict[int, set[int]],
        selected_rule_points : dict[int, set[int]],
        selected_rule_coverage: dict[int, set[int]]
    ) -> float:
        """
        Computes the gain from the selected rules.

        Args:
            selected_rule_labels (dict[int, set[int]]): A dictionary mapping each selected rule index
            selected_rule_points (dict[int, set[int]]): A dictionary mapping each selected rule index 
                to the full set of data points it covers.
            selected_rule_coverage (dict[int, set[int]]): A dictionary mapping each selected rule index
                to the set of data points it covers within its assigned cluster.
        Returns:
            gain (float): The gain from the selected rules.
        """
        total_coverage = set()
        for rule, label in selected_rule_labels.items():
            r_coverage = selected_rule_points[rule]
            total_coverage = total_coverage.union(r_coverage)
        return len(total_coverage)


    def cost(
        self,
        selected_rule_labels : dict[int, set[int]],
        selected_rule_points : dict[int, set[int]],
        selected_rule_coverage: dict[int, set[int]]
    ) -> float:
        """
        Computes the cost of the selected rules.

        Args:
            selected_rule_labels (dict[int, set[int]]): A dictionary mapping each selected rule index
            selected_rule_points (dict[int, set[int]]): A dictionary mapping each selected rule index 
                to the full set of data points it covers.
            selected_rule_coverage (dict[int, set[int]]): A dictionary mapping each selected rule index
                to the set of data points it covers within its assigned cluster.
        Returns:
            cost (float): The cost of the selected rules.
        """
        total_mistakes = 0
        for rule, label in selected_rule_labels.items():
            r_points = selected_rule_points[rule]
            r_coverage = selected_rule_coverage[rule]
            mistakes = r_points.difference(r_coverage)
            total_mistakes += len(mistakes)
        return total_mistakes

    
    def marginal_gain(
        self,
        rule : int,
        rule_label : int,
        rule_points : dict[int, set[int]],
        rule_cluster_coverage : dict[int, set[int]],
        selected_cluster_coverage : dict[int, set[int]]
    ) -> float:
        """
        Computes the marginal gain as new coverage from selected rule.

        Args:
            rule (int): The index of the rule being considered.
            rule_label (int): The cluster label of the rule being considered.
            rule_points (dict[int, set[int]]): A dictionary mapping each rule index to the full 
                set of data points it covers.
            rule_cluster_coverage (dict[int, set[int]]): A dictionary mapping each rule index 
                to the set of data points it covers within its assigned cluster.
            selected_cluster_coverage (dict[int, set[int]]): A dictionary mapping each cluster
                label to the set of data points already covered by selected rules.
        
        Returns:
            coverage (float): The coverage of the selected rules.
        """
        r_points = rule_points[rule]
        s_points = set()
        for l, covered in selected_cluster_coverage.items():
            s_points = s_points.union(covered)
        new_points = r_points.difference(s_points)
        return len(new_points)


    def marginal_cost(
        self,
        rule : int,
        rule_label : int,
        rule_points : dict[int, set[int]],
        rule_cluster_coverage : dict[int, set[int]],
        selected_cluster_coverage : dict[int, set[int]]
    ) -> float:
        """
        Computes the marginal cost as the number of mistakes made by the selected rule.

        Args:
            rule (int): The index of the rule being considered.
            rule_label (int): The cluster label of the rule being considered.
            rule_points (dict[int, set[int]]): A dictionary mapping each rule index to the full 
                set of data points it covers.
            rule_cluster_coverage (dict[int, set[int]]): A dictionary mapping each rule index 
                to the set of data points it covers within its assigned cluster.
            selected_cluster_coverage (dict[int, set[int]]): A dictionary mapping each cluster
                label to the set of data points already covered by selected rules.
        
        Returns:
            cost (float): The cost of the selected rules.
        """
        r_points = rule_points[rule]
        r_coverage = rule_cluster_coverage[rule]
        mistakes = r_points.difference(r_coverage)
        return len(mistakes)
        

####################################################################################################


class CoverageCostObjective(Objective):
    """
    Objective that selects rules based on a coverage and cluster cost objective.

    Args:
        n_rules (int): The *maximum* number of rules to select.
        lambda_val (float): A hyperparameter that controls tradeoff between coverage and overlap.
            Defaults to 1.0.
        cluster_centers (NDArray): (k x d) array where each row i is the given 
            representative for cluster i.
        method (str): The method used to compute cluster costs. 
            Currently only "kmeans" or "kmedians" are supported.
    """
    def __init__(
            self,
            cluster_centers : NDArray,
            n_rules : int,
            lambda_val : float = 1.0,
            method : str = "kmeans"
        ):
        super().__init__(
            n_rules = n_rules,
            lambda_val = lambda_val,
            cluster_centers = cluster_centers
        )

        if method not in ["kmeans", "kmedians"]:
            raise ValueError(f"Method {method} not supported. Supported methods are 'kmeans' and 'kmedians'.")
        self.method = method


    def gain(
        self,
        selected_rule_labels : dict[int, set[int]],
        selected_rule_points : dict[int, set[int]],
        selected_rule_coverage: dict[int, set[int]]
    ) -> float:
        """
        Computes the gain from the selected rules.

        Args:
            selected_rule_labels (dict[int, set[int]]): A dictionary mapping each selected rule index
            selected_rule_points (dict[int, set[int]]): A dictionary mapping each selected rule index 
                to the full set of data points it covers.
            selected_rule_coverage (dict[int, set[int]]): A dictionary mapping each selected rule index
                to the set of data points it covers within its assigned cluster.
        Returns:
            gain (float): The gain from the selected rules.
        """
        per_cluster_coverage = {}
        for rule, label in selected_rule_labels.items():
            if label not in per_cluster_coverage:
                per_cluster_coverage[label] = set()
            per_cluster_coverage[label] = per_cluster_coverage[label].union(
                selected_rule_coverage[rule]
            )
        total_coverage = 0
        for l, covered in per_cluster_coverage.items():
            total_coverage += len(covered)
        return total_coverage


    def cost(
        self,
        selected_rule_labels : dict[int, set[int]],
        selected_rule_points : dict[int, set[int]],
        selected_rule_coverage: dict[int, set[int]]
    ) -> float:
        """
        Computes the cost of the selected rules.

        Args:
            selected_rule_labels (dict[int, set[int]]): A dictionary mapping each selected rule index
            selected_rule_points (dict[int, set[int]]): A dictionary mapping each selected rule index 
                to the full set of data points it covers.
            selected_rule_coverage (dict[int, set[int]]): A dictionary mapping each selected rule index
                to the set of data points it covers within its assigned cluster.
        Returns:
            cost (float): The cost of the selected rules.
        """
        total_cost = 0.0
        for rule, label in selected_rule_labels.items():
            r_points = selected_rule_points[rule]
            r_center = self.cluster_centers[label]
            if self.method == "kmeans":
                cluster_cost = np.sum((self.data[list(r_points)] - r_center)**2)
            else:  # self.method == "kmedians"
                cluster_cost = np.sum(np.abs(self.data[list(r_points)] - r_center))
            total_cost += cluster_cost
        return total_cost


    def marginal_gain(
        self,
        rule : int,
        rule_label : int,
        rule_points : dict[int, set[int]],
        rule_cluster_coverage : dict[int, set[int]],
        selected_cluster_coverage : dict[int, set[int]]
    ) -> float:
        """
        Computes the marginal gain as new coverage from selected rule.

        Args:
            rule (int): The index of the rule being considered.
            rule_label (int): The cluster label of the rule being considered.
            rule_points (dict[int, set[int]]): A dictionary mapping each rule index to the full 
                set of data points it covers.
            rule_cluster_coverage (dict[int, set[int]]): A dictionary mapping each rule index 
                to the set of data points it covers within its assigned cluster.
            selected_cluster_coverage (dict[int, set[int]]): A dictionary mapping each cluster
                label to the set of data points already covered by selected rules.
        
        Returns:
            coverage (float): The coverage of the selected rules.
        """
        r_coverage = rule_cluster_coverage[rule]
        s_coverage = selected_cluster_coverage[rule_label]
        new_coverage = r_coverage.difference(s_coverage)
        return len(new_coverage)


    def marginal_cost(
        self,
        rule : int,
        rule_label : int,
        rule_points : dict[int, set[int]],
        rule_cluster_coverage : dict[int, set[int]],
        selected_cluster_coverage : dict[int, set[int]]
    ) -> float:
        """
        Computes the marginal cost as the number of mistakes made by the selected rule.

        Args:
            rule (int): The index of the rule being considered.
            rule_label (int): The cluster label of the rule being considered.
            rule_points (dict[int, set[int]]): A dictionary mapping each rule index to the full 
                set of data points it covers.
            rule_cluster_coverage (dict[int, set[int]]): A dictionary mapping each rule index 
                to the set of data points it covers within its assigned cluster.
            selected_cluster_coverage (dict[int, set[int]]): A dictionary mapping each cluster
                label to the set of data points already covered by selected rules.
        
        Returns:
            cost (float): The cost of the selected rules.
        """
        r_points = rule_points[rule]
        r_center = self.cluster_centers[rule_label]
        if self.method == "kmeans":
            cluster_cost = np.sum((self.data[list(r_points)] - r_center)**2)
        else:  # self.method == "kmedians"
            cluster_cost = np.sum(np.abs(self.data[list(r_points)] - r_center))
        return cluster_cost
    

####################################################################################################


class TotalCoverageCostObjective(Objective):
    """
    Objective that selects rules based on a coverage and cluster cost objective. The difference
    with CoverageCostObjective is that the coverage is computed across all clusters,
    rather than within each cluster.

    Args:
        n_rules (int): The *maximum* number of rules to select.
        lambda_val (float): A hyperparameter that controls tradeoff between coverage and overlap.
            Defaults to 1.0.
        cluster_centers (NDArray): (k x d) array where each row i is the given 
            representative for cluster i.
        method (str): The method used to compute cluster costs. 
            Currently only "kmeans" or "kmedians" are supported.
    """
    def __init__(
            self,
            cluster_centers : NDArray,
            n_rules : int,
            lambda_val : float = 1.0,
            method : str = "kmeans"
        ):
        """
        Args:
            n_rules (int): The *maximum* number of rules to select.
            
            lambda_val (float): A hyperparameter that controls tradeoff between coverage and overlap.

            cluster_centers (NDArray): (k x d) array where each row i is the given 
                representative for cluster i.

            method (str): The method used to compute cluster costs. 
                Currently only "kmeans" or "kmedians" are supported.
        """
        super().__init__(
            n_rules = n_rules,
            lambda_val = lambda_val,
            cluster_centers = cluster_centers
        )

        if method not in ["kmeans", "kmedians"]:
            raise ValueError(f"Method {method} not supported. Supported methods are 'kmeans' and 'kmedians'.")
        self.method = method


    def gain(
        self,
        selected_rule_labels : dict[int, set[int]],
        selected_rule_points : dict[int, set[int]],
        selected_rule_coverage: dict[int, set[int]]
    ) -> float:
        """
        Computes the gain from the selected rules.

        Args:
            selected_rule_labels (dict[int, set[int]]): A dictionary mapping each selected rule index
            selected_rule_points (dict[int, set[int]]): A dictionary mapping each selected rule index 
                to the full set of data points it covers.
            selected_rule_coverage (dict[int, set[int]]): A dictionary mapping each selected rule index
                to the set of data points it covers within its assigned cluster.
        Returns:
            gain (float): The gain from the selected rules.
        """
        total_coverage = set()
        for rule, label in selected_rule_labels.items():
            r_coverage = selected_rule_points[rule]
            total_coverage = total_coverage.union(r_coverage)
        return len(total_coverage)


    def cost(
        self,
        selected_rule_labels : dict[int, set[int]],
        selected_rule_points : dict[int, set[int]],
        selected_rule_coverage: dict[int, set[int]]
    ) -> float:
        """
        Computes the cost of the selected rules.

        Args:
            selected_rule_labels (dict[int, set[int]]): A dictionary mapping each selected rule index
            selected_rule_points (dict[int, set[int]]): A dictionary mapping each selected rule index 
                to the full set of data points it covers.
            selected_rule_coverage (dict[int, set[int]]): A dictionary mapping each selected rule index
                to the set of data points it covers within its assigned cluster.
        Returns:
            cost (float): The cost of the selected rules.
        """
        total_cost = 0.0
        for rule, label in selected_rule_labels.items():
            r_points = selected_rule_points[rule]
            r_center = self.cluster_centers[label]
            if self.method == "kmeans":
                cluster_cost = np.sum((self.data[list(r_points)] - r_center)**2)
            else:  # self.method == "kmedians"
                cluster_cost = np.sum(np.abs(self.data[list(r_points)] - r_center))
            total_cost += cluster_cost
        return total_cost


    def marginal_gain(
        self,
        rule : int,
        rule_label : int,
        rule_points : dict[int, set[int]],
        rule_cluster_coverage : dict[int, set[int]],
        selected_cluster_coverage : dict[int, set[int]]
    ) -> float:
        """
        Computes the marginal gain as new coverage from selected rule.

        Args:
            rule (int): The index of the rule being considered.
            rule_label (int): The cluster label of the rule being considered.
            rule_points (dict[int, set[int]]): A dictionary mapping each rule index to the full 
                set of data points it covers.
            rule_cluster_coverage (dict[int, set[int]]): A dictionary mapping each rule index 
                to the set of data points it covers within its assigned cluster.
            selected_cluster_coverage (dict[int, set[int]]): A dictionary mapping each cluster
                label to the set of data points already covered by selected rules.
        
        Returns:
            coverage (float): The coverage of the selected rules.
        """
        r_points = rule_points[rule]
        s_points = set()
        for l, covered in selected_cluster_coverage.items():
            s_points = s_points.union(covered)
        new_points = r_points.difference(s_points)
        return len(new_points)


    def marginal_cost(
        self,
        rule : int,
        rule_label : int,
        rule_points : dict[int, set[int]],
        rule_cluster_coverage : dict[int, set[int]],
        selected_cluster_coverage : dict[int, set[int]]
    ) -> float:
        """
        Computes the marginal cost as the number of mistakes made by the selected rule.

        Args:
            rule (int): The index of the rule being considered.
            rule_label (int): The cluster label of the rule being considered.
            rule_points (dict[int, set[int]]): A dictionary mapping each rule index to the full 
                set of data points it covers.
            rule_cluster_coverage (dict[int, set[int]]): A dictionary mapping each rule index 
                to the set of data points it covers within its assigned cluster.
            selected_cluster_coverage (dict[int, set[int]]): A dictionary mapping each cluster
                label to the set of data points already covered by selected rules.
        
        Returns:
            cost (float): The cost of the selected rules.
        """
        r_points = rule_points[rule]
        r_center = self.cluster_centers[rule_label]
        if self.method == "kmeans":
            cluster_cost = np.sum((self.data[list(r_points)] - r_center)**2)
        else:  # self.method == "kmedians"
            cluster_cost = np.sum(np.abs(self.data[list(r_points)] - r_center))
        return cluster_cost


####################################################################################################