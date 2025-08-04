import numpy as np
import warnings 
from joblib import Parallel, delayed
from typing import Callable, List, Set, Tuple
from numpy.typing import NDArray
import warnings
from intercluster.measurements import (
    coverage,
)
from intercluster.utils import (
    labels_to_assignment,
    assignment_to_dict,
    tiebreak,
)


####################################################################################################


class Pruner:
    """
    Base class for a pruner, which is used to prune rules based on a given objective.
    """
    def __init__(self):
        pass

    def prune(
        self
    ) -> NDArray[np.int64]:
        """
        Prunes the rules based on the given assignment.
                
        Returns:
            selected (NDArray): An array of integers representing the indices of the selected rules.
        """
        pass


####################################################################################################


class CoverageMistakePruner(Pruner):
    """
    Pruner that selects rules based on a coverage and mistake objective.
    """
    def __init__(self, n_rules : int, lambda_val : float):
        """
        Args:
            n_rules (int): The *maximum* number of rules to select.
            
            lambda_val (float): A hyperparameter that controls tradeoff between coverage and overlap.
        """
        super().__init__()
        self.n_rules = n_rules
        self.lambda_val = lambda_val


    def prune(
        self,
        data_to_cluster_assignment : NDArray,
        rule_to_cluster_assignment : NDArray,
        data_to_rules_assignment : NDArray
    ) -> NDArray[np.int64]:
        """
        Prunes the rules based on the given assignment using the coverage and mistake objective.
        
        Args:            
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
        n,k = data_to_cluster_assignment.shape
        r,k2 = rule_to_cluster_assignment.shape
        assert k == k2, "Data and Rule assignment arrays do not match in shape along axis 1."
        assert np.all(np.sum(rule_to_cluster_assignment, axis = 1) == 1), ("Rules must be assigned "
                                                                        "to exactly one cluster.")
        
        rule_list = list(np.arange(r))
        rule_labels = np.array(
            [np.where(rule_to_cluster_assignment[i,:])[0][0] for i in range(r)]
        )
        points_to_cover = assignment_to_dict(data_to_cluster_assignment)
        covered_so_far = {l: set() for l in range(k)}
        
        rule_covers_dict = assignment_to_dict(data_to_rules_assignment)
        rule_label_covers_dict = {}
        for rule, covers in rule_covers_dict.items():
            rule_label = rule_labels[rule]
            label_points_to_cover = points_to_cover[rule_label]
            rule_label_covers_dict[rule] = label_points_to_cover.intersection(covers)
            
        
        selected_rules = set()
        for i in range(self.n_rules):
            # NOTE: perhaps this should include a tiebreak mechansim.
            best_rule = None
            best_rule_label = None
            best_rule_score = -np.inf
            
            for rule in rule_list:
                if rule not in selected_rules:
                    rule_label = rule_labels[rule]
                    rule_covers = rule_covers_dict[rule]
                    rule_label_covers = rule_label_covers_dict[rule]
                    label_covered_so_far = covered_so_far[rule_label]
                    
                    g = len(label_covered_so_far.union(rule_label_covers)) - len(label_covered_so_far)
                    c = len(rule_covers) - len(rule_label_covers)
                    
                    score = (1 - 1/self.n_rules)**(self.n_rules - (i + 1)) * g - self.lambda_val * c
                    
                    if score > best_rule_score:
                        best_rule = rule
                        best_rule_label = rule_label
                        best_rule_score = score
                        
            if best_rule_score > 0:
                selected_rules.add(best_rule)
                best_rule_label_covers = rule_label_covers_dict[best_rule]
                covered_so_far[best_rule_label] = covered_so_far[best_rule_label].union(
                    best_rule_label_covers
                )
                
        return np.array(list(selected_rules))
        


####################################################################################################


def greedy(
    n_rules : int,
    data_to_rules_assignment : NDArray
) -> NDArray:
    
    """
    Implements a greedy algorithm for rule selection. Selects the first n_rules rules that cover 
    data points in the dataset.
    
    Args:
        n_rules (int): The maximum number of rules to select.
        
        data_to_rules_assignment (NDArray): A boolean matrix where entry (i,j) is `True` if 
            data point i is assigned to rule j and `False otherwise.
            
    Returns:
        rule_subset (NDArray): An array of integers representing the indices of the selected rules.
    """
    n,r = data_to_rules_assignment.shape
    index = set(range(r))
    points = set(range(n))
    selected_rules = set()
    covered_points = set()
    for i in range(n_rules):
        remaining_rules = index - selected_rules
        remaining_points = points - covered_points
        rule_list = list(remaining_rules)
        point_list = list(remaining_points)
        sub_assignment = data_to_rules_assignment[:, rule_list]
        sub_assignment = sub_assignment[point_list, :]
        marginal_gain = np.sum(sub_assignment, axis = 0)
        marginal_gain_tiebroken = marginal_gain + np.random.uniform(size = r - len(selected_rules))
        best_rule = np.argmax(marginal_gain_tiebroken)
        best_idx = rule_list[best_rule]
        if marginal_gain[best_rule] > 0:
            selected_rules.add(best_idx)
            covered_points = covered_points.union(
                set(np.where(data_to_rules_assignment[:, best_idx])[0])
            )
        else:
            '''
            warnings.warn(
                "No more rules can be selected that cover data points. "
                "Returning the rules selected so far."
            )
            '''
            break

    return np.array(list(selected_rules))


####################################################################################################


def distorted_greedy(
    n_rules : int,
    lambda_val : float,
    data_to_cluster_assignment : NDArray,
    rule_to_cluster_assignment : NDArray,
    data_to_rules_assignment : NDArray
    ) -> NDArray:
    
    """
    Implements a distorted greedy algorithm for rule selection. Uses an 
    objective function which rewards coverage of datapoints within each cluster,
    but penalizes rules which cover data points from other clusters.
    
    The distorted greedy paradigm is attributed to [Harshaw et al. 2019] in their paper titled
    "Submodular Maximization Beyond Non-negativity: Guarantees, Fast Algorithms, and Applications"
    
    NOTE: The following corresponds to algorithm 1 of their paper, which assumes a fully submodular 
    function g() as part of the objective. Please see their work for more information on distorted 
    greedy and the combination of submodular and modular objectives.
    
    Args:
        n_rules (int): The *maximum* number of rules to select.
        
        lambda_val (float): A hyperparameter that controls tradeoff between coverage and overlap.
        
        data_to_cluster_assignment (np.ndarray): Size (n x k) boolean array where entry (i,j) is 
            `True` if point i is assigned to cluster j and `False` otherwise. Data points may be 
            assigned to multiple clusters. 
        
        rule_to_cluster_assignment (np.ndarray): Size (r x k) boolean array where entry (i,j) is 
            `True` if rule i is assigned to cluster j and `False` otherwise. Each rule must 
            be assigned to a single cluster.
        
        data_to_rules_assignment (NDArray): A boolean matrix where entry (i,j) is `True` if 
            data point i is assigned to rule j and `False otherwise`.
            
    Returns:
        S (NDArray): An array of integers representing the selected rules.
    """
    n,k = data_to_cluster_assignment.shape
    r,k2 = rule_to_cluster_assignment.shape
    assert k == k2, "Data and Rule assignment arrays do not match in shape along axis 1."
    assert np.all(np.sum(rule_to_cluster_assignment, axis = 1) == 1), ("Rules must be assigned "
                                                                       "to exactly one cluster.")
    
    rule_list = list(np.arange(r))
    rule_labels = np.array(
        [np.where(rule_to_cluster_assignment[i,:])[0][0] for i in range(r)]
    )
    points_to_cover = assignment_to_dict(data_to_cluster_assignment)
    covered_so_far = {l: set() for l in range(k)}
    
    rule_covers_dict = assignment_to_dict(data_to_rules_assignment)
    rule_label_covers_dict = {}
    for rule, covers in rule_covers_dict.items():
        rule_label = rule_labels[rule]
        label_points_to_cover = points_to_cover[rule_label]
        rule_label_covers_dict[rule] = label_points_to_cover.intersection(covers)
        
    
    selected_rules = set()
    for i in range(n_rules):
        # NOTE: perhaps this should include a tiebreak mechansim.
        best_rule = None
        best_rule_label = None
        best_rule_score = -np.inf
        
        for rule in rule_list:
            if rule not in selected_rules:
                rule_label = rule_labels[rule]
                rule_covers = rule_covers_dict[rule]
                rule_label_covers = rule_label_covers_dict[rule]
                label_covered_so_far = covered_so_far[rule_label]
                
                g = len(label_covered_so_far.union(rule_label_covers)) - len(label_covered_so_far)
                c = len(rule_covers) - len(rule_label_covers)
                
                score = (1 - 1/n_rules)**(n_rules - (i + 1)) * g - lambda_val * c
                
                if score > best_rule_score:
                    best_rule = rule
                    best_rule_label = rule_label
                    best_rule_score = score
                    
        if best_rule_score > 0:
            selected_rules.add(best_rule)
            best_rule_label_covers = rule_label_covers_dict[best_rule]
            covered_so_far[best_rule_label] = covered_so_far[best_rule_label].union(
                best_rule_label_covers
            )
            
    return np.array(list(selected_rules))


####################################################################################################


def prune_with_grid_search(
    n_rules : int,
    frac_cover : float,
    n_clusters : int,
    data_labels : List[Set[int]],
    rule_labels : List[Set[int]],
    data_to_rules_assignment : NDArray[np.bool_],
    objective : Callable,
    lambda_search_range : NDArray[np.float64],
    cpu_count : int = 1,
    return_full : bool = False
) -> Tuple[NDArray[np.int64], NDArray[np.float64]]:
    
    """
    Performs a grid search over parameter values for
    the distorted greedy objective. Searches for solutions which cover a 
    given fraction of the total data points. 
    
    Args:
        n_rules (int): The number of rules to select.
        
        frac_cover (float): Threshold fraction of the data points required for coverage.
        
        n_clusters (int): The desired number of clusters.
        
        data_labels (List[Set[int]]): Labeling of the data points.
        
        rule_labels (List[Set[int]]): Labeling of the rules.
        
        data_to_rules_assignment (NDArray): A boolean matrix where entry (i,j) is True if 
            data point i is assigned to rule j.
            
        objective (callable): A function that takes the selected rules and returns a score.
        
        lambda_search_range (NDArray): A range of lambda values to search over.
        
        cpu_count (int): The number of cpu cores to use for parallel processing. Default is 8.

        return_full (bool): If True returns the arrays of objective values and 
            coverage values associated with each lambda value in the search range.
            Defaults to False.
            
    Returns:
        S (NDArray): An array of integers representing the selected rules.

        objective_vals (NDArray, optional): Array of objective values where the
            value at index i corresponds to the lambda parameter at index i in search_range.

        coverage_vals (NDArray, optional): Array of coverage values where the
            value at index i corresponds to the lambda parameter at index i in search_range.
    """
    if frac_cover < 0 or frac_cover > 1:
            raise ValueError('Coverage threshold must be between 0 and 1.')
    
    data_to_cluster_assignment = labels_to_assignment(data_labels, n_labels = n_clusters)
    rule_to_cluster_assignment = labels_to_assignment(rule_labels, n_labels = n_clusters)
    
    # Dummy call, used for more precise feedback if any errors are found.
    selected = distorted_greedy(
        n_rules, 
        1,
        data_to_cluster_assignment,
        rule_to_cluster_assignment,
        data_to_rules_assignment
    )
    if len(selected) != 0:
        A = data_to_rules_assignment[:, selected]
        B = rule_to_cluster_assignment[selected, :]
        #pruned_data_to_cluster_assignment = np.dot(A, B)
        pruned_data_to_cluster_assignment = A @ B
        objective(pruned_data_to_cluster_assignment)


    def evaluate_lambda(lambda_val):
        selected = distorted_greedy(
            n_rules,
            lambda_val,
            data_to_cluster_assignment,
            rule_to_cluster_assignment,
            data_to_rules_assignment
        )
        
        # It's possible that nothing was selected, since the cost of selecting 
        # a rule may outweigh the benefits. In this case, we return infinity.
        if len(selected) == 0:
            return np.inf, 0
        
        A = data_to_rules_assignment[:, selected]
        B = rule_to_cluster_assignment[selected, :]
        #pruned_data_to_cluster_assignment = np.dot(A, B)
        pruned_data_to_cluster_assignment = A @ B
        
        cover = coverage(pruned_data_to_cluster_assignment)
        if cover < frac_cover:
            obj = np.inf
        else:
            obj = objective(pruned_data_to_cluster_assignment)

        return obj, cover
               
    search_results = Parallel(n_jobs=cpu_count, backend = 'loky')(
        delayed(evaluate_lambda)(lambd) for lambd in lambda_search_range
    )
    objective_vals = np.array([x[0] for x in search_results])
    cover_vals = np.array([x[1] for x in search_results])

    if np.min(objective_vals) == np.inf:
        warnings.warn(
            "Coverage requirements not met. "
            "Consider adjusting requirements or increasing the search range for lambda. "
            "Returning "
            "None."
        )
        if return_full:
            return None, objective_vals, cover_vals
        
        return None
    
    # Find the lambda value with the smallest objective, breaking ties by preferring 
    # solutions with larger coverage.
    best_lambda_idx = tiebreak(scores = objective_vals, proxy = -1*cover_vals)[0]
    best_lambda = lambda_search_range[best_lambda_idx]

    prune_selection = distorted_greedy(
        n_rules, 
        best_lambda,
        data_to_cluster_assignment,
        rule_to_cluster_assignment,
        data_to_rules_assignment
    )

    if return_full:
        return prune_selection, objective_vals, cover_vals
    
    return prune_selection


####################################################################################################


def prune_with_binary_search(
    n_rules : int,
    frac_cover : float,
    n_clusters : int,
    data_labels : List[Set[int]],
    rule_labels : List[Set[int]],
    data_to_rules_assignment : NDArray[np.bool_],
    objective : Callable,
    lambda_search_range : NDArray[np.float64]
) -> NDArray[np.int64]:
    
    """
    Performs a binary search over lambda parameter values for
    the distorted greedy objective. More specifically, this searches for the largest lambda value 
    for which the solution still satisfies the coverage requirements. 
    
    Args:
        n_rules (int): The number of rules to select.
        
        frac_cover (float): Threshold fraction of the data points required for coverage.
        
        n_clusters (int): The desired number of clusters.
        
        data_labels (List[Set[int]]): Labeling of the data points.
        
        rule_labels (List[Set[int]]): Labeling of the rules.
        
        data_to_rules_assignment (NDArray): A boolean matrix where entry (i,j) is True if 
            data point i is assigned to rule j.
            
        objective (callable): A function that takes the selected rules and returns a score.
        
        lambda_search_range (NDArray): A range of lambda values to search over.
        
        cpu_count (int): The number of cpu cores to use for parallel processing. Default is 8.
            
    Returns:
        S (NDArray): An array of integers representing the selected rules.
    """
    if frac_cover < 0 or frac_cover > 1:
            raise ValueError('Coverage threshold must be between 0 and 1.')
    
    data_to_cluster_assignment = labels_to_assignment(data_labels, n_labels = n_clusters)
    rule_to_cluster_assignment = labels_to_assignment(rule_labels, n_labels = n_clusters)


    def evaluate_lambda(lambda_val):
        selected = distorted_greedy(
            n_rules,
            lambda_val,
            data_to_cluster_assignment,
            rule_to_cluster_assignment,
            data_to_rules_assignment
        )
        
        # It's possible that nothing was selected, since the cost of selecting 
        # a rule may outweigh the benefits. In this case, we return infinity.
        if len(selected) == 0:
            return np.inf, 0
        
        A = data_to_rules_assignment[:, selected]
        B = rule_to_cluster_assignment[selected, :]
        #pruned_data_to_cluster_assignment = np.dot(A, B)
        pruned_data_to_cluster_assignment = A @ B
        
        obj = objective(pruned_data_to_cluster_assignment)
        cover = coverage(pruned_data_to_cluster_assignment)
        return obj, cover
    

    sorted_search_range = np.sort(lambda_search_range)
    n = len(sorted_search_range)
    left_index = 0
    right_index = n - 1
    midpoint = None
    best_lambda = sorted_search_range[0]
    covered = False

    while left_index <= right_index:
        midpoint = (left_index + right_index) // 2
        lambd = sorted_search_range[midpoint]
        obj,cover = evaluate_lambda(lambd)
        if cover < frac_cover:
            right_index = midpoint - 1
        else:
            covered = True
            best_lambda = lambd
            left_index = midpoint + 1

    if not covered:
        return None
    
    prune_selection = distorted_greedy(
        n_rules, 
        best_lambda,
        data_to_cluster_assignment,
        rule_to_cluster_assignment,
        data_to_rules_assignment
    )
    return prune_selection


####################################################################################################
