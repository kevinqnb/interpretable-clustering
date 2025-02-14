import numpy as np
import warnings 
from joblib import Parallel, delayed
from typing import Callable, List
from numpy.typing import NDArray
import warnings
from intercluster.utils import (
    labels_to_assignment,
    assignment_to_dict,
    tiebreak,
    coverage
)


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
    assert (
        np.all(np.sum(rule_to_cluster_assignment, axis = 1) == 1), 
        "Rules must be assigned to exactly one cluster."
    )
    
    rule_list = list(np.arange(r))
    rule_labels = np.array(
        [np.where(rule_to_cluster_assignment[i,:])[0][0] for i in range(r)]
    )
    points_to_cover = assignment_to_dict(data_to_cluster_assignment)
    covered_so_far = {l: set() for l in range(k)}
    
    rule_covers_dict = assignment_to_dict(data_to_rules_assignment)
    rule_label_covers_dict = {}
    for rule, covers in rule_covers_dict.items():
        rule_label = rule_labels[rule][0]
        label_points_to_cover = points_to_cover[rule_label]
        rule_label_covers_dict[rule] = label_points_to_cover.intersection(covers)
        
    
    selected_rules = set()
    for i in range(n_rules):
        best_rule = None
        best_rule_label = None
        best_rule_score = -np.inf
        
        for rule in rule_list:
            if rule not in selected_rules:
                rule_label = rule_labels[rule][0]
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
    data_labels : List[List[int]],
    rule_labels : List[List[int]],
    data_to_rules_assignment : NDArray[np.bool_],
    objective : Callable,
    lambda_search_range : NDArray[np.float64],
    cpu_count : int = 8
) -> NDArray[np.int64]:
    
    """
    Performs a grid search over parameter values for
    the distorted greedy objective. Searches for solutions which cover a 
    given fraction of the total data points. 
    
    Args:
        n_rules (int): The number of rules to select.
        
        frac_cover (float): Threshold fraction of the data points required for coverage.
        
        n_clusters (int): The desired number of clusters.
        
        data_labels (NDArray): An array of integers representing the cluster labels of the data.
        
        rule_labels (NDArray): An array of integers representing the cluster labels of the rules.
        
        data_to_rules_assignment (NDArray): A boolean matrix where entry (i,j) is True if 
            data point i is assigned to rule j.
            
        objective (callable): A function that takes the selected rules and returns a score.
        
        search_range (NDArray): A range of lambda values to search over.
        
        cpu_count (int): The number of cpu cores to use for parallel processing. Default is 8.
            
    Returns:
        S (NDArray): An array of integers representing the selected rules.
    """
    if frac_cover < 0 or frac_cover > 1:
            raise ValueError('Coverage threshold must be between 0 and 1.')
    
    data_to_cluster_assignment = labels_to_assignment(data_labels, n_labels = n_clusters)
    rule_to_cluster_assignment = labels_to_assignment(rule_labels, n_labels = n_clusters)
    
    # Dummy call, used for more precise feedback if any errors are found.
    distorted_greedy(
        n_rules, 
        1,
        data_to_cluster_assignment,
        rule_to_cluster_assignment,
        data_to_rules_assignment
    )
    
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
            return (np.inf, np.inf), lambda_val
        
        A = data_to_rules_assignment[:, selected]
        B = rule_to_cluster_assignment[selected, :]
        data_to_cluster_assignment = np.dot(A, B)
        obj = objective(data_to_cluster_assignment)
        
        # If frac_cover points are not covered, return infinity.
        if coverage(data_to_cluster_assignment) < frac_cover:
            return (np.inf, np.inf), lambda_val
        
        # Otherwise, return the objective along with a random tiebreak value.        
        return (obj, np.random.uniform()), lambda_val
               
    search_results = Parallel(n_jobs=cpu_count)(
        delayed(evaluate_lambda)(lambd) for lambd in lambda_search_range
    )
    
    objective_vals = [x[0][0] for x in search_results]
    if np.min(objective_vals) == np.inf:
        warnings.warn(
            "Coverage requirements not met. "
            "Consider adjusting requirements or increasing the search range for lambda. "
            "Returning None."
        )
        return None
    
    tiebreak_vals = [x[0][1] for x in search_results]
    best_lambda_idx = tiebreak(scores = objective_vals, proxy = tiebreak_vals)[0]
    best_lambda = lambda_search_range[best_lambda_idx]
    return distorted_greedy(
        n_rules, 
        best_lambda,
        data_to_cluster_assignment,
        rule_to_cluster_assignment,
        data_to_rules_assignment
    )


####################################################################################################