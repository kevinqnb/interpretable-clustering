import numpy as np
import warnings 
from joblib import Parallel, delayed
from typing import Callable, List
from numpy.typing import NDArray
import warnings
from intercluster.utils import (
    labels_to_assignment,
    flatten_labels,
    assignment_to_dict,
    tiebreak,
)


####################################################################################################


def distorted_greedy(
    q : int,
    lambda_val : float,
    data_labels : List[List[int]],
    rule_labels : List[List[int]],
    data_to_rules_assignment : NDArray[np.bool_]
    ) -> NDArray[np.int64]:
    
    """
    Implements a distorted greedy algorithm for rule selection. Uses an 
    objective function which rewards coverage of datapoints within each cluster,
    but penalizes rules which cover data points from other clusters.
    
    The distorted greedy paradigm is attributed to [Harshaw et al. 2019] in their paper titled
    "Submodular Maximization Beyond Non-negativity: Guarantees, Fast Algorithms, and Applications"
    
    Please see their work for more information on distorted greedy and the combination of 
    submodular and linear objectives.
    
    Args:
        q (int): The number of rules to select.
        
        lambda_val (float): A hyperparameter that controls tradeoff between coverage and overlap.
        
        data_labels (List[List[int]]): A 2d list of integers where the inner list at index i 
            represents the cluster labels of the data point with index i.
            NOTE: Each data point can have multiple labels.
        
        rule_labels (List[List[int]]): A 2d list of integers where the inner list at index i 
            represents the cluster labels of the rule with index i.
            NOTE: Each rule can only have a single label, so each inner list must be length 1.
        
        data_to_rules_assignment (NDArray): A boolean matrix where entry (i,j) is True if 
            data point i is assigned to rule j.
            
    Returns:
        S (NDArray): An array of integers representing the selected rules.
    """
    n = len(data_labels)
    rule_list = list(np.arange(data_to_rules_assignment.shape[1]))
    flattened_rule_labels = flatten_labels(rule_labels)
    if len(flattened_rule_labels) != len(rule_labels):
        raise ValueError("Each rule must have exactly one label.")
    
    unique_data_labels = np.unique(flatten_labels(data_labels))
    unique_rule_labels = np.unique(flattened_rule_labels)
    if not np.array_equal(unique_data_labels, unique_rule_labels):
        warnings.warn(
            "The set of unique data labels and set of unique rule labels do not match. "
            "This may lead to unexpected behavior."
        )
    #n_labels = len(unique_cluster_labels)
    n_labels = len(unique_data_labels)
    
    rule_covers_dict = assignment_to_dict(data_to_rules_assignment)
    points_to_cluster_assignment = labels_to_assignment(data_labels, n_labels = n_labels)
    points_to_cover = assignment_to_dict(points_to_cluster_assignment)
    #covered_so_far = {l: set() for l in unique_cluster_labels}
    covered_so_far = {l: set() for l in unique_data_labels}
    
    S = []
    points_satisfied_by_rules = set()
    i = 0
    # for i in range(q):
    while i < q and len(points_satisfied_by_rules) < n:
        best_rule = None
        best_rule_label = None
        best_score = -np.inf
        
        for r in rule_list:
            if r not in S:
                r_label = rule_labels[r][0]
                label_points_to_cover = points_to_cover[r_label]
                label_covered_so_far = covered_so_far[r_label]
                r_covers = label_points_to_cover.intersection(rule_covers_dict[r])
                
                g = len(label_covered_so_far.union(r_covers)) - len(label_covered_so_far)
                c = len(rule_covers_dict[r]) - len(r_covers)
                
                score = (1 - 1/q)**(q - (i + 1)) * g - lambda_val * c
                
                if score > best_score:
                    best_rule = r
                    best_rule_label = r_label
                    best_score = score
                    
        if best_score > 0:
            S.append(best_rule)
            best_rule_covers = points_to_cover[best_rule_label].intersection(
                rule_covers_dict[best_rule]
            )
            covered_so_far[best_rule_label] = covered_so_far[best_rule_label].union(
                best_rule_covers
            )
            # Union of all points covered by rules in S. (For now...maybe this should be
            # more specific to cluster membership??)
            points_satisfied_by_rules = points_satisfied_by_rules.union(rule_covers_dict[best_rule])
            
        i += 1
            
    return np.array(S)


####################################################################################################


def prune_with_grid_search(
    q : int,
    k : int,
    data_labels : List[List[int]],
    rule_labels : List[List[int]],
    data_to_rules_assignment : NDArray[np.bool_],
    objective : Callable,
    lambda_search_range : NDArray[np.float64],
    cpu_count : int = 8
    ) -> NDArray[np.int64]:
    
    """
    Performs a grid search over normalization values for
    the distorted greedy objective. Returns the set of rules that ...
    
    Args:
        q (int): The number of rules to select.
        
        k (int): The desired number of clusters.
        
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
    # This is a dummy call -- used to catch any errors in the input. 
    distorted_greedy(q, 1, data_labels, rule_labels, data_to_rules_assignment)
    rule_to_cluster_assignment = labels_to_assignment(rule_labels, n_labels = k)
    
    def evaluate_lambda(lambda_val):
        selected = distorted_greedy(
            q,
            lambda_val,
            data_labels,
            rule_labels,
            data_to_rules_assignment
        )
        
        # It's possible that nothing was selected, since the cost of selecting 
        # a rule may outweigh the benefits. In this case, we return infinity.
        if len(selected) == 0:
            return (np.inf, np.inf), lambda_val
        
        A = data_to_rules_assignment[:, selected]
        B = rule_to_cluster_assignment[selected, :]
        data_to_cluster_assignment = np.dot(A, B)
        return objective(data_to_cluster_assignment), lambda_val
               
    search_results = Parallel(n_jobs=cpu_count)(
        delayed(evaluate_lambda)(lambd) for lambd in lambda_search_range
    )
    objective_vals = [x[0][0] for x in search_results]
    if np.min(objective_vals) == np.inf:
        warnings.warn(
            "Coverage requirements not met. "
            "Consider adjusting requirements or increasing the search range for lambda."
        )
    tiebreak_vals = [x[0][1] for x in search_results]
    best_lambda_idx = tiebreak(scores = objective_vals, proxy = tiebreak_vals)[0]
    best_lambda = lambda_search_range[best_lambda_idx]
            
    return distorted_greedy(q, best_lambda, data_labels, rule_labels, data_to_rules_assignment) 


####################################################################################################