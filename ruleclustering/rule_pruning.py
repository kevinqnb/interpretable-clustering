import numpy as np
import copy
from sklearn.cluster import KMeans
from joblib import Parallel, delayed
from typing import Dict, Callable


####################################################################################################


def distorted_greedy(
    q : int,
    lambda_val : float,
    data_labels : np.ndarray[np.int64],
    rule_labels : np.ndarray[np.int64],
    rule_covers_dict : Dict[int, np.ndarray[np.int64]]
    ) -> np.ndarray[np.int64]:
    
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
        
        data_labels (np.ndarray): An array of integers representing the cluster labels of the data.
        
        rule_labels (np.ndarray): An array of integers representing the cluster labels of the rules.
        
        rule_covers_dict (dict[int: set]): A dictionary where keys are integers (rule labels) and 
            values are the sets of data point indices covered by the rule.
            
    Returns:
        S (np.ndarray): An array of integers representing the selected rules.
    """
    
    unique_labels = np.unique(rule_labels)
    points_to_cover = {l: set(np.where(data_labels == l)[0]) for l in unique_labels}
    covered_so_far = {l: set() for l in unique_labels}
    
    S = []
    rule_list = list(rule_covers_dict.keys())
    for i in range(q):
        best_rule = None
        best_rule_label = None
        best_score = -np.inf
        
        for r in rule_list:
            if r not in S:
                rlabel = rule_labels[r]
                label_covered = covered_so_far[rlabel]
                to_cover = points_to_cover[rlabel]
                r_covers = to_cover.intersection(rule_covers_dict[r])
                
                g = len(label_covered.union(r_covers)) - len(label_covered)
                c = len(rule_covers_dict[r]) - len(r_covers)
                
                score = (1 - 1/q)**(q - (i + 1)) * g - lambda_val * c
                
                if score > best_score:
                    best_rule = r
                    best_rule_label = rlabel
                    best_score = score
                    
        if best_score > 0:
            S.append(best_rule)
            covered_so_far[best_rule_label] = covered_so_far[best_rule_label].union(
                rule_covers_dict[best_rule]
            )
            
    return np.array(S)


####################################################################################################


def prune_with_grid_search(
    q : int,
    data_labels : np.ndarray[np.int64],
    rule_labels : np.ndarray[np.int64],
    rule_covers_dict : Dict[int, np.ndarray[np.int64]],
    objective : Callable,
    search_range : np.ndarray[np.float64],
    coverage_threshold : float
    ) -> np.ndarray[np.int64]:
    
    """
    Performs a grid search over normalization values for
    the distorted greedy objective. Returns the set of rules that ...
    
    Args:
        q (int): The number of rules to select.
        
        data_labels (np.ndarray): An array of integers representing the cluster labels of the data.
        
        rule_labels (np.ndarray): An array of integers representing the cluster labels of the rules.
        
        rule_covers_dict (dict[int: set]): A dictionary where keys are integers (rule labels) and 
            values are the sets of data point indices covered by the rule.
            
        objective (callable): A function that takes the selected rules and returns a score.
        
        search_range (np.ndarray): A range of lambda values to search over.
        
        coverage_threshold (float): The minimum number of data points that must
            be covered by the selected rules.
            
    Returns:
        S (np.ndarray): An array of integers representing the selected rules.
    """
    
    def evaluate_lambda(lambda_val):
        selected = distorted_greedy(q, lambda_val, data_labels, rule_labels, rule_covers_dict)
        
        covered = set()
        for r in selected:
            covered = covered.union(rule_covers_dict[r])
            
        if len(covered) < coverage_threshold:
            return np.inf, lambda_val
        
        else:
            return objective(selected), lambda_val
               
    search_results = Parallel(n_jobs=-1)(delayed(evaluate_lambda)(s) 
                                            for s in search_range)
    best_score, best_val = min(search_results, key=lambda x: x[0])
            
    return distorted_greedy(q, best_val, data_labels, rule_labels, rule_covers_dict) 

