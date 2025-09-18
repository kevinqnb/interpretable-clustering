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

try:
    from intercluster.pruning.coverage_mistake_prune import coverage_mistake_prune_cy
    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False

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
    This makes use of a distorted greedy approach to select rules.
    
    For more information, see the following paper:
    "Submodular Maximization Beyond Non-Negativity: Guarantees, Fast Algorithms, and Applications"
    by Harshaw el al., ICML 2019.
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
        
        # Use Cython implementation if available
        if CYTHON_AVAILABLE:
            return coverage_mistake_prune_cy(
                data_to_cluster_assignment.astype(np.bool_),
                rule_to_cluster_assignment.astype(np.bool_),
                data_to_rules_assignment.astype(np.bool_),
                self.n_rules,
                self.lambda_val
            )
        
        # Fallback to original Python implementation
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