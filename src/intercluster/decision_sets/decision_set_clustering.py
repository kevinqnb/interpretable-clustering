import numpy as np
from typing import List, Set, Tuple
from numpy.typing import NDArray
from intercluster import (
    Condition,
    LinearCondition,
    satisfies_conditions,
    can_flatten,
    labels_to_assignment,
    unique_labels
)
from intercluster.mining import RuleMiner
from intercluster.pruning import CoverageMistakePruner
from .decision_set import DecisionSet


####################################################################################################


class DSCluster(DecisionSet):
    """
    Collection of rules drawn as boxes (rules) around collections of points in the dataset.
    """
    def __init__(
        self,
        lambd : float,
        n_rules : int,
        rule_miner : RuleMiner = None, 
        rules : List[List[Condition]] = None,
        rule_labels : List[Set[int]] = None
    ):
        """
        Args:
            lambd (float): Penalization factor for mistakes. Larger values penalize mistakes 
                more heavily, resulting in rules which are more accurate, but may cover 
                fewer points.
            n_rules (int): Number of rules to use in the decision set.
            rule_miner (RuleMiner, optional): Rule mining algorithm used to generate the rules.
                If None, the rules must be provided directly. Defaults to None.
            rules (List[List[Condition]], optional): List of rules to initialize the decision set with.
                If None, the rules will be generated using the rule_miner. Defaults to None.
            rule_labels (List[Set[int]], optional): List of labels corresponding to each rule.
                If None, the labels will be generated using the rule_miner. Defaults to None.
        """
        super().__init__(rule_miner, rules, rule_labels)
        self.lambd = lambd
        self.n_rules = n_rules
        self.pruner = CoverageMistakePruner(n_rules=n_rules, lambda_val=lambd)
    

    def prune(
            self,
            X : NDArray,
            y : List[Set[int]]
        ) -> List[List[Condition]]:
        """
        Prunes the decision set by removing rules that do not cover any points in the dataset.

        Args:
            X (np.ndarray): Input dataset.
            y (List[Set[int]]): Target labels.
        """
        if self.decision_set is None or self.decision_set_labels is None:
            raise ValueError('Decision set has not been fitted yet.')
        
        # Remove rules covering outliers
        self.decision_set = [rule for i,rule in enumerate(self.decision_set) 
                             if self.decision_set_labels[i] != {-1}]
        self.decision_set_labels = [label for label in self.decision_set_labels if label != {-1}]
        
        n_labels = len(unique_labels(y, ignore ={-1}))
        data_to_cluster_assignment = labels_to_assignment(
            y, n_labels = n_labels, ignore = {-1}
        )
        rule_to_cluster_assignment = labels_to_assignment(
            self.decision_set_labels, n_labels = n_labels, ignore = {-1}
        )
        data_to_rules_assignment = self.get_data_to_rules_assignment(X, self.decision_set)
        selected_rules = self.pruner.prune(
            data_to_cluster_assignment = data_to_cluster_assignment,
            rule_to_cluster_assignment = rule_to_cluster_assignment,
            data_to_rules_assignment = data_to_rules_assignment
        )

        self.decision_set = [self.decision_set[i] for i in selected_rules]
        self.decision_set_labels = [self.decision_set_labels[i] for i in selected_rules]


    def get_data_to_rules_assignment(
            self,
            X : NDArray,
            decision_set : List[List[Condition]] = None
        ) -> NDArray:
        """
        Finds data points of X covered by each rule in the decision set.
        
        Args:
            X (np.ndarray): Input dataset.
            
            rule_list (List[List[Condition]], optional): List of rules to use for assignment.
        Returns:
            assignment (np.ndarray): n x n_rules boolean matrix with entry (i,j) being True
                if point i is covered by rule j and False otherwise.
        """
        if decision_set is None:
            decision_set = self.decision_set
        assignment = np.zeros((X.shape[0], len(decision_set)), dtype=bool)
        for i, condition_list in enumerate(decision_set):
            data_points_satisfied = satisfies_conditions(X, condition_list)
            assignment[data_points_satisfied, i] = True
        return assignment
    

####################################################################################################