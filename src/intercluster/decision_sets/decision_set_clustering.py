import numpy as np
from typing import List, Set, Tuple
from numpy.typing import NDArray
from intercluster import (
    Condition,
    satisfies_conditions,
    labels_to_assignment,
    unique_labels
)
from .mining import RuleMiner
from .objectives import Objective
from .decision_set import DecisionSet


####################################################################################################


class DSCluster(DecisionSet):
    """
    Collection of rules drawn as boxes (rules) around collections of points in the dataset.
    """
    def __init__(
        self,
        objective : Objective,
        rule_miner : RuleMiner = None, 
        rules : List[List[Condition]] = None,
        rule_labels : List[Set[int]] = None,
    ):
        """
        Args:
            objective (Objective): Objective function class used to select the rules.
            rule_miner (RuleMiner, optional): Rule mining algorithm used to generate the rules.
                If None, the rules must be provided directly. Defaults to None.
            rules (List[List[Condition]], optional): List of rules to initialize the decision set with.
                If None, the rules will be generated using the rule_miner. Defaults to None.
            rule_labels (List[Set[int]], optional): List of labels corresponding to each rule.
                If None, the labels will be generated using the rule_miner. Defaults to None.
        """
        super().__init__(rule_miner, rules, rule_labels)
        self.objective = objective


    def set_lambda(
        self,
        lambda_val : float
    ) -> None:
        """
        Sets the lambda value for the objective function.

        Args:
            lambda_val (float): The lambda value to set.
        """
        self.objective.set_lambda(lambda_val)


    def filter_rules(
        self,
        X : NDArray,
        y : List[Set[int]] = None,
        remove_top : float = 0.1
    ) -> NDArray:
        """
        Computes the lambda value for the objective function based on the dataset.

        Args:
            X (np.ndarray): Input dataset.
            y (List[Set[int]]): Target labels.
            remove_top (float, optional): The proportion of rules to remove from the top. 
                Defaults to 0.1.
        Returns:
            lambda_vals (NDArray): A sorted array of lambda values, starting from the minimum 
                most value for which the approximation guarantee holds, and increasing
                until reaching the maximum coverage/cost ratio seen
                for any (rule, cluster) assignment pair. 
        """
        if self.rules is None:
            raise ValueError('Rules have not been mined yet.')

        # DO I NEED TO WORRY about {-1} labels here??
        if y is None:
            y = [{-1} for _ in range(X.shape[0])]
            n_labels = 1
        else:
            n_labels = len(unique_labels(y, ignore ={-1}))

        data_to_cluster_assignment = labels_to_assignment(
            y, n_labels = n_labels, ignore = {-1}
        )
        data_to_rules_assignment = self.get_data_to_rules_assignment(X, self.rules)
        rule_lengths = [len(rule) for rule in self.rules]
        rules_indices = self.objective.filter_rules(
            data = X,
            data_to_cluster_assignment = data_to_cluster_assignment,
            data_to_rules_assignment = data_to_rules_assignment,
            rule_lengths = rule_lengths,
            remove_top = remove_top
        )

        self.rules = [self.rules[i] for i in rules_indices]
        if self.rule_labels is not None:
            self.rule_labels = [self.rule_labels[i] for i in rules_indices]

    
    def compute_lambdas(
        self,
        X : NDArray,
        y : List[Set[int]] = None
    ) -> NDArray:
        """
        Computes the lambda value for the objective function based on the dataset.

        Args:
            X (np.ndarray): Input dataset.
            y (List[Set[int]]): Target labels.

        Returns:
            lambda_vals (NDArray): A sorted array of lambda values, starting from the minimum 
                most value for which the approximation guarantee holds, and increasing
                until reaching the maximum coverage/cost ratio seen
                for any (rule, cluster) assignment pair. 
        """
        if self.rules is None:
            raise ValueError('Rules have not been mined yet.')
        

        # DO I NEED TO WORRY about {-1} labels here??
        if y is None:
            y = [{-1} for _ in range(X.shape[0])]
            n_labels = 1
        else:
            n_labels = len(unique_labels(y, ignore ={-1}))

        data_to_cluster_assignment = labels_to_assignment(
            y, n_labels = n_labels, ignore = {-1}
        )
        data_to_rules_assignment = self.get_data_to_rules_assignment(X, self.rules)
        rule_lengths = [len(rule) for rule in self.rules]
        lambda_vals = self.objective.compute_lambdas(
            data = X,
            data_to_cluster_assignment = data_to_cluster_assignment,
            data_to_rules_assignment = data_to_rules_assignment,
            rule_lengths = rule_lengths
        )
        return lambda_vals


    def select(
            self,
            X : NDArray,
            y : List[Set[int]]
        ) -> List[List[Condition]]:
        """
        selects the decision set by removing rules that do not cover any points in the dataset.

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
        rule_lengths = [len(rule) for rule in self.decision_set]
        
        n_labels = len(unique_labels(self.decision_set_labels, ignore ={-1}))
        data_to_cluster_assignment = labels_to_assignment(
            y, n_labels = n_labels, ignore = {-1}
        )
        rule_to_cluster_assignment = labels_to_assignment(
            self.decision_set_labels, n_labels = n_labels, ignore = {-1}
        )
        data_to_rules_assignment = self.get_data_to_rules_assignment(X, self.decision_set)
        selected_rules = self.objective.select(
            data = X,
            data_to_cluster_assignment = data_to_cluster_assignment,
            rule_to_cluster_assignment = rule_to_cluster_assignment,
            data_to_rules_assignment = data_to_rules_assignment,
            rule_lengths = rule_lengths
        )

        selected_set = [self.decision_set[i] for i in selected_rules]
        selected_set_labels = [self.decision_set_labels[i] for i in selected_rules]
        return selected_set, selected_set_labels


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