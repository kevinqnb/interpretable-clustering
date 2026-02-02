import numpy as np
from typing import List, Set
from numpy.typing import NDArray
from intercluster import (
    Rule,
    Decision,
    satisfies_rule,
    labels_to_assignment,
    unique_labels,
    simplify_decision,
    simplified_rule_length
)


class DecisionSet:
    """
    Base class for a decision set.
    """
    def __init__(
        self,
        rules : List[Rule],
        rule_labels : List[Set[int]] = None,
    ):
        """
        Args:
            rules (List[Rule], optional): List of rules to select from.
            rule_labels (List[Set[int]], optional): List of labels corresponding to each rule.
                If None, the labels will be created using the input dataset. Defaults to None.

        Attributes:
            decision_set (List[Rule]): List of rules in the selected decision set.

            rule_length (int): Maximum rule length.
        """
        if not isinstance(rules, list) or not all(isinstance(r, Rule) for r in rules):
            raise ValueError("rules must be a list of Rule objects.")
        self.rules = rules

        if rule_labels is not None:
            assert isinstance(rule_labels, list) and all(isinstance(lbl, set) for lbl in rule_labels), \
                "rule_labels must be a list of sets."
            
            for lbl in rule_labels:
                if len(lbl) != 1:
                    raise ValueError("Each set in rule_labels must contain exactly one label.")

        self.rule_labels = rule_labels

        self.decision_set = None
        self.max_rule_length = 0
    

    def select(self, X : NDArray, y : List[Set[int]] = None) -> set[Decision]:
        """
        selects the decision set using the selector.
        
        Args:
            X (np.ndarray): Input dataset.
            
            y (List[Set[int]], optional): Target labels. Defaults to None.
        """
        if self.decision_set is None:
            raise ValueError('Decision set has not been initialized yet.')
        
        pass
    

    def trim(self) -> set[Decision]:
        """Simplify rules in the decision set (remove degenerate + redundant conditions)."""
        if self.decision_set is None:
            raise ValueError('Decision set has not been fitted yet.')

        trimmed_set: set[Decision] = set()
        for decision in self.decision_set:
            trimmed_decision = simplify_decision(decision)
            if len(trimmed_decision.rule) > 0:
                trimmed_set.add(trimmed_decision)
        return trimmed_set
    

    def set_labels(self, X : NDArray, y : List[Set[int]]):
        """
        Sets the labels of the decision rules in the decision set based on the input dataset.
        
        Args:
            X (np.ndarray): Input dataset.
            
            y (List[Set[int]]): Target labels.
        """
        if self.rule_labels is None and y is None:
            # Each decision rule gets its own unique label, independent of y
            self.decision_set = {
                Decision(rule, i) for i, rule in enumerate(self.decision_set)
            }

            y = [set() for _ in range(X.shape[0])]
            for i, decision in enumerate(self.decision_set):
                data_points_satisfied = satisfies_rule(X, decision.rule)
                for j in data_points_satisfied:
                    y[j] = y[j].union({decision.label})

        elif self.rule_labels is None and y is not None:
            # Each rule is assigned to every possible unique label from y
            ulabels = unique_labels(y)
            self.decision_set = set()
            for i, rule in enumerate(self.rules):
                for u in ulabels:
                    self.decision_set.add(
                        Decision(rule, u)
                    )

        elif self.rule_labels is not None and y is None:
            self.decision_set = set()
            for i, rule in enumerate(self.rules):
                self.decision_set.add(
                    Decision(rule, next(iter(self.rule_labels[i])))
                )

            y = [set() for _ in range(X.shape[0])]
            for decision in self.decision_set:
                data_points_satisfied = satisfies_rule(X, decision.rule)
                for j in data_points_satisfied:
                    y[j] = y[j].union({decision.label})

        else:
            self.decision_set = set()
            for i, rule in enumerate(self.rules):
                self.decision_set.add(
                    Decision(rule, next(iter(self.rule_labels[i])))
                )
            
        
        # Remove rules covering outliers
        self.decision_set = {
            decision for decision in self.decision_set if decision.label != -1
        }

        return y
        
        
        
    def fit(self, X : NDArray, y : List[Set[int]] = None):
        """
        Public fit function. 
        Fits a decision set to an input dataset. 
        
        Args:
            X (np.ndarray): Input dataset.
            
            y (List[Set[int]], optional): Target labels. Defaults to None.
        """
        y = self.set_labels(X, y)
        self.decision_set = self.select(X, y)
        self.decision_set = self.trim()
        self.max_rule_length = max([len(decision.rule) for decision in self.decision_set]) \
            if self.decision_set else 0

        self.decision_set = list(self.decision_set)


    def get_data_to_rules_assignment(self, X : NDArray) -> NDArray:
        """
        Finds data points of X covered by each rule in the decision set.
        
        Args:
            X (np.ndarray): Input dataset.
            
        Returns:
            assignment (np.ndarray): n x n_rules boolean matrix with entry (i,j) being True
                if point i is covered by rule j and False otherwise.
        """
        assignment = np.zeros((X.shape[0], len(self.decision_set)), dtype = bool)
        for i, decision in enumerate(self.decision_set):
            assignment[:, i] = decision.rule.evaluate(X)
        return assignment
    

    def get_rules_to_clusters_assignment(self, n_labels : int) -> NDArray:
        """
        Finds data points of X covered by each rule in the decision set.
        
        Args:
            n_labels (int): Number of labels in the dataset.
            
        Returns:
            assignment (np.ndarray): n_rules x k boolean matrix with entry (i,j) being True
                if point i is covered by rule j and False otherwise.
        """
        decision_set_labels = [{decision.label} for decision in self.decision_set]
        assignment = labels_to_assignment(decision_set_labels, n_labels)
        return assignment
    
    
    def predict(self, X : NDArray, rule_labels : bool = False) -> List[Set[int]]:
        """
        Predicts the label(s) of each data point in X.
        
        Args:
            X (np.ndarray): Input dataset.
            
            rule_labels (bool, optional): If true, gives labels based soley upon 
                rule membership. That is, each rule is given a unique label. 
                Otherwise, returns the orignal predictions from the fitted rule models -- 
                whatever label is given to the rule. Defaults to False.
            
        Returns:
            labels (List[Set[int]]): 2d list of predicted labels, with the internal list 
                at index i representing the group of decision rules which satisfy X[i,:].
        """
        labels = [set() for _ in range(len(X))]
        for i, decision in enumerate(self.decision_set):
            r_covers = satisfies_rule(X, decision.rule)
            for j in r_covers:
                if rule_labels:
                    labels[j].add(i)
                else:
                    labels[j] = labels[j].union({decision.label})
        
        return labels


    def get_weighted_average_rule_length(self, X : NDArray) -> float:
        """
        Finds the weighted average length of the rules (after redundancy removal),
        weighted by the number of data points covered by each rule.
        """
        wad = 0
        total_covers = 0
        for decision in self.decision_set:
            r_covers = satisfies_rule(X, decision.rule)
            total_covers += len(r_covers)
            if len(r_covers) != 0:
                wad += len(r_covers) * simplified_rule_length(decision.rule)

        if total_covers == 0:
            return np.nan
        return wad / total_covers
        
    
    def get_sum_of_rule_lengths(self) -> float:
        """
        Finds the sum of the lengths of the rules.

        NOTE: If the decision set has been selectd this will automatically use the 
            selectd decision set.

        Args:

        Returns:
            sum (float): Sum of lengths of all rules.
        """
        return sum([simplified_rule_length(decision.rule) for decision in self.decision_set])