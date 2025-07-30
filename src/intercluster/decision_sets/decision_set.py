import numpy as np
from typing import List, Set, Tuple
from numpy.typing import NDArray
from intercluster import (
    Condition,
    satisfies_conditions,
    labels_to_assignment
)


class DecisionSet:
    """
    Base class for a decision set.
    """
    def __init__(
        self
    ):
        
        """            
        Attributes:
            decision_set (List[Condition]): List of rules in the decision set.
            
            decision_set_labels (List[Set[int]]): List of labels corresponding to each
                rule in the decision set. 

            rule_length (int): Maximum rule length.

            pruner (Callable, optional): Function/Object used to prune rules. 
                Defaults to None, in which case no pruning is performed.
            
        """
        '''
        if pruner is not None:
            assert issubclass(pruner, Pruner), \
                "Input pruner must be a valid instance of the Pruner object."
        self.pruner = pruner
        '''

        self.decision_set = None
        self.decision_set_labels = None
        self.max_rule_length = 0
        self.pruner = None
        
        
    def _fitting(
        self,
        X : NDArray,
        y : List[Set[int]] = None
    ) -> Tuple[List[List[Condition]], List[Set[int]]]:
        """
        Privately used, custom fitting function.
        Fits a decision set to an input dataset. 
        
        Args:
            X (np.ndarray): Input dataset.
            
            y (List[Set[int]], optional): Target labels. Defaults to None.
            
        returns:
            decision_set (List[Condition]): List of rules.
            
            decision_set_labels (List[int]): List of labels corresponding to each rule.
        """
        raise NotImplementedError('Method not implemented.')
    

    def prune(self, X : NDArray, y : List[Set[int]] = None):
        """
        Prunes the decision set using the pruner.
        
        Args:
            X (np.ndarray): Input dataset.
            
            y (List[Set[int]], optional): Target labels. Defaults to None.
        """
        if self.pruner is not None:
            if self.decision_set is None or self.decision_set_labels is None:
                raise ValueError('Decision set has not been fitted yet.')
            
            pass
    

    def trim(self):
        """
        Trims the rules in the decision set to remove any redundant conditions. 
        """
        if self.decision_set is None or self.decision_set_labels is None:
            raise ValueError('Decision set has not been fitted yet.')
        
        trimmed_set = []
        trimmed_labels = []
        for i, rule in enumerate(self.decision_set):
            trimmed_rule = []
            for j, condition in enumerate(rule):
                if np.abs(condition.threshold) < np.inf:
                    trimmed_rule.append(condition)
            if len(trimmed_rule) > 0:
                trimmed_set.append(trimmed_rule)
                trimmed_labels.append(self.decision_set_labels[i])
        
        self.decision_set = trimmed_set
        self.decision_set_labels = trimmed_labels
        
        
    def fit(self, X : NDArray, y : List[Set[int]] = None):
        """
        Public fit function. 
        Fits a decision set to an input dataset. 
        
        Args:
            X (np.ndarray): Input dataset.
            
            y (List[Set[int]], optional): Target labels. Defaults to None.
        """
        self.decision_set, self.decision_set_labels = self._fitting(X, y)
        self.prune(X, y)
        self.trim()
    
        
    def get_data_to_rules_assignment(self, X : NDArray) -> NDArray:
        """
        Finds data points of X covered by each rule in the decision set.
        
        Args:
            X (np.ndarray): Input dataset.
            
        Returns:
            assignment (np.ndarray): n x n_rules boolean matrix with entry (i,j) being True
                if point i is covered by rule j and False otherwise.
        """
        assignment = np.zeros((X.shape[0], len(self.decision_set)))
        for i, condition_list in enumerate(self.decision_set):
            data_points_satisfied = satisfies_conditions(X, condition_list)
            assignment[data_points_satisfied, i] = True
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
        assignment = labels_to_assignment(self.decision_set_labels, n_labels)
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
        
        data_to_rules_assignment = self.get_data_to_rules_assignment(X)
        
        labels = [set() for _ in range(len(X))]
        for i in range(len(self.decision_set)):
            #r_covers = np.where(data_to_rules_assignment[:,i])[0]
            r_covers = data_to_rules_assignment[:,i].nonzero()[0]
            for j in r_covers:
                if rule_labels:
                    labels[j].add(i)
                else:
                    labels[j] = labels[j].union(self.decision_set_labels[i])

        # Mark uncovered points with {-1}
        labels = [label if label else {-1} for label in labels]
        
        return labels


    def get_weighted_average_rule_length(self, X : NDArray) -> float:
        """
        Finds the weighted average length of the rules, which is adjusted by the number 
        data points which fall into each rule. 

        NOTE: If the decision set has been pruned this will automatically use the 
            pruned decision set.

        Args:
            X : Input dataset to predict with. 

        Returns:
            wad (float): Weighted average depth.
        """
        data_to_rules_assignment = self.get_data_to_rules_assignment(X)
        decision_set = self.decision_set

        wad = 0
        total_covers = 0
        for i, rule in enumerate(decision_set):
            #r_covers = np.where(data_to_rules_assignment[:,i])[0]
            r_covers = data_to_rules_assignment[:,i].nonzero()[0]
            total_covers += len(r_covers)
            if len(r_covers) != 0:
                wad += len(r_covers) * (len(rule))
            
        return wad/total_covers
        
    
    