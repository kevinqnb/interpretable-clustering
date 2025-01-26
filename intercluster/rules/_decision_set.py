import numpy as np
from typing import List, Dict, Any, Tuple, Callable
from numpy.typing import NDArray
from intercluster.pruning import prune_with_grid_search

class DecisionSet:
    """
    Base class for a decision set.
    """
    def __init__(
        self,
        feature_labels : List[str] = None
    ):
        
        """
        Args:                
            feature_labels (List[str]): Iterable object with strings representing feature names. 
            
        Attributes:
            feature_labels (List[str]): Iterable object with strings representing feature names. 
            
            decision_set (List[Rule]): List of rules in the decision set.
            
            decision_set_labels (List[int]): List of labels corresponding to each
                rule in the decision set.
        """
        self.feature_labels = feature_labels
        self.decision_set = None
        self.decision_set_labels = None
        self.pruned_indices = None
        self.pruned_decision_set = None
        self.pruned_decision_set_labels = None
        
        
    def _fitting(self, X : NDArray, y : NDArray = None) -> Tuple[List[Any], List[int]]:
        """
        Privately used, custom fitting function.
        Fits a decision set to an input dataset. 
        
        Args:
            X (np.ndarray): Input dataset.
            
            y (np.ndarray, optional): Target labels. Defaults to None.
            
        returns:
            decision_set (List[Any]): List of rules.
            
            decision_set_labels (List[int]): List of labels corresponding to each rule.
        """
        raise NotImplementedError('Method not implemented.')
        
        
    def fit(self, X : NDArray, y : NDArray = None):
        """
        Public fit function. 
        Fits a decision set to an input dataset. 
        
        Args:
            X (np.ndarray): Input dataset.
            
            y (np.ndarray, optional): Target labels. Defaults to None.
        """
        self.decision_set, self.decision_set_labels = self._fitting(X, y)
    
        
    def get_data_to_rules_assignment(self, X : NDArray) -> NDArray:
        """
        Finds data points of X covered by each rule in the decision set.
        
        Args:
            X (np.ndarray): Input dataset.
            
        Returns:
            assignment (np.ndarray): n x n_rules boolean matrix with entry (i,j) being True
                if point i is covered by rule j and False otherwise.
        """
        raise NotImplementedError('Method not implemented.')
    
    
    def predict(self, X : NDArray, rule_labels : bool = True) -> List[List[int]]:
        """
        Predicts the label(s) of each data point in X.
        
        Args:
            X (np.ndarray): Input dataset.
            
            rule_labels (bool, optional): If true, gives labels based soley upon 
                rule membership. That is, each rule is given a unique label. 
                Otherwise, returns the orignal predictions from the fitted rule models -- 
                whatever label is given to the rule. Defaults to True.
            
        Returns:
            labels (List[List[int]]): 2d list of predicted labels, with the internal list 
                at index i representing the group of decision rules which satisfy X[i,:].
        """
        
        data_to_rules_assignment = self.get_data_to_rules_assignment(X)
        
        labels = [[] for _ in range(len(X))]
        for i in range(len(self.decision_set)):
            r_covers = np.where(data_to_rules_assignment[:,i])[0]
            for j in r_covers:
                if rule_labels:
                    labels[j] += [i]
                else:
                    labels[j] += self.decision_set_labels[i]
        
        return labels
    
    
    def prune(
        self,
        q : int,
        k : int,
        X : NDArray,
        y : List[List[int]],
        objective : Callable, 
        lambda_search_range : NDArray = np.linspace(0,1,10)
    ):
        """
        Prunes the decision set using a distorted greedy algorithm for a submodular 
        coverage objective. 
        
        Args:
            q (int): The number of rules to select.
            
            k (int) : The desired number of clusters. 
            
            X (np.ndarray): Input dataset.
            
            y (List[List[int]]): Labeling of the data points.
            
            objective (Callable): A function that takes an assignment matrix of data points 
                to clusters and returns a score.
            
            lambda_search_range (np.ndarray, optional): A range of lambda values to search over. 
                Defaults to np.linspace(0,1,10).
        """
        data_to_rules_assignment = self.get_data_to_rules_assignment(X)
        
        selected_rules = prune_with_grid_search(
            q = q,
            k = k,
            data_labels = y,
            rule_labels = self.decision_set_labels,
            data_to_rules_assignment = data_to_rules_assignment,
            objective = objective,
            search_range = lambda_search_range,
        )
        
        self.pruned_indices = selected_rules
        
        
    def pruned_predict(self, X : NDArray, rule_labels : bool = True) -> List[List[int]]:
        """
        Predicts the label(s) of each data point in X.
        
        Args:
            X (np.ndarray): Input dataset.
            
            rule_labels (bool, optional): If true, gives labels based soley upon 
                rule membership. That is, each rule is given a unique label. 
                Otherwise, returns the orignal predictions from the fitted rule models -- 
                whatever label is given to the rule. Defaults to True.
            
        Returns:
            labels (List[List[int]]): 2d list of predicted labels, with the internal list 
                at index i representing the group of decision rules which satisfy X[i,:].
        """
        if self.pruned_indices is None:
            raise ValueError('Decision set has not been pruned!')
        
        data_to_rules_assignment = self.get_data_to_rules_assignment(X)
        pruned_data_to_rules_assignment = data_to_rules_assignment[:,self.pruned_indices]
        pruned_decision_set = [self.decision_set[i] for i in self.pruned_indices]
        pruned_decision_set_labels = [self.decision_set_labels[i] for i in self.pruned_indices]
        #pruned_feature_labels = [self.feature_labels[i] for i in self.pruned_indices]
        
        labels = [[] for _ in range(len(X))]
        for i in range(len(pruned_decision_set)):
            r_covers = np.where(pruned_data_to_rules_assignment[:,i])[0]
            for j in r_covers:
                if rule_labels:
                    labels[j] += [i]
                else:
                    labels[j] += pruned_decision_set_labels[i]
        return labels
        
    
    