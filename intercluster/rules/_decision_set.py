import numpy as np
from typing import List, Set, Any, Tuple, Callable
from numpy.typing import NDArray
from intercluster.pruning import prune_with_grid_search
from ._conditions import Condition

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
            
            decision_set (List[Condition]): List of rules in the decision set.
            
            decision_set_labels (List[Set[int]]): List of labels corresponding to each
                rule in the decision set. 
                
            pruned_indices (np.ndarray): Indices for rules selected in the pruning process.
            
            pruned_status (bool): `True` if the pruning was successful and `False` otherwise. 
            
        """
        self.feature_labels = feature_labels
        self.decision_set = None
        self.decision_set_labels = None
        self.pruned_indices = None
        self.prune_status = False
        
        
    def _fitting(
        self,
        X : NDArray,
        y : List[Set[int]] = None
    ) -> Tuple[List[Condition], List[Set[int]]]:
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
        
        
    def fit(self, X : NDArray, y : List[Set[int]] = None):
        """
        Public fit function. 
        Fits a decision set to an input dataset. 
        
        Args:
            X (np.ndarray): Input dataset.
            
            y (List[Set[int]], optional): Target labels. Defaults to None.
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
    
    
    def predict(self, X : NDArray, rule_labels : bool = True) -> List[Set[int]]:
        """
        Predicts the label(s) of each data point in X.
        
        Args:
            X (np.ndarray): Input dataset.
            
            rule_labels (bool, optional): If true, gives labels based soley upon 
                rule membership. That is, each rule is given a unique label. 
                Otherwise, returns the orignal predictions from the fitted rule models -- 
                whatever label is given to the rule. Defaults to True.
            
        Returns:
            labels (List[Set[int]]): 2d list of predicted labels, with the internal list 
                at index i representing the group of decision rules which satisfy X[i,:].
        """
        
        data_to_rules_assignment = self.get_data_to_rules_assignment(X)
        
        labels = [set() for _ in range(len(X))]
        for i in range(len(self.decision_set)):
            r_covers = np.where(data_to_rules_assignment[:,i])[0]
            for j in r_covers:
                if rule_labels:
                    labels[j].add(i)
                else:
                    labels[j] = labels[j].union(self.decision_set_labels[i])
        
        return labels
    
    
    def prune(
        self,
        n_rules : int,
        frac_cover : float,
        n_clusters : int,
        X : NDArray,
        y : List[Set[int]],
        objective : Callable, 
        lambda_search_range : NDArray = np.linspace(0,1,10),
        cpu_count : int = 1
    ):
        """
        Prunes the decision set using a distorted greedy algorithm for a submodular 
        coverage objective. 
        
        Args:
            n_rules (int): The number of rules to select.
            
            frac_cover (float): Required threshold for coverage of data points.
            
            n_clusters (int) : The desired number of clusters. 
            
            X (np.ndarray): Input dataset.
            
            y (List[Set[int]]): Labeling of the data points.
            
            objective (Callable): A function that takes an assignment matrix of data points 
                to clusters and returns a score.
            
            lambda_search_range (np.ndarray, optional): A range of lambda values to search over. 
                Defaults to np.linspace(0,1,10).

            cpu_count (int): Number of cores to use for a parallelized grid search.
        """
        data_to_rules_assignment = self.get_data_to_rules_assignment(X)
        
        selected_rules = prune_with_grid_search(
            n_rules = n_rules,
            frac_cover = frac_cover,
            n_clusters = n_clusters,
            data_labels = y,
            rule_labels = self.decision_set_labels,
            data_to_rules_assignment = data_to_rules_assignment,
            objective = objective,
            lambda_search_range = lambda_search_range,
            cpu_count = cpu_count
        )
        
        if selected_rules is None:
            self.prune_status = False
        else:
            self.prune_status = True
            self.pruned_indices = selected_rules
        
        
    def pruned_predict(self, X : NDArray, rule_labels : bool = True) -> List[Set[int]]:
        """
        Predicts the label(s) of each data point in X.
        
        Args:
            X (np.ndarray): Input dataset.
            
            rule_labels (bool, optional): If true, gives labels based soley upon 
                rule membership. That is, each rule is given a unique label. 
                Otherwise, returns the orignal predictions from the fitted rule models -- 
                whatever label is given to the rule. Defaults to True.
            
        Returns:
            labels (List[Set[int]]): 2d list of predicted labels, with the internal set 
                at index i representing the group of decision rules which satisfy X[i,:].
        """
        if self.pruned_indices is None:
            raise ValueError('Decision set has not been pruned. If prune() was called, this is '
                             'likely because coverage requirements were not met.')
        
        data_to_rules_assignment = self.get_data_to_rules_assignment(X)
        pruned_data_to_rules_assignment = data_to_rules_assignment[:,self.pruned_indices]
        pruned_decision_set = [self.decision_set[i] for i in self.pruned_indices]
        pruned_decision_set_labels = [self.decision_set_labels[i] for i in self.pruned_indices]
        #pruned_feature_labels = [self.feature_labels[i] for i in self.pruned_indices]
        
        labels = [set() for _ in range(len(X))]
        for i in range(len(pruned_decision_set)):
            r_covers = np.where(pruned_data_to_rules_assignment[:,i])[0]
            for j in r_covers:
                if rule_labels:
                    labels[j].add(i)
                else:
                    labels[j] = labels[j].union(pruned_decision_set_labels[i])
        return labels
        
    
    