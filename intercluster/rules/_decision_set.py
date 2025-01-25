from typing import List, Dict, Any, Tuple
from numpy.typing import NDArray

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
        self.covers = self.get_covers(X)
        
        
    def get_covers(self, X : NDArray) -> Dict[int, List[int]]:
        """
        Finds data points of X covered by each rule in the decision set.
        
        Args:
            X (np.ndarray): Input dataset.
            
        Returns:
            covers (dict[int, List[int]]): Dictionary with rules indices 
                as keys and list of data point indices as values.
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
        
        set_covers = self.get_covers(X)
        
        labels = [[] for _ in range(len(X))]
        
        for i,r_covers in set_covers.items():
            for j in r_covers:
                if rule_labels:
                    labels[j] += [i]
                else:
                    labels[j] += self.decision_set_labels[i]
        
        return labels