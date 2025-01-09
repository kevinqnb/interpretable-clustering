from typing import List, Dict, Any, Tuple
from numpy.typing import NDArray
from ._tree import Tree
from ..utils import *
from .prune import *

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
        """
        self.feature_labels = feature_labels
        
        self.all_rules = None
        self.decision_set = None
        
        
    def _fitting(self, X : NDArray, y : NDArray = None) -> List[Any]:
        """
        Privately used, custom fitting function.
        Fits a decision set to an input dataset. 
        
        Args:
            X (np.ndarray): Input dataset.
            
            y (np.ndarray, optional): Target labels. Defaults to None.
            
        returns:
            decision_set (List[Any]): List of rules.
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
        self.decision_set = self._fitting(X, y)
        self.covers = self.get_covers(X)
        
        
    def get_covers(self, X : NDArray) -> Dict[int, List[int]]:
        """
        Finds data points of X covered by each rule in the decision set.
        
        Args:
            X (np.ndarray): Input dataset.
            
        Returns:
            covers (dict[rule, List[int]]): Dictionary with rules indices 
                as keys and list of data point indices as values.
        """
        raise NotImplementedError('Method not implemented.')
    
    
    def predict(self, X : NDArray) -> List[List[int]]:
        """
        Predicts the label(s) of each data point in X.
        
        Args:
            X (np.ndarray): Input dataset.
            
        Returns:
            labels (List[List[int]]): 2d list of predicted labels, with the internal list 
                at index i representing the group of decision rules which satisfy X[i,:].
        """
        
        set_covers = self.get_covers(X)
        
        labels = [[] for _ in range(len(X))]
        
        for i,r_covers in set_covers.items():
            for j in r_covers:
                labels[j].append(i)
        
        return labels