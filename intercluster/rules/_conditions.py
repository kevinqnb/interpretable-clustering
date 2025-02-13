import numpy as np
from numpy.typing import NDArray


class Condition():
    """
    Base class for a rule-based condition
    """
    def __init__(self):
        pass
    
    def evaluate(self, X : NDArray):
        """
        Evaluates the linear condition upon a given subset of data.
        
        Args:
            X (np.ndarray): Dataset for evaluation. May be either a single 
                data point or a 2d array.
        """
        pass
        


class LinearCondition(Condition):
    """
    Object for a linear splitting condition (either axis aligned)
    or oblique. 
    
    Args:
        features (np.ndarray): The chosen features to split on. 
        
        weights (np.ndarray): Weights for each of the splitting features.
        
        threshold (float): The threshold value to split on.
        
        direction (int): Specifies the direction for the inequality. 
            Use -1 for less than or equal (<=) or 1 for greater (>). 
            Defaults to None in which case a direction is not specified.
    """
    def __init__(
        self,
        features : NDArray,
        weights : NDArray,
        threshold : float,
        direction : int = None,
    ):
        self.features = features
        self.weights = weights
        self.threshold = threshold
        self.direction = direction
        
    
    def evaluate(self, X : NDArray):
        """
        Evaluates the linear condition upon a given subset of data.
        
        Args:
            X (np.ndarray): Dataset for evaluation. May be either a single 
                data point or a 2d array.
        """
        features_needed = np.max(self.features)
        if len(X.shape) < 2:
            if len(X) < features_needed:
                raise ValueError("Shape of data does not match the number of features required.")
        elif X.shape[1] < features_needed:
            raise ValueError("Shape of data does not match the number of features required.")
        
        evals = np.sign(np.dot(X[:,self.features], self.weights) - self.threshold)
        evals[evals == 0] = -1
        
        if self.direction is not None:
            return evals == self.direction
        
        return evals