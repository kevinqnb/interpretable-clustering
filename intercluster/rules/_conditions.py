import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from numpy.typing import NDArray
from typing import Callable, List


class Condition():
    """
    Base class for a rule-based condition
    """
    def __init__(self):
        pass
    
    def evaluate(self, X : NDArray) -> NDArray:
        """
        Evaluates the condition upon a given subset of data.
        
        Args:
            X (np.ndarray): Size n x d dataset for evaluation. May be either a single 
                data point or a 2d array.
        
        Returns:
            (np.ndarray): Length n boolean array with entry i being `True` if point i satisfies
                the condition, and `False` otherwise.
        """
        pass
        


class LinearCondition(Condition):
    """
    Object for a linear splitting condition, either axis aligned
    or oblique. 
    
    Args:
        features (np.ndarray): The chosen features to split on. 
        
        weights (np.ndarray): Weights for each of the splitting features.
        
        threshold (float): The threshold value to split on.
        
        direction (int): Specifies the direction for the inequality. 
            Use -1 for less than or equal (<=) or 1 for greater (>). 
            Defaults to -1.
    """
    def __init__(
        self,
        features : NDArray,
        weights : NDArray,
        threshold : float,
        direction : int = -1,
    ):
        self.features = features
        self.weights = weights
        self.threshold = threshold
        self.set_direction(direction)
        
        
    def set_direction(self, direction : int):
        """
        Sets the direction for the inequality.
        
        direction (int): Specifies the direction for the inequality. 
            Use -1 for less than or equal (<=) or 1 for greater (>). 
        """
        if direction not in {1, -1}:
            raise ValueError("Invalid inequality direction, must be -1 (<=) or 1 (>).")
        self.direction = direction
        
    
    def evaluate(self, X : NDArray) -> NDArray:
        """
        Evaluates the linear condition upon a given subset of data.
        
        Args:
            X (np.ndarray): Size n x d dataset for evaluation. May be either a single 
                data point or a 2d array.
                
        Returns:
            (np.ndarray): Length n boolean array with entry i being `True` if point i satisfies
                the condition, and `False` otherwise.
        """        
        features_needed = np.max(self.features)
        if len(X.shape) < 2:
            raise ValueError("Data must be two dimensional.")
            
        if X.shape[1] < features_needed:
            raise ValueError("Shape of data does not match the number of features required.")
        
        evals = np.sign(np.dot(X[:,self.features], self.weights) - self.threshold)
        evals[evals == 0] = -1
        
        if self.direction is not None:
            return evals == self.direction
        
        return evals
    

    def display(
            self,
            feature_labels : List[str] = None,
            scaler : Callable = None,
            newline : bool = True
        ) -> str:
        """
        Displays the condition by returning a string representation.

        Args:
            feature_labels (List[str], optional): List of feature labels used for display.
                The feature at index i should correspond to feature i in the dataset.  
                Defaults to None, in which case conditions will be plotted as is.

            scaler (Callable): Sklearn data scaler, which will be used to convert
                thresholds, weights back to their unscaled versions (better interpretability).
                This current supports the StandardScaler or the MinMaxScaler. Defaults 
                to None which leaves values as is.

            newline (bool): Decides whether to add a line break between each summand in 
                in the condition.

        Returns:
            (str): String representation for the condition. 
        """
        est_max_features = np.max(self.features) + 1
        if feature_labels is None:
            feature_labels = [rf"$x_{i}$" for i in range(est_max_features)]

        elif len(feature_labels) < est_max_features:
            raise ValueError(
                "Input feature labels must have as least as many features as the condition's "
                "maximum feature index."
            )
        
        # Convert back to normal scaling, if applicable:
        features = self.features
        weights = self.weights
        threshold = self.threshold
        direction = self.direction

        if isinstance(scaler, StandardScaler):
            scaled_weights = np.zeros(len(weights))
            for i,feat in enumerate(features):
                w = weights[i]
                mu = scaler.mean_[feat]
                std = scaler.scale_[feat]
                scaled_weights[i] = w/std
                threshold += w * mu / std

            if len(features) == 1:
                threshold /= scaled_weights[0]
                if np.sign(scaled_weights[0]) < 0:
                    direction *= -1
                scaled_weights[0] = 1

            weights = scaled_weights

        elif isinstance(scaler, MinMaxScaler):
            scaled_weights = np.zeros(len(weights))
            for i,feat in enumerate(features):
                w = weights[i]
                scale = scaler.scale_[feat]
                minf = scaler.min_[feat]
                scaled_weights[i] = w*scale
                threshold += -1*(w * minf)

            if len(features) == 1:
                threshold /= scaled_weights[0]
                if np.sign(scaled_weights[0]) < 0:
                    direction *= -1
                scaled_weights[0] = 1

            weights = scaled_weights

        condition_str = ""
        escape = "\n" if (len(features) > 1 and newline) else " "
        for i,feat in enumerate(features):
            w = np.round(weights[i], 3)
            addit = r" $+$" if i < len(features) - 1 else ""
            if w != 1:
                condition_str += str(w) + r"$\cdot$" + feature_labels[feat] + addit + escape
            else:
                condition_str += feature_labels[feat] + addit + escape

    
        if self.direction == -1:
            condition_str += r"$\leq$ "
        else:
            condition_str += r"$>$ "

        condition_str += str(np.round(threshold, 3))

        return condition_str

        



