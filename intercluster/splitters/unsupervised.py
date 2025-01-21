import numpy as np
import numpy.typing as npt
from typing import Tuple
from ._splitter import Splitter

class UnsupervisedSplitter(Splitter):
    """
    Splits leaf nodes in order to minimize distances to a set of input centers.
    """
    def __init__(
        self,
        norm : int = 2
    ):
        """
        Args:
            centers (npt.NDArray): Array of centroid representatives.
            
            norm (int, optional): Norm to use for computing distances. 
                Takes values 1 or 2. Defaults to 2.
                
            min_points_leaf (int, optional): Minimum number of points in a leaf.
        """
        self.norm = norm
        super().__init__()
        
    def cost(
        self,
        indices : npt.NDArray
    ) -> float:
        """
        Given a set of points X, computes the cost as the sum of distances to 
        the closest center.
        
        Args:
            X (npt.NDArray): Array of points to compute cost with.
            
            y (npt.NDArray, optional): Array of labels. Dummy variable, not used 
                for this class. Defaults to None.
                
            indices (npt.NDArray, optional): Indices of points to compute cost with.
                
        Returns:
            (float): cost of the given data.
        """
        X_ = self.X[indices, :]
        
        if len(indices) == 0:
            return 0
        else:
            if self.norm == 2:
                mu = np.mean(X_, axis = 0)
                cost = np.sum(np.linalg.norm(X_ - mu, axis = 1)**2)
                
            elif self.norm == 1:
                eta = np.median(X_, axis = 0)
                cost = np.sum(np.abs(X_ - eta))
                
            return cost
        
        
    def split(
        self,
        indices : npt.NDArray = None
    ) -> Tuple[float, Tuple[npt.NDArray, npt.NDArray, float]]:
        """
        Given a set of points X, computes the best split of the data.
        
        The following optimized version is attributed to 
        [Dasgupta, Frost, Moshkovitz, Rashtchian '20] in their paper
        'Explainable k-Means and k-Medians Clustering' 
        
        Args:
            X (npt.NDArray): Array of points to compute split with.
            
            y (npt.NDArray, optional): Array of labels. Dummy variable, not used 
                for this class. Defaults to None.
                
            indices (npt.NDArray, optional): Indices of points to compute split with.
                
        Returns:
            split_info ((np.ndarray, np.ndarray, float)): Features, weights,
                and threshold of the split.
        """
        if self.norm == 1:
            # NOTE: still need to optimize for the norm = 1 case.
            return super().split(indices)
        else:
            """
            The following optimized version is attributed to 
            [Dasgupta, Frost, Moshkovitz, Rashtchian '20] in their paper
            'Explainable k-Means and k-Medians Clustering' 
            """
            X_ = self.X[indices, :]
            n, d = X_.shape
            parent_cost = self.cost(indices)
            
            u = np.linalg.norm(X_)**2
            best_splits = []
            best_gain_val = -np.inf
            for i in range(X_.shape[1]):
                s = np.zeros(d)
                r = np.sum(X_, axis = 0)
                order = np.argsort(X_[:, i])
                
                for j, idx in enumerate(order[:-1]):
                    threshold = X_[idx, i]
                    split = ([i], [1], threshold)
                    s = s + X_[idx, :]
                    r = r - X_[idx, :]
                    split_cost = u - np.sum(s**2)/(j + 1) - np.sum(r**2)/(n - j - 1)
                    gain_val = parent_cost - split_cost
                    
                    if gain_val > best_gain_val:
                        best_gain_val = gain_val
                        best_splits = [split]
                        
                    elif gain_val == best_gain_val:
                        best_splits.append(split)
            
            # Randomly break ties if necessary:
            best_split = best_splits[np.random.randint(len(best_splits))]
            return best_gain_val, best_split