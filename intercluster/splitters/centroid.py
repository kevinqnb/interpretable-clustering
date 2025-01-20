import numpy as np
import numpy.typing as npt
from ._splitter import AxisAlignedSplitter
from ..utils import center_dists

class CentroidSplitter(AxisAlignedSplitter):
    """
    Splits leaf nodes in order to minimize distances to a set of input centers.
    """
    def __init__(
        self,
        centers : npt.NDArray, 
        norm : int = 2,
        min_points_leaf : int = 1
    ):
        """
        Args:
            centers (npt.NDArray): Array of centroid representatives.
            
            norm (int, optional): Norm to use for computing distances. 
                Takes values 1 or 2. Defaults to 2.
                
            min_points_leaf (int, optional): Minimum number of points in a leaf.
        """
        self.centers = centers
        self.norm = norm
        super().__init__(min_points_leaf = min_points_leaf)
        
    def fit(
        self,
        X : npt.NDArray,
        y : npt.NDArray = None
    ):
        """
        Fits the splitter to a dataset X. 
        
        Args:
            X (npt.NDArray): Input dataset.
            
            y (npt.NDArray, optional): Target labels. Defaults to None.
        """
        self.X = X
        self.y = y
        self.center_dists = center_dists(X, self.centers, self.norm)
        
    def cost(
        self,
        indices : npt.NDArray
    ) -> float:
        """
        Given a set of points X, computes the cost as the sum of distances to 
        the closest center.
        
        Args:                
            indices (npt.NDArray, optional): Indices of points to compute cost with.
                
        Returns:
            (float): cost of the given data.
        """
        if len(indices) == 0:
            return 0
        else:
            dists_ = self.center_dists[indices,:]
            sum_array = np.sum(dists_, axis = 0)
            return np.min(sum_array)