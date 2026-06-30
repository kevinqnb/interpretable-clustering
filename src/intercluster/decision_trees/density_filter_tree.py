from sklearn.cluster import DBSCAN
from typing import List, Set, Callable
from numpy.typing import NDArray
from .decision_tree import DecisionTree
        
                    
####################################################################################################


class DensityFilterTree(DecisionTree):
    """
    A DecisionTree subclass that pre-processes input data using DBSCAN to remove outliers
    before training. The tree is trained only on inliers, but prediction still applies to
    the full dataset.
    
    This class acts exactly like DecisionTree but with density-based filtering applied
    during the fit() operation.
    """
    
    def __init__(
        self,
        eps: float = 0.5,
        min_samples: int = 5,
        criterion: str = 'entropy',
        max_leaf_nodes: int = None,
        max_depth: int = None,
        min_points_leaf: int = 1,
        random_state: int = None,
        selector: Callable = None,
    ):
        """
        Args:
            eps (float): The maximum distance between two samples for one to be considered
                as in the neighborhood of the other. Defaults to 0.5.
            
            min_samples (int): The number of samples in a neighborhood for a point to be 
                considered as a core point. Defaults to 5.
            
            criterion (str): The function to measure the quality of a split. Supported criteria
                are 'gini' for the Gini impurity and 'entropy' for the information gain.
                
            max_leaf_nodes (int, optional): Optional constraint for maximum number of leaf nodes. 
                Defaults to None.
                
            max_depth (int, optional): Optional constraint for maximum depth. 
                Defaults to None.
                
            min_points_leaf (int, optional): Optional constraint for the minimum number of points
                within a single leaf. Defaults to 1.

            random_state (int, optional): Seed used by the random number generator.
                Defaults to None.

            selector (Callable, optional): Function/Object used to prune branches of the tree.
                Defaults to None, in which case no pruning is performed.
        
        Attributes:
            root (Node): Root node of the tree.
        
            heap (heapq list): Maintains the heap structure of the tree.
            
            leaf_count (int): Number of leaves in the tree.
            
            node_count (int): Number of nodes in the tree.
                
            depth (int): The maximum depth of the tree.
        """
        self.eps = eps
        self.min_samples = min_samples
        
        super().__init__(
            criterion=criterion,
            max_leaf_nodes=max_leaf_nodes,
            max_depth=max_depth,
            min_points_leaf=min_points_leaf,
            random_state=random_state,
            selector=selector
        )
    
    
    def fit(
        self,
        X: NDArray,
        y: List[Set[int]] = None
    ):
        """
        Fits a Sklearn tree to a dataset X and labels y, with DBSCAN pre-processing to 
        remove outliers.
        
        First, DBSCAN is applied to X to identify outliers (points labeled as -1). 
        These outliers are removed from both X and y before training the tree.
        The tree is then fitted on the inlier data only.
        
        Args:
            X (np.ndarray): Input dataset.
            
            y (List[Set[int]], optional): Target labels. Defaults to None.
        """
        # Apply DBSCAN to identify outliers
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        dbscan_labels = dbscan.fit_predict(X)
        
        # Filter out outliers (points labeled as -1)
        inlier_mask = dbscan_labels != -1
        X_inliers = X[inlier_mask]
        
        # Filter labels if provided
        if y is not None:
            y_inliers = [y[i] for i in range(len(y)) if inlier_mask[i]]
        else:
            y_inliers = None
        
        # Call parent's fit() method with the filtered data
        super().fit(X_inliers, y_inliers)


####################################################################################################


