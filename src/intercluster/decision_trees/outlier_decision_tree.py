import numpy as np
from typing import List, Set, Callable
from numpy.typing import NDArray
from .decision_tree import DecisionTree


class OutlierDecisionTree(DecisionTree):
    """
    Extends DecisionTree to exclude outlier points (those farthest from any cluster center)
    during tree construction, while still predicting their labels at inference time.
    """

    def __init__(
        self,
        cluster_centers: NDArray,
        p_outliers: float = 0.1,
        criterion: str = 'entropy',
        max_leaf_nodes: int = None,
        max_depth: int = None,
        min_points_leaf: int = 1,
        random_state: int = None,
        selector: Callable = None,
    ):
        """
        Args:
            cluster_centers (np.ndarray): Array of shape (k, d) with cluster center coordinates.

            p_outliers (float): Fraction of points with the largest distance to their nearest
                cluster center to exclude from tree construction. Must be in [0, 1).

            criterion (str): Split quality measure; 'gini' or 'entropy'.

            max_leaf_nodes (int, optional): Maximum number of leaf nodes. Defaults to None.

            max_depth (int, optional): Maximum tree depth. Defaults to None.

            min_points_leaf (int, optional): Minimum points per leaf. Defaults to 1.

            random_state (int, optional): Random seed. Defaults to None.

            selector (Callable, optional): Pruning function. Defaults to None.
        """
        if not (0.0 <= p_outliers < 1.0):
            raise ValueError("p_outliers must be in [0, 1).")

        self.cluster_centers = np.array(cluster_centers)
        self.p_outliers = p_outliers

        super().__init__(
            criterion=criterion,
            max_leaf_nodes=max_leaf_nodes,
            max_depth=max_depth,
            min_points_leaf=min_points_leaf,
            random_state=random_state,
            selector=selector,
        )

    def _outlier_mask(self, X: NDArray) -> NDArray:
        """Returns a boolean mask of shape (n,) that is True for inlier points."""
        distances = np.min(
            np.linalg.norm(X[:, np.newaxis, :] - self.cluster_centers[np.newaxis, :, :], axis=2),
            axis=1,
        )
        n_outliers = int(np.floor(self.p_outliers * len(X)))
        if n_outliers == 0:
            return np.ones(len(X), dtype=bool)
        threshold = np.sort(distances)[-n_outliers]
        # Mark the n_outliers points with the largest distances as outliers.
        # Break ties by preferring points with higher indices to be outliers.
        outlier_indices = np.argsort(distances)[-n_outliers:]
        mask = np.ones(len(X), dtype=bool)
        mask[outlier_indices] = False
        return mask

    def fit(
        self,
        X: NDArray,
        y: List[Set[int]] = None,
    ):
        """
        Fits the decision tree on inlier points only, then stores the full dataset
        so outlier predictions can be made at inference time.

        Args:
            X (np.ndarray): Input dataset of shape (n, d).

            y (List[Set[int]], optional): Target labels. Defaults to None.
        """
        inlier_mask = self._outlier_mask(X)

        X_inliers = X[inlier_mask]
        y_inliers = [y[i] for i in np.where(inlier_mask)[0]] if y is not None else None

        super().fit(X_inliers, y_inliers)
