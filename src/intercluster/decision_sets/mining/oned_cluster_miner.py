import numpy as np
import pandas as pd
from typing import List, Set, Tuple
from numpy.typing import NDArray
from intercluster import (
    Condition,
    Rule,
    LinearCondition,
    oned_cluster,
)

from .rule_miner import RuleMiner

####################################################################################################


class ClusterMiner(RuleMiner):
    """
    Rule miner that uses clustering to generate rules.

    Args:
        cluster_cost (float, optional): Cost associated with adding a new cluster. 
            Must be between 0.0 and 1.0. Defaults to 0.0.
        method (str, optional): Clustering method to use. Options are "kmeans" or "kmedians".
            Defaults to "kmeans".

    Attrs:
        decision_set (List[Rule]): The mined decision set, 
            where each rule is a list of conditions.
        bin_df (pd.DataFrame): The binned version of the input dataset used for mining rules.
    """
    def __init__(self, cluster_cost : float = 0.0, method = "kmeans"):
        """
        Initialize the ClusterMiner.

        Args:
            cluster_cost (float, optional): Cost parameter for clustering. Defaults to 0.05.
        """
        if not isinstance(cluster_cost, float) or cluster_cost < 0 or cluster_cost > 1:
            raise ValueError("cluster_cost must be a floating point number in [0, 1].")
        self.cluster_cost = cluster_cost

        if method not in ["kmeans", "kmedians"]:
            raise ValueError("Unsupported clustering method. Choose 'kmeans' or 'kmedians'.")
        self.method = method
        super().__init__()


    def fit(
            self,
            X : NDArray,
            y : List[Set[int]] = None
        ) -> Tuple[List[Rule], List[Set[int]]]:
        """
        Fit the rule mining algorithm to the input dataset.

        Args:
            X (pd.DataFrame): Input dataset.
            y (List[Set[int]], optional): Dummy parameter for compatibility. Defaults to None.

        Returns:
            rules (List[Rule]): List of rules.
            rule_labels (List[Set[int]]): None, dummy variable.
        """
        assert X.ndim == 2, "Input data X must be a 2D array."
        n,d = X.shape

        self.decision_set = []
        bin_df = []
        for feature in range(d):
            cluster_labels = oned_cluster(
                X[:, feature],
                cluster_cost=self.cluster_cost,
                method=self.method
            )

            mapping = {}
            for cluster in np.unique(cluster_labels):
                cluster_points = X[:, feature][cluster_labels == cluster]
                if len(cluster_points) == 0:
                    continue
                else:
                    lower_bound = np.min(cluster_points)
                    upper_bound = np.max(cluster_points)

                    conditions = []
                    condition1 = LinearCondition(
                        features=np.array([feature]),
                        weights=np.array([1.0]),
                        threshold=lower_bound,
                        direction=1
                    )
                    condition2 = LinearCondition(
                        features=np.array([feature]),
                        weights=np.array([1.0]),
                        threshold=upper_bound,
                        direction=-1
                    )
                    conditions.append(condition1)
                    conditions.append(condition2)
                    self.decision_set.append(Rule(conditions))
                    mapping[cluster] = f"({lower_bound}, {upper_bound}]"

            bin_df.append([mapping[cluster] for cluster in cluster_labels])

        self.bin_df = pd.DataFrame(bin_df).T
        return self.decision_set, None


####################################################################################################