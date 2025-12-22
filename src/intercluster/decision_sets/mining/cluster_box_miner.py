import numpy as np
from scipy.stats import multivariate_normal
from sklearn.metrics.pairwise import pairwise_distances
from typing import List, Set, Tuple
from numpy.typing import NDArray
from intercluster import (
    Condition,
    LinearCondition,
    satisfies_conditions,
)
from .rule_miner import RuleMiner

####################################################################################################


class ClusterBoxMiner(RuleMiner):
    """
    Rule miner that uses the Pointwise algorithm to generate rules. Specifically, this will 
    expand randomly chosen rectangular regions around each data point until specified stopping 
    conditions are reached.
    """
    def __init__(
        self,
        samples : int = 10,
        prob_dim : float = 1/2,
    ):
        """
        Initialize the PointwiseMiner.

        Args:
            samples (int, optional): Number of samples to draw for each data point. 
                Defaults to 10.
            prob_dim (float, optional): Probability for a geometric distribution used to choose 
                the number of dimensions to use in each rule. Defaults to 1/2, in which case 
                the expected number of dimensions used is 2.
            lambd (float, optional): Parameter for exponential distribution.

        Attributes:
            decision_set (List[List[Condition]]): The mined decision set, where each rule is a list of conditions.
            decision_set_labels (List[Set[int]]): The labels corresponding to each rule.
        """
        if not isinstance(samples, int) or samples <= 0:
            raise ValueError("Number of samples must be a positive integer.")
        if not isinstance(prob_dim, float) or prob_dim < 0 or prob_dim > 1:
            raise ValueError("prob_dim must be a floating point number in [0, 1].")
        self.samples = samples
        self.prob_dim = prob_dim
        super().__init__()
    

    def fit(
            self,
            X : NDArray,
            y : List[Set[int]] = None,
        ) -> Tuple[List[List[Condition]], List[Set[int]]]:
        """
        Creates rules for the decision set by drawing boxes around dense sets of points 
        in randomly chosen dimensions. 

        Args:
            X (NDArray): Input dataset.
            y (List[Set[int]], optional): Dummy parameter for compatibility. Defaults to None.

        Returns:
            rules (List[List[Condition]]): List of rules, where each rule is a list of conditions.
            rule_labels (List[Set[int]]): List of labels corresponding to each rule.
        """
        n,d = X.shape
        var = np.trace(np.cov(X, rowvar=False))
        X_sorted = np.argsort(X, axis=0)
        X_pairwise = pairwise_distances(X) ** 2
        decision_set = []

        for _ in range(self.samples):
            for i in range(n):
                cov = (X - X[i,:]).T @ (X - X[i,:]) / (X.shape[0])
                # Randomly select features to create a box around the point
                n_features = min(np.random.geometric(self.prob_dim), d)
                features = np.random.choice(d, n_features, replace=False)
                satisfies = np.zeros((n, n_features), dtype=bool)
                satisfies[i, :] = True

                # Expand the box around the point until no more points can be added
                point_loc = np.where(X_sorted[:, features] == i)[0]
                lower_idx = np.copy(point_loc)
                upper_idx = np.copy(point_loc)

                # 2d array where row 0 indicates whether each dimension is moving its lower index,
                # and row 1 indicates whether each dimension is moving its upper index.
                is_moving = np.ones((2, n_features), dtype=bool)

                while np.any(is_moving):
                    # Sample from moving dimensions:
                    moving_indices = np.where(is_moving)
                    dim_to_move = np.random.choice(len(moving_indices[0]))
                    dim_type = moving_indices[0][dim_to_move]  # 0 for lower, 1 for upper
                    dim_idx = moving_indices[1][dim_to_move]                    

                    # Expand in the chosen dimension:
                    if dim_type == 0:
                        new_idx = lower_idx[dim_idx] - 1
                        if new_idx >= 0:
                            new_point = X_sorted[new_idx, features[dim_idx]]
                            satisfies[new_point, dim_idx] = True
                            lower_idx[dim_idx] -= 1

                            # Once a point has been satisfied in all dimensions, it is therefore
                            # covered by the rule. At this point, we attempt to add it.
                            if np.all(satisfies[new_point, :]):
                                prob_stop = multivariate_normal(mean=X[i,:], cov=cov).cdf(X[new_point, :])
                                if np.random.rand() > prob_stop:
                                    pass
                                else:
                                    # If the point is not added, we stop expanding the box
                                    is_moving[:] = False
                                    satisfies[new_point, dim_idx] = False
                                    lower_idx[dim_idx] += 1

                        else:
                            lower_idx[dim_idx] -= 1
                            is_moving[0, dim_idx] = False
                    else:
                        new_idx = upper_idx[dim_idx] + 1
                        if new_idx < n:
                            new_point = X_sorted[new_idx, features[dim_idx]]
                            satisfies[new_point, dim_idx] = True
                            upper_idx[dim_idx] += 1

                            # Once a point has been satisfied in all dimensions, it is therefore
                            # covered by the rule. At this point, we attempt to add it.
                            if np.all(satisfies[new_point, :]):
                                square_dist = X_pairwise[new_point, i]
                                prob_stop = self.expon_dist.cdf(square_dist / var)
                                if np.random.rand() > prob_stop:
                                    pass
                                else:
                                    # If the point is not added, we stop moving backwards 
                                    # in this dimension
                                    is_moving[:] = False
                                    satisfies[new_point, dim_idx] = False
                                    upper_idx[dim_idx] -= 1
                        
                        else:
                            upper_idx[dim_idx] += 1
                            is_moving[1, dim_idx] = False
                    
                # Add the conditions to the rule, and the rule to the decision set.
                # Subtract 1 from the lower indices, since the lower bounds will be strict 
                # greater-than indequalities (>)
                lower_idx -= 1
                rule = []
                for j, f in enumerate(features):
                    feature_vec = X_sorted[:, f]
                    lower_bound = (
                        X[feature_vec[lower_idx[j]], f]
                        if lower_idx[j] >= 0 else -np.inf
                    )
                    upper_bound = (
                        X[feature_vec[upper_idx[j]], f]
                        if upper_idx[j] < n else np.inf
                    )

                    condition1 = LinearCondition(
                        features=np.array([f]),
                        weights=np.array([1.0]),
                        threshold=lower_bound,
                        direction=1
                    )
                    condition2 = LinearCondition(
                        features=np.array([f]),
                        weights=np.array([1.0]),
                        threshold=upper_bound,
                        direction=-1
                    )
                    rule.append(condition1)
                    rule.append(condition2)


                decision_set.append(rule)


        self.decision_set = decision_set
        return self.decision_set, None


####################################################################################################