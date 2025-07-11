import numpy as np
from typing import List, Set, Any, Tuple, Callable
from numpy.typing import NDArray
from intercluster import can_flatten, flatten_labels, satisfies_conditions
from intercluster import Condition, LinearCondition
from .decision_set import DecisionSet

class PointSet(DecisionSet):
    """
    Collection of rules drawn as random boxes around individual points in the dataset.
    """
    def __init__(
        self,
        n_rules : int = 1,
        n_features : int = 2,
        #step_prob : float = 0.5,
        epsilon : float = 1
    ):
        """
        Args:
            n_rules (int): Number of rules to create around each individual point.
            
            n_features (int): Number of randomly chosen features to use for each rule.
        """
        super().__init__()
        self.n_rules = n_rules
        self.n_features = n_features
        #self.step_prob = step_prob
        self.epsilon = epsilon


    def _fitting(
        self,
        X : NDArray,
        y : List[Set[int]],
    ) -> Tuple[List[List[Condition]], List[Set[int]]]:
        """
        Fits a point set to an input dataset by creating rules around individual points.
        
        Args:
            X (np.ndarray): Input dataset.
            
            y (List[Set[int]], optional): Target labels.
            
        Returns:
            decision_set (List[Condition]): List of rules.
            
            decision_set_labels (List[int]): List of labels corresponding to each rule.
        """
        if not can_flatten(y):
            raise ValueError("Each data point must have exactly one label.")
        y_ = flatten_labels(y)
        
        n,d = X.shape
        #X_sorted = np.argsort(X, axis=0)

        decision_set = []
        decision_set_labels = []
        for i in range(n):
            point = X[i]
            for _ in range(self.n_rules):
                rule = []
                # Randomly select features to create a box around the point
                features = np.random.choice(d, self.n_features, replace=False)
                for f in features:
                    '''
                    feature_vec = X_sorted[:, f]
                    point_loc = np.where(feature_vec == i)[0][0]
                    steps_backward = np.random.geometric(self.step_prob)
                    steps_forward = np.random.geometric(self.step_prob)
                    lower_bound = (
                        X[feature_vec[point_loc - steps_backward], f]
                        if point_loc - steps_backward >= 0 else -np.inf
                    )
                    upper_bound = (
                        X[feature_vec[point_loc + steps_forward], f]
                        if point_loc + steps_forward < n else np.inf
                    )
                    '''
                    val = X[i,f]
                    condition1 = LinearCondition(
                        features=np.array([f]),
                        weights=np.array([1.0]),
                        #threshold=lower_bound,
                        threshold=val - self.epsilon,
                        direction=1
                    )
                    condition2 = LinearCondition(
                        features=np.array([f]),
                        weights=np.array([1.0]),
                        #threshold=upper_bound,
                        threshold=val + self.epsilon,
                        direction=-1
                    )
                    rule.append(condition1)
                    rule.append(condition2)

                decision_set.append(rule)
                decision_set_labels.append({y_[i]})

        return decision_set, decision_set_labels


    def get_data_to_rules_assignment(self, X : NDArray) -> NDArray:
        """
        Finds data points of X covered by each rule in the decision set.
        
        Args:
            X (np.ndarray): Input dataset.
            
        Returns:
            assignment (np.ndarray): n x n_rules boolean matrix with entry (i,j) being True
                if point i is covered by rule j and False otherwise.
        """
        assignment = np.zeros((X.shape[0], len(self.decision_set)))
        for i, condition_list in enumerate(self.decision_set):
            data_points_satisfied = satisfies_conditions(X, condition_list)
            assignment[data_points_satisfied, i] = True
        return assignment