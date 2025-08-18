import numpy as np
from typing import List, Set, Tuple
from numpy.typing import NDArray
from intercluster import (
    Condition,
    LinearCondition,
    satisfies_conditions,
    can_flatten,
    labels_to_assignment,
    unique_labels
)
from intercluster.pruning import CoverageMistakePruner
from .decision_set import DecisionSet


####################################################################################################


class DSCluster(DecisionSet):
    """
    Collection of rules drawn as boxes (rules) around collections of points in the dataset.
    """
    def __init__(
        self,
        lambd : float,
        n_rules : int,
        n_features : int,
        rules_per_point : int = 1
    ):
        """
        Args:
            lambd (float): Penalization factor for mistakes. Larger values penalize mistakes 
                more heavily, resulting in rules which are more accurate, but may cover 
                fewer points.

            n_rules (int): Number of rules to use in the decision set.
            
            n_features (int): Number of randomly chosen features to use for each rule.

            rules_per_point (int): Number of random rules to create for each point in the dataset.
        """
        super().__init__()
        self.lambd = lambd
        self.n_rules = n_rules
        self.n_features = n_features
        self.rules_per_point = rules_per_point
        self.pruner = CoverageMistakePruner(n_rules=n_rules, lambda_val=lambd)

        '''
        if self.pruner is not None:
            supported_pruners = ['CoverageMistakePruner']
            if self.pruner.__name__ not in supported_pruners:
                raise ValueError(
                    f"Pruner {pruner.__name__} is not supported. "
                    "Supported pruners are: {supported_pruners}"
                )
        '''
    

    def generate_rules(self, X : NDArray, y : List[Set[int]]) -> List[List[Condition]]:
        """
        Creates rules for the decision set by drawing boxes around dense sets of points 
        in randomly chosen dimensions. 

        Args:
            X (np.ndarray): Input dataset of size n x d.
            y (List[Set[int]]): Labels for each point in the dataset.
        Returns:
            decision_set (List[Condition]): List of rules created.
        """        
        n,d = X.shape
        X_sorted = np.argsort(X, axis=0)
        decision_set = []
        decision_set_labels = []
        # Assuming for now that outliers are labeled as {-1} or set()
        non_outliers = [i for i, label in enumerate(y) if label != {-1} and label != set()]
        for i in non_outliers:
            for _ in range(self.rules_per_point):
                # Randomly select features to create a box around the point
                features = np.random.choice(d, self.n_features, replace=False)
                satisfies = np.zeros((n, self.n_features), dtype=bool)
                satisfies[i, :] = True

                # Expand the box around the point until no more points can be added
                point_loc = np.where(X_sorted[:, features] == i)[0]
                lower_idx = np.copy(point_loc)
                lower_idx_is_moving = np.ones(self.n_features, dtype=bool)
                upper_idx = np.copy(point_loc)
                upper_idx_is_moving = np.ones(self.n_features, dtype=bool)
                while (np.any(lower_idx_is_moving) or np.any(upper_idx_is_moving)):
                    # Attempt to expand the box in each feature dimension
                    for j, f in enumerate(features):
                        feature_vec = X_sorted[:, f]

                        # Attempt to take a step backwards
                        if lower_idx_is_moving[j]:
                            new_idx = lower_idx[j] - 1
                            if new_idx >= 0:
                                new_point = feature_vec[new_idx]
                                satisfies[new_point, j] = True
                                lower_idx[j] -= 1
                                
                                # Once a point has been satisfied in all dimensions, it is therefore
                                # covered by the rule. At this point, we attempt to add it.
                                if np.all(satisfies[new_point, :]):
                                    if y[i] == y[new_point]:
                                        # If the label of the new point matches the label of the 
                                        # current point, we automatically add it.
                                        pass
                                    elif np.random.rand() < 2 ** (-self.lambd):
                                        # If the label does not match, we only add it with a 
                                        # probability of 2 ** (-lambd)
                                        pass
                                    else:
                                        # If the point is not added, we stop moving backwards 
                                        # in this dimension
                                        lower_idx_is_moving[j] = False
                                        satisfies[new_point, j] = False
                                        lower_idx[j] += 1

                            else:
                                lower_idx[j] -= 1
                                lower_idx_is_moving[j] = False


                        # Attempt to take a step forwards
                        if upper_idx_is_moving[j]:
                            new_idx = upper_idx[j] + 1
                            if new_idx < n:
                                new_point = feature_vec[new_idx]
                                satisfies[new_point, j] = True
                                upper_idx[j] += 1
                                
                                # Once a point has been satisfied in all dimensions, it is therefore
                                # covered by the rule. At this point, we attempt to add it.
                                if np.all(satisfies[new_point, :]):
                                    if y[i] == y[new_point]:
                                        # If the label of the new point matches the label of the 
                                        # current point, we automatically add it.
                                        pass
                                    elif np.random.rand() < 2 ** (-self.lambd):
                                        # If the label does not match, we only add it with a 
                                        # probability of 2 ** (-lambd)
                                        pass
                                    else:
                                        # If the point is not added, we stop moving backwards 
                                        # in this dimension
                                        upper_idx_is_moving[j] = False
                                        satisfies[new_point, j] = False
                                        upper_idx[j] -= 1
                            
                            else:
                                upper_idx[j] += 1
                                upper_idx_is_moving[j] = False
                    


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
                decision_set_labels.append(y[i])

        return decision_set, decision_set_labels


    def _fitting(
        self,
        X : NDArray,
        y : List[Set[int]] = None,
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
        n,d = X.shape
        if y is None:
            raise ValueError("Reference labels must be provided for fitting.")
        if not can_flatten(y):
            raise ValueError("Each data point must have exactly one label.")
        
        decision_set, decision_set_labels = self.generate_rules(X, y)
        return decision_set, decision_set_labels
    

    def prune(
            self,
            X : NDArray,
            y : List[Set[int]]
        ) -> List[List[Condition]]:
        """
        Prunes the decision set by removing rules that do not cover any points in the dataset.

        Args:
            X (np.ndarray): Input dataset.
            y (List[Set[int]]): Target labels.
        """
        n_labels = len(unique_labels(y, ignore ={-1}))
        data_to_cluster_assignment = labels_to_assignment(
            y, n_labels = n_labels, ignore = {-1}
        )
        rule_to_cluster_assignment = labels_to_assignment(
            self.decision_set_labels, n_labels = n_labels, ignore = {-1}
        )
        data_to_rules_assignment = self.get_data_to_rules_assignment(X, self.decision_set)
        selected_rules = self.pruner.prune(
            data_to_cluster_assignment = data_to_cluster_assignment,
            rule_to_cluster_assignment = rule_to_cluster_assignment,
            data_to_rules_assignment = data_to_rules_assignment
        )

        self.decision_set = [self.decision_set[i] for i in selected_rules]
        self.decision_set_labels = [self.decision_set_labels[i] for i in selected_rules]


    def get_data_to_rules_assignment(
            self,
            X : NDArray,
            decision_set : List[List[Condition]] = None
        ) -> NDArray:
        """
        Finds data points of X covered by each rule in the decision set.
        
        Args:
            X (np.ndarray): Input dataset.
            
            rule_list (List[List[Condition]], optional): List of rules to use for assignment.
        Returns:
            assignment (np.ndarray): n x n_rules boolean matrix with entry (i,j) being True
                if point i is covered by rule j and False otherwise.
        """
        if decision_set is None:
            decision_set = self.decision_set
        assignment = np.zeros((X.shape[0], len(decision_set)), dtype=bool)
        for i, condition_list in enumerate(decision_set):
            data_points_satisfied = satisfies_conditions(X, condition_list)
            assignment[data_points_satisfied, i] = True
        return assignment
    

####################################################################################################