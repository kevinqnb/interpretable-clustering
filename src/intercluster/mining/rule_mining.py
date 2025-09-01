import numpy as np
import pandas as pd
import pyarc
from pyids.algorithms.ids_classifier import mine_CARs
from mdlp.discretization import MDLP
from typing import List, Set, Tuple, Any, Dict
from numpy.typing import NDArray
from intercluster import (
    Condition,
    LinearCondition,
    entropy_bin,
    quantile_bin,
    uniform_bin,
    interval_to_condition,
    can_flatten,
    flatten_labels,
    satisfies_conditions,
)


####################################################################################################


class RuleMiner:
    """
    Base class for rule mining algorithms.
    """
    def __init__(self):
        pass

    
    def fit(
            self,
            X : pd.DataFrame,
            y : List[Set[int]] = None
        ) -> Tuple[List[List[Condition]], List[Set[int]]]:
        """
        Fit the rule mining algorithm to the input dataset.

        Args:
            X (pd.DataFrame): Input dataset.
            y (List[Set[int]], optional): Target labels. Defaults to None.

        Returns:
            rules (List[List[Condition]]): List of rules, where each rule is a list of conditions.
            rule_labels (List[Set[int]]): List of labels corresponding to each rule.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
    

####################################################################################################


class AssociationRuleMiner(RuleMiner):
    """
    Rule miner that uses association rule mining to generate rules.

    This is a wrapper around the PyIDS package [https://github.com/jirifilip/pyIDS/tree/master],
    which implements a classifcationa and association rule mining algorithm based upon:
    Liu, B., Hsu, W., & Ma, Y. (1998, July). Integrating classification and association rule mining.
    """
    def __init__(
        self,
        max_rules : int = 100,
        bin_type : str = 'mdlp',
        n_bins : int = None
    ):
        """
        Initialize the AssociationRuleMiner.

        Args:
            max_rules (int, optional): Maximum number of rules to mine. Defaults to 100.
            bin_type (str, optional): Type of binning to use 
                ('mdlp', 'quantile', or 'uniform'). Defaults to 'mdlp' in which case bins are 
                chosen to minimize entropy.
            n_bins (int, optional): Number of bins to use if bin_type is 'uniform'. Defaults to None.
        """
        if not isinstance(max_rules, int) or max_rules <= 0:
            raise ValueError("max_rules must be a positive integer.")
        if bin_type not in ['mdlp', 'quantile', 'uniform']:
            raise ValueError("bin_type must be one of 'mdlp', 'quantile', or 'uniform'.")
        if bin_type == 'uniform' and (not isinstance(n_bins, int) or n_bins <= 1):
            raise ValueError("n_bins must be an integer greater than 1 when bin_type is 'uniform'.")
        if bin_type == 'quantile' and (not isinstance(n_bins, int) or n_bins <= 1):
            raise ValueError("n_bins must be an integer greater than 1 when bin_type is 'quantile'.")
        self.max_rules = max_rules
        self.bin_type = bin_type
        self.n_bins = n_bins
        super().__init__()


    def cars_to_decision_set(
            self,
            cars : List[Any]
        ) -> Tuple[List[List[Condition]], List[Set[int]]]:
        """
        Convert a list of rules found with PyIDS to a list of Conditions and label sets.
        Args:
            cars (List[Any]): A list of Class Association Rules (CARs).
        Returns:
            decision_set (List[Condition]): List of rules.
            decision_set_labels (List[Set[int]]): List of labels corresponding to each rule.
        """
        decision_set = []
        decision_set_labels = []
        for car in cars:
            antecedent = car.antecedent
            consequent = car.consequent
            rule_conditions = []
            for a in antecedent:
                feature = int(a[0])
                interval = a[1]
                lower_condition, upper_condition = interval_to_condition(feature, interval)
                rule_conditions.append(lower_condition)
                rule_conditions.append(upper_condition)
            decision_set.append(rule_conditions)
            decision_set_labels.append({int(consequent[1])})
        return decision_set, decision_set_labels


    def fit(
            self,
            X : pd.DataFrame,
            y : List[Set[int]],
        ) -> Tuple[List[List[Condition]], List[Set[int]]]:
        """
        Fit the AssociationRuleMiner to the input dataset.

        Args:
            X (pd.DataFrame): Input dataset.
            y (List[Set[int]], optional): Target labels. Defaults to None.

        Returns:
            rules (List[List[Condition]]): List of rules, where each rule is a list of conditions.
            rule_labels (List[Set[int]]): List of labels corresponding to each rule.
        """
        if not can_flatten(y):
            raise ValueError("Each data point must be assigned to a single label.")
        
        y_ = flatten_labels(y)

        if self.bin_type == 'mdlp':
            bin_df = entropy_bin(X, y)
        elif self.bin_type == 'quantile':
            bin_df = quantile_bin(X, self.n_bins)
        elif self.bin_type == 'uniform':
            bin_df = uniform_bin(X, self.n_bins)
        bin_df.columns = bin_df.columns.astype(str)
        bin_df['class'] = y_
        bin_df = bin_df.astype(str)
        self.bin_df = bin_df

        self.cars = mine_CARs(bin_df, self.max_rules)
        self.decision_set, self.decision_set_labels = self.cars_to_decision_set(self.cars)
        return self.decision_set, self.decision_set_labels
    

####################################################################################################


class PointwiseMiner(RuleMiner):
    """
    Rule miner that uses the Pointwise algorithm to generate rules. Specifically, this will 
    expand randomly chosen rectangular regions around each data point until specified stopping 
    conditions are reached.
    """
    def __init__(
        self,
        lambd : float = 1.0,
        n_features : int = 2,
        rules_per_point : int = 10,
    ):
        """
        Initialize the PointwiseMiner.

        Args:
            lambd (float): Penalization factor for mistakes. Larger values penalize mistakes 
                more heavily, resulting in rules which are more accurate, but may cover 
                fewer points.
            
            n_features (int): Number of randomly chosen features to use for each rule.

            rules_per_point (int): Number of random rules to create for each point in the dataset.
        """
        if not isinstance(lambd, float) or lambd < 0:
            raise ValueError("lambd must be a non-negative floating point.")
        if not isinstance(n_features, int) or n_features <= 0:
            raise ValueError("n_features must be a positive integer.")
        if not isinstance(rules_per_point, int) or rules_per_point <= 0:
            raise ValueError("rules_per_point must be a positive integer.")
        self.lambd = lambd
        self.n_features = n_features
        self.rules_per_point = rules_per_point
        super().__init__()
    

    def fit(
            self,
            X : NDArray,
            y : List[Set[int]],
        ) -> Tuple[List[List[Condition]], List[Set[int]]]:
        """
        Creates rules for the decision set by drawing boxes around dense sets of points 
        in randomly chosen dimensions. 

        Args:
            X (NDArray): Input dataset.
            y (List[Set[int]], optional): Target labels. Defaults to None.

        Returns:
            rules (List[List[Condition]]): List of rules, where each rule is a list of conditions.
            rule_labels (List[Set[int]]): List of labels corresponding to each rule.
        """
        n,d = X.shape
        X_sorted = np.argsort(X, axis=0)
        decision_set = []
        decision_set_labels = []
        cars = []
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
                antecedent = []
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

                    antecedent.append((f, str('(' + str(lower_bound) + ', ' + str(upper_bound) + ']')))


                decision_set.append(rule)
                decision_set_labels.append(y[i])

                antecedent = pyarc.data_structures.antecedent.Antecedent(items = antecedent)
                consequent = pyarc.data_structures.consequent.Consequent(attribute = 'class', value = str(list(y[i])[0]))
                sat = satisfies_conditions(X, rule)
                support = len(sat) / len(X)
                confidence = len([idx for idx in sat if y[idx] == y[i]]) / len(sat) if len(sat) > 0 else 0
                car = pyarc.data_structures.car.ClassAssocationRule(antecedent, consequent, support, confidence)
                cars.append(car)


        self.decision_set = decision_set
        self.decision_set_labels = decision_set_labels
        self.cars = cars
        return self.decision_set, self.decision_set_labels


####################################################################################################