import numpy as np
import pandas as pd
import pyarc
from pyids.algorithms.ids_classifier import mine_CARs
from pyarc import TransactionDB
from pyarc.algorithms.rule_generation import generateCARs
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


class ClassAssociationMiner(RuleMiner):
    """
    Classification Association Rule Miner
    Rule miner that uses association rule mining to generate rules.

    This is a wrapper around the PyIDS package [https://github.com/jirifilip/pyIDS/tree/master],
    which implements a classifcation and association rule mining algorithm based upon:
    Liu, B., Hsu, W., & Ma, Y. (1998, July). Integrating Classification and Association Rule Mining.
    """
    def __init__(
        self,
        min_support : float = 0.1,
        min_confidence : float = 0.8,
        max_length : int = 10
    ):
        """
        Initialize the AssociationRuleMiner.

        Args:
            min_support (float, optional): Minimum support for a rule. Defaults to 0.1.
            min_confidence (float, optional): Minimum confidence for a rule. Defaults to 0.8.
        """
        if not isinstance(min_support, float) or min_support < 0 or min_support > 1:
            raise ValueError("min_support must be a floating point number in [0, 1].")
        if not isinstance(min_confidence, float) or min_confidence < 0 or min_confidence > 1:
            raise ValueError("min_confidence must be a floating point number in [0, 1].")
        if not isinstance(max_length, int) or max_length <= 0:
            raise ValueError("max_length must be a positive integer.")
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.max_length = max_length
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
        bin_df = entropy_bin(X, y)
        bin_df.columns = bin_df.columns.astype(str)
        bin_df['class'] = y_
        bin_df = bin_df.astype(str)
        self.bin_df = bin_df

        txns = TransactionDB.from_DataFrame(bin_df, target = 'class')
        self.cars = generateCARs(
            txns,
            support = int(self.min_support * 100),
            confidence = int(self.min_confidence * 100),
            maxlen = self.max_length
        )
        self.decision_set, self.decision_set_labels = self.cars_to_decision_set(self.cars)
        return self.decision_set, self.decision_set_labels
    

####################################################################################################


class PointwiseMinerV2(RuleMiner):
    """
    Rule miner that uses the Pointwise algorithm to generate rules. Specifically, this will 
    expand randomly chosen rectangular regions around each data point until specified stopping 
    conditions are reached.
    """
    def __init__(
        self,
        samples : int = 10,
        prob_dim : float = 1/2,
        prob_stop : float = 1.0
    ):
        """
        Initialize the PointwiseMiner.

        Args:
            samples (int, optional): Number of samples to draw for each data point. 
                Defaults to 10.
            prob_dim (float, optional): Probability for a geometric distribution used to choose 
                the number of dimensions to use in each rule. Defaults to 1/2, in which case 
                the expected number of dimensions used is 2.
            prob_stop (float, optional): Probability of stopping expansion when a 
                mistake is encountered (geometric distribution). Defaults to 1.0, in which case
                the expansion will always stop when a mistake is encountered.
        """
        if not isinstance(samples, int) or samples <= 0:
            raise ValueError("Number of samples must be a positive integer.")
        if not isinstance(prob_dim, float) or prob_dim < 0 or prob_dim > 1:
            raise ValueError("prob_dim must be a floating point number in [0, 1].")
        if not isinstance(prob_stop, float) or prob_stop < 0 or prob_stop > 1:
            raise ValueError("prob_stop must be a floating point number in [0, 1].")
        self.samples = samples
        self.prob_dim = prob_dim
        self.prob_stop = prob_stop
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

        for _ in range(self.samples):
            for i in non_outliers:
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
                                if y[i] == y[new_point]:
                                    # If the label of the new point matches the label of the 
                                    # current point, we automatically add it.
                                    pass
                                elif np.random.rand() > self.prob_stop:
                                    # If the label does not match, we only add it with a 
                                    # probability of prob_stop
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
                                if y[i] == y[new_point]:
                                    # If the label of the new point matches the label of the 
                                    # current point, we automatically add it.
                                    pass
                                elif np.random.rand() > self.prob_stop:
                                    # If the label does not match, we only add it with a 
                                    # probability of prob_stop
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