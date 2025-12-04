import pandas as pd
from pyarc import TransactionDB
from pyarc.algorithms.rule_generation import generateCARs
from typing import List, Set, Tuple, Any
from intercluster import (
    Condition,
    entropy_bin,
    interval_to_condition,
    can_flatten,
    flatten_labels,
)

from .rule_miner import RuleMiner


####################################################################################################


class AssociationRuleMiner(RuleMiner):
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
        max_length : int = 2,
        ignore : Set[Any] = {-1},
        random_state : int = None
    ):
        """
        Initialize the AssociationRuleMiner.

        Args:
            min_support (float, optional): Minimum support for a rule. Defaults to 0.1.
            min_confidence (float, optional): Minimum confidence for a rule. Defaults to 0.8.
            max_length (int, optional): Maximum length of a rule (number of conditions). Defaults to 10.
            ignore (Set[Any], optional): Set of labels to ignore when mining rules. Defaults to {-1}.
            random_state (int, optional): Seed used by the random number generator.
                Defaults to None.

        Attributes:
            decision_set (List[List[Condition]]): The mined decision set, where each rule is a list of conditions.
            decision_set_labels (List[Set[int]]): The labels corresponding to each rule.
            cars (List[Any]): Decision set and labels in the classification association rule format 
                used by the IDS and CBA packages and algorithms.
            bin_df (pd.DataFrame): The binned version of the input dataset used for mining rules.
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
        self.ignore = ignore
        self.random_state = random_state
        super().__init__()

        self.bin_df = None


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
        bin_df = entropy_bin(X, y, random_state = self.random_state)
        bin_df.columns = bin_df.columns.astype(str)
        bin_df['class'] = y_
        bin_df = bin_df.astype(str)
        self.bin_df = bin_df

        txns = TransactionDB.from_DataFrame(bin_df, target = 'class')
        self.cars = generateCARs(
            txns,
            support = int(self.min_support * 100),
            confidence = int(self.min_confidence * 100),
            maxlen = self.max_length + 1, # +1 to account for the class label
            zmin = 1 # force rules with length at least 1
        )
        self.decision_set, self.decision_set_labels = self.cars_to_decision_set(self.cars)

        # remove rules covering outliers
        self.decision_set = [rule for i,rule in enumerate(self.decision_set) 
                             if self.decision_set_labels[i] not in self.ignore]
        self.decision_set_labels = [label for label in self.decision_set_labels 
                                    if label not in self.ignore]
        return self.decision_set, self.decision_set_labels
    

####################################################################################################