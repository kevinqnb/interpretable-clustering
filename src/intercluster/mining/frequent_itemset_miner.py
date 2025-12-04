import numpy as np
import pandas as pd
from pyarc import TransactionDB
from pyarc.algorithms.rule_generation import generateCARs
import fim
import pyarc
from numpy.typing import NDArray
from typing import List, Set, Tuple, Any
from intercluster import (
    Condition,
    uniform_bin,
    interval_to_condition,
    can_flatten,
    flatten_labels,
    satisfies_conditions
)

from .rule_miner import RuleMiner


####################################################################################################


class FrequentItemsetMiner(RuleMiner):
    """
    Frequent Itemset Rule Miner
    
    This is based upon the following code:
    https://github.com/jirifilip/pyIDS/blob/master/pyids/rule_mining/rule_miner.py

    But is implemented with some custom logic specific to this library. 
    Given a set of classes, this miner will generate association rules by creating 
    frequent itemsets and then assigning each itemset to every class in the set 
    (creating k copies of each rule).
    """
    def __init__(
        self,
        n_bins : int = 10,
        min_support : float = 0.1,
        max_length : int = 2,
        class_labels : Set[Any] = None,
        random_state : int = None
    ):
        """
        Initialize the AssociationRuleMiner.

        Args:
            n_bins (int, optional): Number of bins to use when discretizing continuous features.
            min_support (float, optional): Minimum support for a rule. Defaults to 0.1.
            max_length (int, optional): Maximum length of a rule (number of conditions). Defaults to 10.
            class_labels (Set[Any], optional): Set of labels to consider when mining rules. Defaults to 
                None, in which case all rules will be classless. If given a set of classes, the 
                output will contain a copy of each rule assigned to each class in the set.
            random_state (int, optional): Seed used by the random number generator.
                Defaults to None.

        Attributes:
            decision_set (List[List[Condition]]): The mined decision set, 
                where each rule is a list of conditions.
            decision_set_labels (List[Set[int]]): If given a set of class labels, 
                this list gives the labels corresponding to each rule.
            cars (List[Any]): If given a set of class labels, this
                gives the rules and labels classification association rule format 
                used by the IDS and CBA packages and algorithms.
            bin_df (pd.DataFrame): The binned version of the input dataset used for mining rules.
        """
        if not isinstance(n_bins, int) or n_bins <= 0:
            raise ValueError("n_bins must be a positive integer.")
        if not isinstance(min_support, float) or min_support < 0 or min_support > 1:
            raise ValueError("min_support must be a floating point number in [0, 1].")
        if not isinstance(max_length, int) or max_length <= 0:
            raise ValueError("max_length must be a positive integer.")
        
        self.n_bins = n_bins
        self.min_support = min_support
        self.max_length = max_length
        self.class_labels = class_labels
        self.random_state = random_state
        super().__init__()

        self.decision_set = None
        self.decision_set_labels = None
        self.cars = None
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
            X : NDArray,
            y : List[Set[int]] = None,
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
        bin_df = uniform_bin(X, self.n_bins)
        bin_df.columns = bin_df.columns.astype(str)
        bin_df = bin_df.astype(str)
        self.bin_df = bin_df

        txns = TransactionDB.from_DataFrame(bin_df)
        frequent_itemsets = fim.apriori(
            txns.string_representation, supp=self.min_support*100, report="s"
        )

        # Convert to decision set format:
        rules = []
        for itemset in frequent_itemsets:
            antecedent, support = itemset
            rule = []
            for condition in antecedent:
                feature, interval = condition.split(':=:')
                feature = int(feature)
                lower_condition, upper_condition = interval_to_condition(feature, interval)
                rule.append(lower_condition)
                rule.append(upper_condition)
            rules.append(rule)
        
        # Create association rules if applicable:
        if self.class_labels is not None:
            # Assign each itemset to each class label:
            self.decision_set = []
            self.decision_set_labels = []
            for rule in rules:
                for label in self.class_labels:
                    self.decision_set.append(rules)
                    self.decision_set_labels.append({label})

            # Create cars in classification association rule format:
            if y is not None:
                if not can_flatten(y):
                    raise ValueError("Each data point must be assigned to a single label.")
                y_ = flatten_labels(y)
                self.cars = []
                for rule in rules:
                    for label in self.class_labels:
                        antecedent = []
                        for condition in rule:
                            feature = condition.feature
                            interval = condition.interval
                            antecedent.append((str(feature), interval))
                        consequent = ('class', str(label))
                        car = fim.CAR(antecedent, consequent)
                        self.cars.append(car)


                    antecedent = pyarc.data_structures.antecedent.Antecedent(items = antecedent)
                    consequent = pyarc.data_structures.consequent.Consequent(attribute = 'class', value = str(list(y_[i])[0]))
                    sat = satisfies_conditions(X, rule)
                    support = len(sat) / len(X)
                    confidence = len([idx for idx in sat if y[idx] == y[i]]) / len(sat) if len(sat) > 0 else 0
                    car = pyarc.data_structures.car.ClassAssocationRule(antecedent, consequent, support, confidence)
                    cars.append(car)
        else:
            self.decision_set = rules

        return self.decision_set, self.decision_set_labels
    

####################################################################################################