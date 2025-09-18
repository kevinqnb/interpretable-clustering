import numpy as np
import pandas as pd
from pyarc import TransactionDB
from pyarc.algorithms import M1Algorithm
from typing import List, Set, Tuple, Any
from numpy.typing import NDArray
from intercluster import (
    Condition,
    interval_to_condition,
)
from intercluster.mining import RuleMiner
from .decision_set import DecisionSet


####################################################################################################

# NOTE: The following code is a private submodule used to interface with the PyIDS package.
# Since PyIDS is not a dependency of Intercluster, it must be installed separately.

from pyids.algorithms.ids_classifier import mine_CARs
from pyids.algorithms.ids import IDS
from pyids.model_selection.coordinate_ascent import CoordinateAscent
from pyids.data_structures.ids_rule import IDSRule
from pyarc.qcba.data_structures import QuantitativeDataFrame


####################################################################################################


class CBA(DecisionSet):
    """
    A Classification Based Association Rule Set.

    This algorithm is based upon the paper:
    "Integrating Classification and Association Rule Mining"
    by Liu, Hsu, and Ma, 1998.

    We make use of the PyARC package to implement the algorithm:
    Jiri Filip, Tomas Kliegr. 
    https://github.com/jirifilip/pyARC
    

    Args:
        rule_miner (RuleMiner, optional): Rule mining algorithm used to generate the rules.
            If None, the rules must be provided directly. Defaults to None.
        rules (List[List[Condition]], optional): List of rules to initialize the decision set with.
            If None, the rules will be generated using the rule_miner. Defaults to None.
        rule_labels (List[Set[int]], optional): List of labels corresponding to each rule.
            If None, the labels will be generated using the rule_miner. Defaults to None.
    """
    def __init__(
        self,
        rule_miner : RuleMiner = None, 
        rules : List[List[Condition]] = None,
        rule_labels : List[Set[int]] = None

    ):
        super().__init__(rule_miner, rules, rule_labels)
        if self.rule_miner is None:
            raise ValueError("A rule_miner must be provided for this decision set.")

    
    def cars_to_decision_set(self, cars : List[IDSRule]) -> List[List[Condition]]:
        """
        Convert a list of rules found with PyARC to a list of Conditions.
        Args:
            cars (List[IDSRule]): A list of Class Association Rules (CARs).
        Returns:
            list: A list of Rule objects.
        """
        decision_set = []
        decision_set_labels = []
        for car in cars:
            antecedent = car.antecedent
            consequent = car.consequent
            rule_conditions = []
            for condition in antecedent:
                feature = int(condition[0])
                interval = condition[1]
                # Convert the interval to two Conditions
                # (one for the lower bound and one for the upper bound)
                lower_condition, upper_condition = interval_to_condition(feature, interval)
                rule_conditions.append(lower_condition)
                rule_conditions.append(upper_condition)
            decision_set.append(rule_conditions)
            decision_set_labels.append({int(consequent.value)})
        return decision_set, decision_set_labels


    def prune(self, X : NDArray, y : List[Set[int]] = None):
        """
        Prunes the decision set using the pruner.
        
        Args:
            X (np.ndarray): Input dataset.
            
            y (List[Set[int]], optional): Target labels. Defaults to None.
        """
        if self.decision_set is None or self.decision_set_labels is None:
            raise ValueError('Decision set has not been fitted yet.')
        
        transactions = TransactionDB.from_DataFrame(self.rule_miner.bin_df, target = 'class')

        valid_cars = [car for i,car in enumerate(self.rule_miner.cars) 
                                if int(self.rule_miner.cars[i].consequent[1]) != -1]
        if len(valid_cars) == 0:
            raise ValueError("No valid (non-outlier) class association rules found. " \
            "Try increasing the number of mined rules.")

        transactions = TransactionDB.from_DataFrame(self.rule_miner.bin_df, target = 'class')
        classifier = M1Algorithm(valid_cars, transactions).build()
        decision_set, decision_set_labels = self.cars_to_decision_set(classifier.rules)
        return decision_set, decision_set_labels
    

####################################################################################################