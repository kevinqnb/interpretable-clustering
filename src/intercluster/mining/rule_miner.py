import pandas as pd
from typing import List, Set, Tuple
from intercluster import (
    Condition
)


####################################################################################################


class RuleMiner:
    """
    Base class for rule mining algorithms.

    Attributes:
        decision_set (List[List[Condition]]): The mined decision set, where each rule is a list of conditions.
        decision_set_labels (List[Set[int]]): The labels corresponding to each rule.
    """
    def __init__(self):
        self.decision_set = None
        self.decision_set_labels = None
        self.bin_df = None
        self.cars = None


    def clear_cache(self):
        """
        Clear the currently stored rules.
        """
        self.decision_set = None
        self.decision_set_labels = None
        self.bin_df = None
        self.cars = None

    
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