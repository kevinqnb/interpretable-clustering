import pandas as pd
from typing import List, Set, Tuple
from intercluster import Rule



####################################################################################################


class RuleMiner:
    """
    Base class for rule mining algorithms.

    Attributes:
        rules (List[Rule]): The mined rules, where each rule is a list of conditions.
        rule_labels (List[Set[int]]): The labels corresponding to each rule.
    """
    def __init__(self):
        self.rules = None
        self.rule_labels = None


    def clear_cache(self):
        """
        Clear the currently stored rules.
        """
        self.rules = None
        self.rule_labels = None

    
    def fit(
            self,
            X : pd.DataFrame,
            y : List[Set[int]] = None
        ) -> Tuple[List[Rule], List[Set[int]]]:
        """
        Fit the rule mining algorithm to the input dataset.

        Args:
            X (pd.DataFrame): Input dataset.
            y (List[Set[int]], optional): Target labels. Defaults to None.

        Returns:
            rules (List[Rule]): List of rules.
            rule_labels (List[Set[int]]): List of labels corresponding to each rule.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
    

####################################################################################################