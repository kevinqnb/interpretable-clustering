import pandas as pd
from pyarc import TransactionDB
import fim
from pyarc.algorithms.rule_generation import generateCARs
from typing import List, Set, Tuple, Any
from intercluster import (
    Condition,
    Rule,
    entropy_bin,
    uniform_bin,
    quantile_bin,
    oned_cluster_bin,
    interval_to_condition,
    can_flatten,
    flatten_labels,
    cars_to_decision_set
)

from .rule_miner import RuleMiner


####################################################################################################


class ClassAssociationRuleMiner(RuleMiner):
    """
    Classification Association Rule Miner
    Rule miner that uses association rule mining to generate rules.

    This is based upon the following code:
    https://github.com/jirifilip/pyARC/blob/master/pyarc/algorithms/rule_generation.py
    """
    def __init__(
        self,
        min_support : float = 0.1,
        min_confidence : float = 0.8,
        max_length : int = None,
        bin_df = None,
        binning_method : str = "entropy",
        bin_params : dict = dict(),
        ignore : Set[Any] = {-1},
    ):
        """
        Initialize the AssociationRuleMiner.

        Args:
            min_support (float, optional): Minimum support for a rule. Defaults to 0.1.
            min_confidence (float, optional): Minimum confidence for a rule. Defaults to 0.8.
            max_length (int, optional): Maximum length of a rule (number of conditions). Defaults to 10.
            binning_method (str, optional): Binning method to use. 
                Options are "uniform", "quantile", "cluster", or "entropy". Defaults to "entropy".
            bin_params (dict, optional): Parameters for the binning method. Defaults to standard 
                entropy binning parameters (just a random state).
            ignore (Set[Any], optional): Set of labels to ignore when mining rules. Defaults to {-1}.

        Attributes:
            decision_set (List[Rule]): The mined decision set.
            decision_set_labels (List[Set[int]]): The labels corresponding to each rule.
            bin_df (pd.DataFrame): The binned version of the input dataset used for mining rules.
        """
        if not isinstance(min_support, float) or min_support < 0 or min_support > 1:
            raise ValueError("min_support must be a floating point number in [0, 1].")
        if not isinstance(min_confidence, float) or min_confidence < 0 or min_confidence > 1:
            raise ValueError("min_confidence must be a floating point number in [0, 1].")
        if max_length is not None:
            if not (isinstance(max_length, int) or max_length is None) or max_length <= 0:
                raise ValueError("max_length must be a positive integer.")
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.max_length = max_length

        self.bin_df = bin_df
        if self.bin_df is None:
            if binning_method not in ["uniform", "quantile", "cluster", "entropy"]:
                raise ValueError(
                    "Unsupported binning method. Choose 'uniform', 'quantile', 'cluster', or 'entropy'."
                )
            self.binning_method = binning_method
            self.bin_params = bin_params
        self.ignore = ignore
        super().__init__()

        self.bin_df = None


    def fit(
            self,
            X : pd.DataFrame,
            y : List[Set[int]],
        ) -> Tuple[List[Rule], List[Set[int]]]:
        """
        Fit the AssociationRuleMiner to the input dataset.

        Args:
            X (pd.DataFrame): Input dataset.
            y (List[Set[int]], optional): Target labels. Defaults to None.

        Returns:
            rules (List[Rule]): List of rules.
            rule_labels (List[Set[int]]): List of labels corresponding to each rule.
        """
        if not can_flatten(y):
            raise ValueError("Each data point must be assigned to a single label.")
        y_ = flatten_labels(y)

        if self.bin_df is not None:
            bin_df = self.bin_df
        elif self.binning_method == "quantile":
            bin_df = quantile_bin(X, **self.bin_params)
        elif self.binning_method == "uniform":
            bin_df = uniform_bin(X, **self.bin_params)
        elif self.binning_method == "cluster":
            bin_df = oned_cluster_bin(X, **self.bin_params)
        else: #elif self.binning_method == "entropy":
            bin_df = entropy_bin(X, y, **self.bin_params)

        bin_df.columns = bin_df.columns.astype(str)
        bin_df['class'] = y_
        bin_df = bin_df.astype(str)
        self.bin_df = bin_df

        # Diagnostic information
        print(f"Dataset size: {bin_df.shape[0]} rows, {bin_df.shape[1]} columns")
        print(f"Min support: {self.min_support} ({self.min_support*100}%)")
        print(f"Min confidence: {self.min_confidence} ({self.min_confidence*100}%)")
        print(f"Max length: {self.max_length}")
        print(f"Unique values per column: {[bin_df[col].nunique() for col in bin_df.columns[:5]]}...")  # First 5 cols
        
        txns = TransactionDB.from_DataFrame(bin_df, target = 'class')
        print(f"Transaction DB size: {len(txns.string_representation)} transactions")

        if self.max_length is not None:
            cars = fim.apriori(
                txns.string_representation,
                supp=self.min_support*100,
                conf=self.min_confidence*100,
                mode="o",
                target="r",
                report="sc",
                zmax= self.max_length + 1, # +1 to account for the class label
                appear=txns.appeardict
            )
        else:
            cars = fim.apriori(
                txns.string_representation,
                supp=self.min_support*100,
                conf=self.min_confidence*100,
                mode="o",
                target="r",
                report="sc",
                appear=txns.appeardict
            )

        print(f"Generated {len(cars)} class association rules.")

        self.decision_set = []
        self.decision_set_labels = []

        for car in cars:
            con, ant, support, confidence = car
            consequent = int(con.split(':=:')[1])
            conditions = []
            for condition in ant:
                feature, interval = condition.split(':=:')
                feature = int(feature)
                lower_condition, upper_condition = interval_to_condition(feature, interval)
                conditions.append(lower_condition)
                conditions.append(upper_condition)
            self.decision_set.append(Rule(conditions))
            self.decision_set_labels.append({consequent})

        # remove rules covering outliers
        self.decision_set = [rule for i,rule in enumerate(self.decision_set) 
                             if self.decision_set_labels[i] not in self.ignore]
        self.decision_set_labels = [label for label in self.decision_set_labels 
                                    if label not in self.ignore]
        return self.decision_set, self.decision_set_labels
    

####################################################################################################