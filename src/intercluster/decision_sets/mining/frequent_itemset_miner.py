from pyarc import TransactionDB
import fim
from numpy.typing import NDArray
from typing import List, Set
from intercluster import (
    uniform_bin,
    quantile_bin,
    oned_cluster_bin,
    interval_to_condition,
    Rule
)

from .rule_miner import RuleMiner


####################################################################################################


class FrequentItemsetMiner(RuleMiner):
    """
    Rule miner that uses frequent itemset mining to generate rules.

    This is a wrapper around the PyFIM package, and is based upon the following code:
    https://github.com/jirifilip/pyIDS/blob/master/pyids/rule_mining/rule_miner.py

    Args:
        min_support (float, optional): Minimum support for a rule. Defaults to 0.1.
        binning_method (str, optional): Binning method to use. Options are "uniform" or "quantile".
            Defaults to "uniform".
        bin_params (dict, optional): Parameters for the binning method. Defaults to standard 
            uniform binning parameters.

    Attrs:
        decision_set (List[Rule]): The mined decision set.
        bin_df (pd.DataFrame): The binned version of the input dataset used for mining rules.
    """
    def __init__(
        self,
        min_support : float = 0.1,
        max_length : int = None,
        binning_method : str = "uniform",
        bin_params : dict = {'n_bins': 5}
    ):
        if not isinstance(min_support, float) or min_support < 0 or min_support > 1:
            raise ValueError("min_support must be a floating point number in [0, 1].")
        self.min_support = min_support

        if max_length is not None:
            if not isinstance(max_length, int) or max_length < 1:
                raise ValueError("max_length must be a positive integer.")
        self.max_length = max_length

        if binning_method not in ["uniform", "quantile", "cluster"]:
            raise ValueError("Unsupported binning method. Choose 'uniform' or 'quantile' or 'cluster'.")
        self.binning_method = binning_method

        self.bin_params = bin_params

        super().__init__()

    def fit(
            self,
            X : NDArray,
            y : List[Set[int]] = None
    ) -> tuple[list[Rule], list[set[int]]]:
        """
        Fit the FrequentItemsetMiner to the input dataset.

        Args:
            X (pd.DataFrame): Input dataset.
            y (List[Set[int]], optional): Dummy parameter for compatibility. Defaults to None.

        Returns:
            rules (List[Rule]): List of rules.
            rule_labels (List[Set[int]]): None, dummy variable.
        """
        if self.binning_method == "quantile":
            bin_df = quantile_bin(X, **self.bin_params)
        elif self.binning_method == "uniform":
            bin_df = uniform_bin(X, **self.bin_params)
        else:
            bin_df = oned_cluster_bin(X, **self.bin_params)

        bin_df.columns = bin_df.columns.astype(str)
        bin_df = bin_df.astype(str)
        self.bin_df = bin_df

        txns = TransactionDB.from_DataFrame(bin_df)
        if self.max_length is not None:
            frequent_itemsets = fim.apriori(
                txns.string_representation,
                supp=self.min_support*100,
                zmax = self.max_length
            )
        else:
            frequent_itemsets = fim.apriori(
                txns.string_representation,
                supp=self.min_support*100
            )

        # Convert to decision set format:
        self.decision_set = []
        for itemset in frequent_itemsets:
            antecedent, support = itemset
            conditions = []
            for condition in antecedent:
                feature, interval = condition.split(':=:')
                feature = int(feature)
                lower_condition, upper_condition = interval_to_condition(feature, interval)
                conditions.append(lower_condition)
                conditions.append(upper_condition)
            self.decision_set.append(Rule(conditions))

        return self.decision_set, None
    

####################################################################################################