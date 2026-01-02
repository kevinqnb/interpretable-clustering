from numpy.typing import NDArray
from typing import List, Set
from intercluster import (
    collect_node_rules
)
from intercluster.decision_trees import Tree

from .rule_miner import RuleMiner


####################################################################################################


class TreeMiner(RuleMiner):
    """
    Rule miner that extracts rules from a decision tree.

    Args:
        tree (Tree): The decision tree to mine rules from.

    Attrs:
        decision_set (List[List[Condition]]): The mined decision set,
            where each rule is a list of conditions.
        bin_df (pd.DataFrame): Not implemented. Dummy variable for compatibility.
    """
    def __init__(
        self,
        tree: Tree,
    ):
        self.tree = tree
        super().__init__()
        self.bin_df = None

    def fit(
        self,
        X : NDArray,
        y : List[Set[int]] = None
    ):
        """
        Fit the FrequentItemsetMiner to the input dataset.

        Args:
            X (pd.DataFrame): Input dataset.
            y (List[Set[int]], optional): Dummy parameter for compatibility. Defaults to None.

        Returns:
            rules (List[List[Condition]]): List of rules, where each rule is a list of conditions.
            rule_labels (List[Set[int]]): None, dummy variable.
        """
        self.tree.fit(X, y)
        self.decision_set = collect_node_rules(self.tree.root)
        return self.decision_set, None
    

####################################################################################################