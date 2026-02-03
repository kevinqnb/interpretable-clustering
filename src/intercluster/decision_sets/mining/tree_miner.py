from numpy.typing import NDArray
from typing import List, Set
from intercluster import (
    Rule,
    collect_node_rules,
    collect_leaf_rules
)
from intercluster.decision_trees import Tree

from .rule_miner import RuleMiner


####################################################################################################


class TreeMiner(RuleMiner):
    """
    Rule miner that extracts rules from a decision tree.

    Args:
        tree (Tree): The decision tree to mine rules from.
        leaf_rules (bool, optional): Whether to only extract rules from leaf nodes. 
            Defaults to False.

    Attrs:
        rules (List[Rule]): The mined rules, where each rule is a list of conditions.
        rule_labels (List[Set[int]]): The labels corresponding to each rule.
    """
    def __init__(
        self,
        tree: Tree,
        leaf_rules: bool = False
    ):
        self.tree = tree
        self.leaf_rules = leaf_rules
        super().__init__()

    def fit(
        self,
        X : NDArray,
        y : List[Set[int]] = None
    ) -> tuple[list[Rule], None]:
        """
        Fit the FrequentItemsetMiner to the input dataset.

        Args:
            X (pd.DataFrame): Input dataset.
            y (List[Set[int]], optional): Dummy parameter for compatibility. Defaults to None.

        Returns:
            rules (List[Rule]): List of rules.
            rule_labels (List[Set[int]]): None, dummy variable.
        """
        self.tree.fit(X, y)
        if self.leaf_rules:
            self.rules = collect_leaf_rules(self.tree.root)
        else:
            self.rules = collect_node_rules(self.tree.root)
        return self.rules, None


####################################################################################################