import numpy as np
from sklearn.ensemble import RandomForestClassifier
from intercluster import Node, collect_node_rules, collect_leaf_rules
from intercluster.rules import Rule
from intercluster.decision_trees import DecisionTree
from intercluster.utils import can_flatten, flatten_labels
from .rule_miner import RuleMiner


class RandomForestMiner(RuleMiner):
    """
    Rule miner that extracts rules from a random forest model.

    Args:
        forest_params (dict, optional): Parameters for the RandomForestClassifier. Defaults to None.

    Attrs:
        decision_set (List[List[Condition]]): The mined decision set,
            where each rule is a list of conditions.
        bin_df (pd.DataFrame): Not implemented. Dummy variable for compatibility.
    """
    def __init__(self, forest_params = None):
        if forest_params is None:
            forest_params = {}
        self.forest_params = forest_params
        super().__init__()
        self.bin_df = None

    def fit(self, X : np.ndarray, y : np.ndarray) -> tuple[list[Rule], None]:
        """
        Fit the RandomForestMiner to the input dataset.

        Args:
            X (pd.DataFrame): Input dataset.
            y (List[Set[int]], optional): Dummy parameter for compatibility. Defaults to None.

        Returns:
            rules (List[Rule]): List of rules.
            rule_labels (List[Set[int]]): None, dummy variable.
        """
        if not can_flatten(y):
            raise ValueError("Each data point must have exactly one label.")
        y_array = flatten_labels(y)
        forest = RandomForestClassifier(**self.forest_params)
        forest.fit(X, y_array)

        self.decision_set = []
        for tree in forest.estimators_:
            dtree = DecisionTree()
            dtree.X = X
            dtree.y_array = y_array
            dtree.classes = tree.classes_
            dtree.tree_info = tree.tree_
            dtree.root = Node()
            indices = np.arange(len(X))
            dtree.grow(indices, 0, dtree.root, 0)
            dtree.node_count += 1
            self.decision_set.extend(collect_node_rules(dtree.root))

        return self.decision_set, None