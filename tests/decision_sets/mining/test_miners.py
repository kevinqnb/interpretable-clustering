import numpy as np
import pandas as pd
import pytest

from intercluster import Rule
from intercluster.utils import labels_format
from intercluster.decision_trees import DecisionTree
from intercluster.decision_sets.mining import (
    RandomForestMiner,
    ClassAssociationRuleMiner,
    FrequentItemsetMiner,
    TreeMiner,
)


####################################################################################################
# Shared fixture: a small, clearly-separable 2-class dataset -- large/clean enough that
# apriori-based miners reliably find rules meeting reasonable support/confidence thresholds.
####################################################################################################


@pytest.fixture
def two_class_dataset():
    rng = np.random.default_rng(0)
    n_per_class = 20
    class0 = rng.normal(loc = [0.0, 0.0], scale = 0.5, size = (n_per_class, 2))
    class1 = rng.normal(loc = [10.0, 10.0], scale = 0.5, size = (n_per_class, 2))
    X = np.vstack([class0, class1])
    y_flat = np.array([0] * n_per_class + [1] * n_per_class)
    y = labels_format(y_flat)
    return X, y, y_flat


####################################################################################################
# RandomForestMiner
####################################################################################################


def test_random_forest_miner_basic(two_class_dataset):
    X, y, y_flat = two_class_dataset
    miner = RandomForestMiner(forest_params = {'n_estimators': 3, 'max_depth': 2, 'random_state': 0})
    rules, rule_labels = miner.fit(X, y)

    assert len(rules) > 0
    assert all(isinstance(r, Rule) for r in rules)
    # RandomForestMiner is documented to always return None for rule_labels.
    assert rule_labels is None
    assert miner.rules is rules


def test_random_forest_miner_leaf_rules_only(two_class_dataset):
    X, y, y_flat = two_class_dataset
    all_rules, _ = RandomForestMiner(
        forest_params = {'n_estimators': 3, 'max_depth': 2, 'random_state': 0}, leaf_rules = False
    ).fit(X, y)
    leaf_rules, _ = RandomForestMiner(
        forest_params = {'n_estimators': 3, 'max_depth': 2, 'random_state': 0}, leaf_rules = True
    ).fit(X, y)
    # Leaf-only rules are a subset of all node rules (leaves are a subset of all nodes).
    assert len(leaf_rules) <= len(all_rules)
    assert len(leaf_rules) > 0


def test_random_forest_miner_rejects_multilabel():
    X = np.array([[0.0], [1.0], [2.0]])
    y = [{0}, {0, 1}, {1}]
    miner = RandomForestMiner(forest_params = {'n_estimators': 2})
    with pytest.raises(ValueError):
        miner.fit(X, y)


####################################################################################################
# TreeMiner (for parity with the other miners; already exercised indirectly via
# test_heap_distorted_greedy.py's aniso_dataset fixture)
####################################################################################################


def test_tree_miner_basic(two_class_dataset):
    X, y, _ = two_class_dataset
    tree = DecisionTree(max_depth = 2, random_state = 0)
    miner = TreeMiner(tree, leaf_rules = False)
    rules, rule_labels = miner.fit(X, y)

    assert len(rules) > 0
    assert all(isinstance(r, Rule) for r in rules)
    assert rule_labels is None


def test_tree_miner_leaf_rules_subset_of_node_rules(two_class_dataset):
    X, y, _ = two_class_dataset
    node_rules, _ = TreeMiner(
        DecisionTree(max_depth = 2, random_state = 0), leaf_rules = False
    ).fit(X, y)
    leaf_rules, _ = TreeMiner(
        DecisionTree(max_depth = 2, random_state = 0), leaf_rules = True
    ).fit(X, y)
    assert len(leaf_rules) <= len(node_rules)
    assert len(leaf_rules) > 0


####################################################################################################
# ClassAssociationRuleMiner
####################################################################################################


def test_class_association_rule_miner_basic(two_class_dataset):
    X, y, _ = two_class_dataset
    X_df = pd.DataFrame(X)
    miner = ClassAssociationRuleMiner(
        min_support = 0.1, min_confidence = 0.5, max_length = 2, binning_method = "uniform",
        bin_params = {'n_bins': 4},
    )
    rules, rule_labels = miner.fit(X_df, y)

    assert len(rules) > 0
    assert len(rules) == len(rule_labels)
    assert all(isinstance(r, Rule) for r in rules)
    assert all(isinstance(lbl, set) and len(lbl) == 1 for lbl in rule_labels)
    assert miner.bin_df is not None


@pytest.mark.parametrize("kwargs", [
    dict(min_support = 1.5),
    dict(min_support = -0.1),
    dict(min_confidence = 1.5),
    dict(min_confidence = -0.1),
    dict(max_length = 0),
    dict(max_length = -1),
    dict(binning_method = "not-a-real-method"),
])
def test_class_association_rule_miner_invalid_params(kwargs):
    with pytest.raises(ValueError):
        ClassAssociationRuleMiner(**kwargs)


####################################################################################################
# FrequentItemsetMiner
####################################################################################################


def test_frequent_itemset_miner_basic(two_class_dataset):
    X, y, _ = two_class_dataset
    miner = FrequentItemsetMiner(min_support = 0.1, max_length = 2, binning_method = "uniform", bin_params = {'n_bins': 4})
    rules, rule_labels = miner.fit(X)

    assert len(rules) > 0
    assert all(isinstance(r, Rule) for r in rules)
    # FrequentItemsetMiner is documented to always return None for rule_labels.
    assert rule_labels is None


def test_frequent_itemset_miner_invalid_min_support():
    with pytest.raises(ValueError):
        FrequentItemsetMiner(min_support = 2.0)


def test_frequent_itemset_miner_default_bin_params_not_shared_across_instances(two_class_dataset):
    """bin_params has a mutable default ({'n_bins': 5}); two independently-constructed miners
    must not end up aliasing (and accidentally mutating) the same dict."""
    X, y, _ = two_class_dataset
    miner1 = FrequentItemsetMiner(min_support = 0.1)
    miner2 = FrequentItemsetMiner(min_support = 0.1)
    assert miner1.bin_params is not miner2.bin_params or miner1.bin_params == {'n_bins': 5}

    miner1.bin_params['n_bins'] = 999
    assert miner2.bin_params.get('n_bins') == 5
