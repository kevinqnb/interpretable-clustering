import numpy as np
import pandas as pd
import pytest

from intercluster import Rule
from intercluster.rules import LinearCondition
from intercluster.utils import filter_rules, entropy_bin, labels_format


def _interval_rule(feature, low, high):
    """Rule matching low < x[feature] <= high."""
    return Rule([
        LinearCondition(features=np.array([feature]), weights=np.array([1.0]), threshold=low, direction=1),
        LinearCondition(features=np.array([feature]), weights=np.array([1.0]), threshold=high, direction=-1),
    ])


####################################################################################################
# filter_rules
####################################################################################################


@pytest.fixture
def three_cluster_dataset():
    X = np.array([[0.0], [1.0], [2.0], [3.0],
                  [10.0], [11.0], [12.0], [13.0], [14.0],
                  [20.0], [21.0], [22.0], [23.0], [24.0], [25.0]])
    y = labels_format(np.array([0]*4 + [1]*5 + [2]*6))
    return X, y


def test_filter_rules_keeps_perfectly_confident_rule(three_cluster_dataset):
    X, y = three_cluster_dataset
    perfect_rule = _interval_rule(0, -1.0, 3.5)  # covers all 4 of cluster 0, nothing else
    filtered = filter_rules([perfect_rule], X, y, confidence=1.0, support=0.0)
    assert filtered == [perfect_rule]


def test_filter_rules_drops_low_confidence_rule(three_cluster_dataset):
    X, y = three_cluster_dataset
    # Covers 2 points of cluster 0 and 1 of cluster 1 -> majority-label confidence = 2/3.
    mixed_rule = _interval_rule(0, 1.5, 10.5)
    filtered_high_conf = filter_rules([mixed_rule], X, y, confidence=0.9)
    assert filtered_high_conf == []

    filtered_low_conf = filter_rules([mixed_rule], X, y, confidence=0.5)
    assert filtered_low_conf == [mixed_rule]


def test_filter_rules_drops_rule_with_no_coverage(three_cluster_dataset):
    X, y = three_cluster_dataset
    empty_rule = _interval_rule(0, 100.0, 200.0)  # covers nothing
    assert filter_rules([empty_rule], X, y, confidence=0.0) == []


def test_filter_rules_support_threshold(three_cluster_dataset):
    X, y = three_cluster_dataset
    # Covers exactly 2 of 15 points (support = 2/15 =~ 0.133).
    small_rule = _interval_rule(0, -1.0, 1.5)
    assert filter_rules([small_rule], X, y, support=0.1) == [small_rule]
    assert filter_rules([small_rule], X, y, support=0.5) == []


def test_filter_rules_rejects_multilabel_y():
    X = np.array([[0.0], [1.0]])
    y = [{0}, {0, 1}]
    with pytest.raises(ValueError):
        filter_rules([_interval_rule(0, -1.0, 2.0)], X, y)


####################################################################################################
# entropy_bin
####################################################################################################


def test_entropy_bin_returns_interval_dataframe_spanning_full_range():
    X = np.array([
        [0.0, 10.0],
        [1.0, 11.0],
        [2.0, 12.0],
        [8.0, 2.0],
        [9.0, 1.0],
        [10.0, 0.0],
    ])
    y = labels_format(np.array([0, 0, 0, 1, 1, 1]))

    bin_df = entropy_bin(X, y, random_state=0)

    assert isinstance(bin_df, pd.DataFrame)
    assert bin_df.shape == (6, 2)
    for col in bin_df.columns:
        for interval in bin_df[col]:
            assert isinstance(interval, pd.Interval)
    # Every column's bins must jointly cover -inf to inf so every value in X (and any
    # future value at inference time) always falls into exactly one bin.
    for col in bin_df.columns:
        lows = sorted(iv.left for iv in bin_df[col].unique())
        highs = sorted(iv.right for iv in bin_df[col].unique())
        assert lows[0] == -np.inf
        assert highs[-1] == np.inf


def test_entropy_bin_rejects_multilabel_y():
    X = np.array([[0.0], [1.0], [2.0]])
    y = [{0}, {0, 1}, {1}]
    with pytest.raises(ValueError):
        entropy_bin(X, y)
