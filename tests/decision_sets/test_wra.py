import numpy as np
import pytest
from intercluster.rules import LinearCondition
from intercluster import Rule
from intercluster.decision_sets import WRABaseline


####################################################################################################
# Helpers
####################################################################################################


def make_condition(feature, threshold, direction):
    return LinearCondition(
        features=np.array([feature]),
        weights=np.array([1.0]),
        threshold=threshold,
        direction=direction,
    )


def left_rule():
    """Rule that covers x[:,0] < 0."""
    return Rule([make_condition(0, 0.0, -1)])


def right_rule():
    """Rule that covers x[:,0] >= 0."""
    return Rule([make_condition(0, 0.0, 1)])


def upper_rule():
    """Rule that covers x[:,1] >= 0 (covers all points in the fixtures)."""
    return Rule([make_condition(1, -1.0, 1)])


####################################################################################################
# Fixtures
####################################################################################################


@pytest.fixture
def two_cluster_data():
    """
    20 points: left 10 in cluster 0, right 10 in cluster 1.
    left_rule perfectly covers cluster 0; right_rule perfectly covers cluster 1.
    """
    X = np.zeros((20, 2))
    X[:10, 0] = -1.0
    X[10:, 0] = 1.0
    y = [{0}] * 10 + [{1}] * 10
    return X, y


####################################################################################################
# Tests
####################################################################################################


def test_selects_n_rules(two_cluster_data):
    X, y = two_cluster_data
    model = WRABaseline(rules=[left_rule(), right_rule()], n_select=2)
    model.fit(X, y)
    assert len(model.decision_set) == 2


def test_selects_correct_labels(two_cluster_data):
    """left_rule should be assigned cluster 0, right_rule cluster 1."""
    X, y = two_cluster_data
    model = WRABaseline(rules=[left_rule(), right_rule()], n_select=2)
    model.fit(X, y)
    label_map = {d.rule: d.label for d in model.decision_set}
    assert label_map[left_rule()] == 0
    assert label_map[right_rule()] == 1


def test_n_select_limits_output(two_cluster_data):
    """Requesting fewer rules than available should cap the output."""
    X, y = two_cluster_data
    model = WRABaseline(rules=[left_rule(), right_rule()], n_select=1)
    model.fit(X, y)
    assert len(model.decision_set) == 1


def test_zero_wra_rule_excluded(two_cluster_data):
    """A rule that covers all points equally has WRA = 0 and should be excluded."""
    X, y = two_cluster_data
    # upper_rule covers all 20 points, split evenly across both clusters -> WRA = 0
    model = WRABaseline(rules=[upper_rule()], n_select=1)
    model.fit(X, y)
    assert len(model.decision_set) == 0


def test_best_cluster_per_rule(two_cluster_data):
    """Each rule is matched to its single best cluster, not both."""
    X, y = two_cluster_data
    # Only one rule, but it could in principle be paired with cluster 0 or 1.
    # WRA is higher for cluster 0 (left_rule covers cluster 0 perfectly).
    model = WRABaseline(rules=[left_rule()], n_select=2)
    model.fit(X, y)
    # Should produce at most 1 decision (one best cluster per rule)
    assert len(model.decision_set) == 1
    assert next(iter(model.decision_set)).label == 0


def test_uniform_weights_equivalent(two_cluster_data):
    """Explicit uniform weights should give the same result as no weights."""
    X, y = two_cluster_data
    n = X.shape[0]
    model_none = WRABaseline(rules=[left_rule(), right_rule()], n_select=2)
    model_ones = WRABaseline(
        rules=[left_rule(), right_rule()],
        n_select=2,
        weights=np.ones(n),
    )
    model_none.fit(X, y)
    model_ones.fit(X, y)

    labels_none = {(d.rule, d.label) for d in model_none.decision_set}
    labels_ones = {(d.rule, d.label) for d in model_ones.decision_set}
    assert labels_none == labels_ones


def test_weights_affect_selection():
    """
    Upweighting cluster 1 points should cause the rule covering cluster 1
    to rank above the rule covering cluster 0 when n_select=1.
    """
    X = np.zeros((20, 2))
    X[:10, 0] = -1.0   # left half
    X[10:, 0] = 1.0    # right half
    y = [{0}] * 10 + [{1}] * 10

    # Upweight the right half (cluster 1) heavily.
    weights = np.ones(20)
    weights[10:] = 10.0

    model = WRABaseline(rules=[left_rule(), right_rule()], n_select=1, weights=weights)
    model.fit(X, y)
    assert len(model.decision_set) == 1
    assert next(iter(model.decision_set)).label == 1
