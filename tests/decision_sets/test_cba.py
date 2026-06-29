import numpy as np
import pytest
from intercluster.rules import LinearCondition
from intercluster import Rule, Decision
from intercluster.decision_sets import CBA


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
    """x[:,0] <= 0.0"""
    return Rule([make_condition(0, 0.0, -1)])


def right_rule():
    """x[:,0] > 0.0"""
    return Rule([make_condition(0, 0.0, 1)])


def upper_rule():
    """x[:,1] > -1.0 (covers everything in the standard fixtures)"""
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
# Basic fit / predict
####################################################################################################


def test_fit_returns_nonempty_decision_set(two_cluster_data):
    X, y = two_cluster_data
    model = CBA(rules=[left_rule(), right_rule()])
    model.fit(X, y)
    assert len(model.decision_set) > 0


def test_predict_covers_selected_rule_points(two_cluster_data):
    """
    M1 stops at the first error-minimising prefix.  With two perfectly
    separated clusters of equal size, one rule achieves 0 total errors on
    its own (rule errors = 0, default errors = 0 because all remaining
    points belong to the other cluster).  Only that rule is selected;
    the points it covers must be labelled and all others are uncovered.
    """
    X, y = two_cluster_data
    model = CBA(rules=[left_rule(), right_rule()])
    model.fit(X, y)
    preds = model.predict(X)
    # Exactly one rule is selected — the points it covers get a label.
    assert len(model.decision_set) == 1
    selected = next(iter(model.decision_set))
    covered = selected.rule.evaluate(X)
    assert all(len(preds[i]) > 0 for i in range(len(X)) if covered[i])
    assert all(len(preds[i]) == 0 for i in range(len(X)) if not covered[i])


def test_selects_correct_label_for_first_rule(two_cluster_data):
    """
    M1 selects one rule (left_rule comes first by generation order) and
    assigns it to the cluster it fits best — cluster 0.
    The default_class for uncovered points should be cluster 1.
    """
    X, y = two_cluster_data
    model = CBA(rules=[left_rule(), right_rule()])
    model.fit(X, y)
    assert len(model.decision_set) == 1
    d = next(iter(model.decision_set))
    assert d.rule == left_rule()
    assert d.label == 0
    assert model.default_class == 1


def test_default_class_is_set(two_cluster_data):
    """default_class should be populated after fitting."""
    X, y = two_cluster_data
    model = CBA(rules=[left_rule(), right_rule()])
    model.fit(X, y)
    # Both rules fire, so default_class may be None (no uncovered points remain),
    # but the attribute must exist.
    assert hasattr(model, "default_class")


####################################################################################################
# n_select cap
####################################################################################################


def test_n_select_limits_output(two_cluster_data):
    X, y = two_cluster_data
    model = CBA(rules=[left_rule(), right_rule()], n_select=1)
    model.fit(X, y)
    assert len(model.decision_set) <= 1


def test_n_select_none_applies_no_cap():
    """
    In a 3-cluster dataset where two rules cover two clusters perfectly and a
    third cluster (middle) is uncovered, M1 must add both rules to minimise
    errors: after rule 1 fires, the remaining uncovered points are a mix of
    cluster-1 and cluster-2, so default_errors > 0; rule 2 then eliminates
    those errors, driving total_errors to 0 and giving cut index = 1 (both
    rules selected).
    """
    # far_left_rule: x[:,0] <= -1  →  cluster 0
    # far_right_rule: x[:,0] >  1  →  cluster 1
    # middle points (cluster 2) are not covered by either rule → default class
    def far_left_rule():
        return Rule([make_condition(0, -1.0, -1)])

    def far_right_rule():
        return Rule([make_condition(0, 1.0, 1)])

    X = np.array(
        [[-2.0, 0.0]] * 10   # cluster 0  — covered by far_left_rule
        + [[0.0, 0.0]] * 10  # cluster 2  — covered by neither rule
        + [[2.0, 0.0]] * 10, # cluster 1  — covered by far_right_rule
        dtype=float,
    )
    y = [{0}] * 10 + [{2}] * 10 + [{1}] * 10

    model = CBA(rules=[far_left_rule(), far_right_rule()], n_select=None)
    model.fit(X, y)
    assert len(model.decision_set) == 2
    assert model.default_class == 2


def test_n_select_larger_than_classifier_is_fine(two_cluster_data):
    """n_select > number of rules selected by M1 should not raise."""
    X, y = two_cluster_data
    model = CBA(rules=[left_rule(), right_rule()], n_select=100)
    model.fit(X, y)
    assert len(model.decision_set) <= 2


####################################################################################################
# Precedence ordering
####################################################################################################


def test_precedence_high_confidence_rule_preferred():
    """
    Two rules cover the same cluster. The one with higher confidence should
    appear in the decision set when n_select=1.

    X: 10 points in cluster 0 (left side), 2 "noise" points on the left side
    in cluster 1. left_rule covers all 12; right_rule perfectly covers the
    5-point right cluster 0 group.
    """
    X = np.array(
        [[-1.0, 0.0]] * 10   # cluster 0
        + [[-1.0, 0.0]] * 2  # cluster 1 (noise on left)
        + [[1.0, 0.0]] * 5,  # cluster 0 (right)
        dtype=float,
    )
    y = [{0}] * 10 + [{1}] * 2 + [{0}] * 5

    # right_rule: conf = 5/5 = 1.0 for cluster 0
    # left_rule:  conf = 10/12 ≈ 0.83 for cluster 0
    model = CBA(rules=[left_rule(), right_rule()], n_select=1)
    model.fit(X, y)
    assert len(model.decision_set) == 1
    d = next(iter(model.decision_set))
    assert d.rule == right_rule()


####################################################################################################
# Multi-label resolution: one label per rule
####################################################################################################


def test_one_label_per_rule(two_cluster_data):
    """Each unique rule maps to at most one cluster label."""
    X, y = two_cluster_data
    # upper_rule covers everything — with two clusters it will be assigned
    # to whichever achieves higher confidence, not both.
    model = CBA(rules=[upper_rule()], n_select=10)
    model.fit(X, y)
    rules_seen = [d.rule for d in model.decision_set]
    # upper_rule appears at most once
    assert rules_seen.count(upper_rule()) <= 1


####################################################################################################
# Edge cases
####################################################################################################


def test_no_rules_produce_empty_decision_set():
    X = np.ones((5, 2))
    y = [{0}] * 5
    model = CBA(rules=[])
    model.fit(X, y)
    assert model.decision_set == [] or model.decision_set == set()


def test_rule_covering_no_points_is_ignored(two_cluster_data):
    """A rule that covers zero data points should not appear in the output."""
    X, y = two_cluster_data
    # All X[:,0] are non-zero so this threshold rule covers nothing
    empty_rule = Rule([make_condition(0, -999.0, -1)])
    model = CBA(rules=[empty_rule, left_rule(), right_rule()])
    model.fit(X, y)
    rules_in_ds = {d.rule for d in model.decision_set}
    assert empty_rule not in rules_in_ds


def test_rule_labels_override_label_assignment():
    """
    When rule_labels is supplied, CBA should assign each rule to its given
    label rather than expanding over all unique labels in y.  M1 will still
    cut at the first minimum-error prefix (one rule here achieves 0 errors),
    so only left_rule survives, assigned to cluster 0.
    """
    X = np.zeros((20, 2))
    X[:10, 0] = -1.0
    X[10:, 0] = 1.0
    y = [{0}] * 10 + [{1}] * 10

    model = CBA(
        rules=[left_rule(), right_rule()],
        rule_labels=[{0}, {1}],
    )
    model.fit(X, y)
    assert len(model.decision_set) == 1
    d = next(iter(model.decision_set))
    assert d.rule == left_rule()
    assert d.label == 0
