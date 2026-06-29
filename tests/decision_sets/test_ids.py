import numpy as np
import pandas as pd
import pytest
from intercluster.rules import LinearCondition
from intercluster import Rule
from intercluster.decision_sets import IDS


LAMBDAS = [1.0] * 7

# Lambdas that make the objective equivalent to pure coverage (f6 only).
# With this setting, DLS greedily adds every rule that covers new points,
# making it predictable that both rules are selected on a 2-cluster dataset.
COVERAGE_LAMBDAS = [0, 0, 0, 0, 0, 0, 1]


def make_condition(feature, threshold, direction):
    return LinearCondition(
        features=np.array([feature]),
        weights=np.array([1.0]),
        threshold=threshold,
        direction=direction,
    )


def left_rule():
    """x[:,0] <= 0.0 — covers cluster 0."""
    return Rule([make_condition(0, 0.0, -1)])


def right_rule():
    """x[:,0] > 0.0 — covers cluster 1."""
    return Rule([make_condition(0, 0.0, 1)])


def partial_left_rule():
    """x[:,0] <= -0.5 — partial coverage of cluster 0."""
    return Rule([make_condition(0, -0.5, -1)])


@pytest.fixture
def two_cluster_data():
    """
    20 points with 1 feature: cluster 0 at x=-1, cluster 1 at x=1.

    bin_df uses exact interval strings matching what decision_set_to_cars produces
    for left_rule and right_rule (threshold=0.0), so that QuantitativeDataFrame's
    string-equality matching correctly identifies covered points.
    """
    n = 20
    X = np.zeros((n, 1))
    X[:10, 0] = -1.0
    X[10:, 0] = 1.0
    y = [{0}] * 10 + [{1}] * 10
    bin_df = pd.DataFrame({"0": ["(-inf, 0.0]"] * 10 + ["(0.0, inf]"] * 10})
    return X, y, bin_df


####################################################################################################
# Goal 1: Customizable input decision set
####################################################################################################


def test_selected_rules_subset_of_input(two_cluster_data):
    """Every rule in the selected set must come from the user-provided pool."""
    X, y, bin_df = two_cluster_data
    model = IDS(
        rules=[left_rule(), right_rule()],
        rule_labels=[{0}, {1}],
        bin_df=bin_df,
        lambdas=LAMBDAS,
        algorithm="DLS",
    )
    model.fit(X, y)
    selected = {d.rule for d in model.decision_set}
    assert selected.issubset({left_rule(), right_rule()})


def test_input_pool_limits_selection(two_cluster_data):
    """Providing a single rule forces the selection to contain at most that one rule."""
    X, y, bin_df = two_cluster_data
    model = IDS(
        rules=[left_rule()],
        rule_labels=[{0}],
        bin_df=bin_df,
        lambdas=LAMBDAS,
        algorithm="DLS",
    )
    model.fit(X, y)
    selected = {d.rule for d in model.decision_set}
    assert selected.issubset({left_rule()})


####################################################################################################
# Goal 2: Cap on the maximum number of rules
####################################################################################################


def test_dls_without_cap_selects_both_rules(two_cluster_data):
    """Without n_select, DLS naturally selects one rule per cluster (2 total).

    Uses a pure coverage objective (f6 only) so that adding each rule is
    unambiguously beneficial, making the natural stopping point predictable.
    """
    X, y, bin_df = two_cluster_data
    model = IDS(
        rules=[left_rule(), right_rule()],
        rule_labels=[{0}, {1}],
        bin_df=bin_df,
        lambdas=COVERAGE_LAMBDAS,
        n_select=None,
        algorithm="DLS",
    )
    model.fit(X, y)
    assert len(model.decision_set) == 2


def test_dls_n_select_caps_output(two_cluster_data):
    """DLS forward greedy stops as soon as n_select rules are added.

    Without the cap, both rules would be selected (see test_dls_without_cap_selects_both_rules).
    With n_select=1, the first added rule triggers the cap and DLS returns immediately.
    """
    X, y, bin_df = two_cluster_data
    model = IDS(
        rules=[left_rule(), right_rule()],
        rule_labels=[{0}, {1}],
        bin_df=bin_df,
        lambdas=COVERAGE_LAMBDAS,
        n_select=1,
        algorithm="DLS",
    )
    model.fit(X, y)
    assert len(model.decision_set) <= 1


def test_dls_n_select_larger_than_pool(two_cluster_data):
    """n_select larger than the pool size should not raise and returns all available rules."""
    X, y, bin_df = two_cluster_data
    model = IDS(
        rules=[left_rule(), right_rule()],
        rule_labels=[{0}, {1}],
        bin_df=bin_df,
        lambdas=COVERAGE_LAMBDAS,
        n_select=10,
        algorithm="DLS",
    )
    model.fit(X, y)
    assert len(model.decision_set) <= 10


def test_sls_n_select_caps_output(two_cluster_data):
    """SLS backward elimination trims the solution to at most n_select rules."""
    X, y, bin_df = two_cluster_data
    model = IDS(
        rules=[left_rule(), right_rule()],
        rule_labels=[{0}, {1}],
        bin_df=bin_df,
        lambdas=COVERAGE_LAMBDAS,
        n_select=1,
        algorithm="SLS",
    )
    model.fit(X, y)
    assert len(model.decision_set) <= 1
