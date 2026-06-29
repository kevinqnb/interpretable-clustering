import numpy as np
import pytest
from sklearn import datasets, preprocessing
from intercluster.decision_sets import CN2


####################################################################################################
# Fixtures
####################################################################################################


@pytest.fixture
def blob_data():
    """Three well-separated Gaussian blobs with k-means-style singleton labels."""
    X, _ = datasets.make_blobs(
        n_samples=150, centers=3, cluster_std=0.5, random_state=0
    )
    X = preprocessing.MinMaxScaler().fit_transform(X)
    y = [{0}] * 50 + [{1}] * 50 + [{2}] * 50
    return X, y


@pytest.fixture
def two_cluster_data():
    """60 points perfectly separated on feature 0; two singleton-label clusters."""
    X = np.zeros((60, 2))
    X[:30, 0] = -2.0
    X[30:, 0] = 2.0
    y = [{0}] * 30 + [{1}] * 30
    return X, y


####################################################################################################
# Init validation
####################################################################################################


def test_invalid_n_select_raises():
    with pytest.raises((ValueError, AssertionError)):
        CN2(n_select=0)


def test_invalid_beam_width_raises():
    with pytest.raises((ValueError, AssertionError)):
        CN2(beam_width=0)


def test_invalid_min_covered_examples_raises():
    with pytest.raises((ValueError, AssertionError)):
        CN2(min_covered_examples=-1)


def test_invalid_max_rule_conditions_raises():
    with pytest.raises((ValueError, AssertionError)):
        CN2(max_rule_conditions=0)


####################################################################################################
# Basic fit
####################################################################################################


def test_fit_produces_nonempty_decision_set(blob_data):
    X, y = blob_data
    model = CN2()
    model.fit(X, y)
    assert len(model.decision_set) > 0


def test_decision_set_is_list_after_fit(blob_data):
    X, y = blob_data
    model = CN2()
    model.fit(X, y)
    assert isinstance(model.decision_set, list)


def test_max_rule_length_is_set_after_fit(blob_data):
    X, y = blob_data
    model = CN2()
    model.fit(X, y)
    expected = max(len(d.rule) for d in model.decision_set)
    assert model.max_rule_length == expected


def test_all_labels_are_from_input_clusters(blob_data):
    """Every rule label must be one of the cluster IDs present in y."""
    X, y = blob_data
    model = CN2()
    model.fit(X, y)
    valid_labels = {next(iter(yi)) for yi in y}
    for d in model.decision_set:
        assert d.label in valid_labels


def test_predict_returns_list_of_sets(blob_data):
    X, y = blob_data
    model = CN2()
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(X)
    assert all(isinstance(p, set) for p in preds)


def test_covered_points_receive_labels(two_cluster_data):
    """Points covered by at least one rule must have a non-empty prediction set."""
    X, y = two_cluster_data
    model = CN2()
    model.fit(X, y)
    preds = model.predict(X)
    for i, d in enumerate(model.decision_set):
        covered = np.where(d.rule.evaluate(X))[0]
        for j in covered:
            assert len(preds[j]) > 0


####################################################################################################
# n_select cap
####################################################################################################


def test_n_select_limits_rule_count(blob_data):
    X, y = blob_data
    for cap in [1, 3, 5]:
        model = CN2(n_select=cap)
        model.fit(X, y)
        assert len(model.decision_set) <= cap


def test_n_select_larger_than_total_is_fine(blob_data):
    """n_select greater than the number of rules found should not raise."""
    X, y = blob_data
    model = CN2(n_select=10_000)
    model.fit(X, y)
    assert len(model.decision_set) > 0


def test_n_select_none_produces_more_rules_than_small_cap(blob_data):
    """Removing the cap should yield at least as many rules as a tight cap."""
    X, y = blob_data
    capped = CN2(n_select=3)
    capped.fit(X, y)

    uncapped = CN2(n_select=None)
    uncapped.fit(X, y)

    assert len(uncapped.decision_set) >= len(capped.decision_set)


####################################################################################################
# max_rule_conditions
####################################################################################################


def test_rules_respect_max_rule_conditions(blob_data):
    """No rule in the decision set may exceed max_rule_conditions conditions."""
    X, y = blob_data
    cap = 2
    model = CN2(max_rule_conditions=cap)
    model.fit(X, y)
    for d in model.decision_set:
        assert len(d.rule) <= cap


####################################################################################################
# Input validation in fit
####################################################################################################


def test_y_none_raises(blob_data):
    X, _ = blob_data
    with pytest.raises(ValueError, match="labels"):
        CN2().fit(X, y=None)


def test_multilabel_input_raises(blob_data):
    X, y = blob_data
    y_bad = [{0, 1}] + y[1:]  # first point has two labels
    with pytest.raises(ValueError, match="multi-label"):
        CN2().fit(X, y_bad)


def test_outlier_points_are_filtered(blob_data):
    """Points with empty label sets should be silently excluded, not cause an error."""
    X, y = blob_data
    y_with_outliers = [set()] * 10 + y[10:]  # first 10 points are unlabeled
    model = CN2()
    model.fit(X, y_with_outliers)
    assert len(model.decision_set) > 0


def test_all_outliers_raises(blob_data):
    X, _ = blob_data
    y_all_outliers = [set()] * len(X)
    with pytest.raises(ValueError):
        CN2().fit(X, y_all_outliers)
