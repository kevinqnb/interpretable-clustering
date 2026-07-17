import numpy as np
import pytest

from intercluster import Rule, Decision
from intercluster.rules import LinearCondition
from intercluster.decision_sets.objectives import CoverageMistakeObjective


####################################################################################################
# Helpers (mirrors tests/decision_sets/objectives/test_heap_distorted_greedy.py)
####################################################################################################


def make_condition(feature, threshold, direction):
    return LinearCondition(
        features=np.array([feature]),
        weights=np.array([1.0]),
        threshold=threshold,
        direction=direction,
    )


def interval_rule(feature, low, high):
    """Rule matching low < x[feature] <= high."""
    return Rule([
        make_condition(feature, low, 1),
        make_condition(feature, high, -1),
    ])


####################################################################################################
# Fixtures: same 3-cluster, 1D dataset as test_heap_distorted_greedy.py.
####################################################################################################


@pytest.fixture
def three_cluster_dataset():
    X = np.array([[0.0], [1.0], [2.0], [3.0],
                  [10.0], [11.0], [12.0], [13.0], [14.0],
                  [20.0], [21.0], [22.0], [23.0], [24.0], [25.0]])
    y = [{0}] * 4 + [{1}] * 5 + [{2}] * 6
    return X, y


def _fitted_objective(X, y, decision_set, **kwargs):
    obj = CoverageMistakeObjective(**kwargs)
    obj.initialize_data(X, y)
    obj.initialize_decision_set(decision_set)
    return obj


####################################################################################################
# set_lambda / compute_lambdas
####################################################################################################


def test_set_lambda_raises_before_initialization():
    obj = CoverageMistakeObjective(n_select=1, alpha_val=0.0)
    with pytest.raises(ValueError):
        obj.set_lambda(None)


def test_compute_lambdas_empty_decision_set_triggers_fallback(three_cluster_dataset):
    X, y = three_cluster_dataset
    obj = _fitted_objective(X, y, set(), n_select=1, alpha_val=0.0)
    assert len(obj.compute_lambdas()) == 0

    obj.set_lambda(None)
    assert obj.lambda_val == 0.0
    assert obj.selection_algorithm == 'lazy-greedy'


def test_set_lambda_all_infinite_ratios_triggers_fallback(three_cluster_dataset):
    """Two rules, each with a single, perfectly-correct (zero-mistake) decision: every
    per-rule ratio is g/0 = inf, and since each rule contributes only one decision, the
    per-rule *second*-best ratio is the default 0.0 for both -- so compute_lambdas returns
    [inf, inf] unfiltered (nothing to append), and set_lambda must fall back."""
    X, y = three_cluster_dataset
    decision_set = {
        Decision(interval_rule(0, -1.0, 3.5), 0),   # covers all of cluster 0, no mistakes
        Decision(interval_rule(0, 9.0, 12.5), 1),   # covers part of cluster 1, no mistakes
    }
    obj = _fitted_objective(X, y, decision_set, n_select=2, alpha_val=0.0)

    lambda_vals = obj.compute_lambdas()
    assert len(lambda_vals) == 2
    assert np.all(np.isinf(lambda_vals))

    obj.set_lambda(None)
    assert obj.lambda_val == 0.0
    assert obj.selection_algorithm == 'lazy-greedy'


def test_compute_lambdas_resolves_to_finite_minimum(three_cluster_dataset):
    """Adds a third, single-decision rule with a genuinely finite, nonzero reward/cost ratio
    (2 correct + 1 mistake -> ratio 2.0) alongside the two perfect (ratio=inf) rules from
    the fallback test above. Per-rule max ratios are [inf, inf, 2.0]; every rule here
    contributes only one decision, so every per-rule *second*-best stays at the default 0.0,
    keeping the global second_max_ratio at 0.0 -- so compute_lambdas keeps all three ratios
    unfiltered (nothing appended) and just sorts them, and set_lambda should resolve to the
    finite minimum (2.0) rather than fall back, since lambda_vals[0] is finite."""
    X, y = three_cluster_dataset
    partial_rule = interval_rule(0, 1.0, 10.0)  # covers x in {2, 3} (cluster 0) and {10} (cluster 1)
    decision_set = {
        Decision(interval_rule(0, -1.0, 3.5), 0),
        Decision(interval_rule(0, 9.0, 12.5), 1),
        Decision(partial_rule, 0),
    }
    obj = _fitted_objective(X, y, decision_set, n_select=3, alpha_val=0.0)

    lambda_vals = obj.compute_lambdas()
    assert list(lambda_vals) == sorted(lambda_vals)
    assert lambda_vals[0] == pytest.approx(2.0)
    assert np.isinf(lambda_vals[1])
    assert np.isinf(lambda_vals[2])

    obj.set_lambda(None)
    assert obj.lambda_val == pytest.approx(2.0)
    # No fallback triggered -- selection_algorithm keeps its original value.
    assert obj.selection_algorithm == 'distorted-greedy'


def test_set_lambda_explicit_value_bypasses_computation(three_cluster_dataset):
    """An explicit lambda_val should be used as-is, without needing data/decision-set
    initialization at all (the None-triggered auto-resolve path is what needs those)."""
    obj = CoverageMistakeObjective(n_select=1, alpha_val=0.0)
    obj.set_lambda(0.75)
    assert obj.lambda_val == 0.75


####################################################################################################
# save_precomputed / load_precomputed round-trip
####################################################################################################


def test_precomputed_round_trip_matches_fresh_fit(three_cluster_dataset, tmp_path):
    X, y = three_cluster_dataset
    decision_set = {
        Decision(interval_rule(0, -1.0, 3.5), 0),
        Decision(interval_rule(0, -1.0, 1.5), 0),
        Decision(interval_rule(0, 9.0, 14.5), 1),
        Decision(interval_rule(0, 9.0, 12.5), 1),
        Decision(interval_rule(0, 19.0, 25.5), 2),
        Decision(interval_rule(0, 3.0, 11.0), 0),
    }
    common_kwargs = dict(n_select=3, alpha_val=0.0, lambda_val=0.5)

    fresh = _fitted_objective(X, y, decision_set, **common_kwargs)
    fresh_selected = fresh.select()
    fresh_value = fresh.objective_value

    path = tmp_path / "precomputed.pkl.gz"
    fresh.save_precomputed(path)

    reloaded = CoverageMistakeObjective(precomputed_path = path, **common_kwargs)
    assert reloaded.precomputed is True
    reloaded.initialize_data(X, y)
    reloaded.initialize_decision_set(decision_set)
    reloaded_selected = reloaded.select()

    assert reloaded_selected == fresh_selected
    assert reloaded.objective_value == pytest.approx(fresh_value)


def test_precomputed_round_trip_uncompressed(three_cluster_dataset, tmp_path):
    X, y = three_cluster_dataset
    decision_set = {
        Decision(interval_rule(0, -1.0, 3.5), 0),
        Decision(interval_rule(0, 9.0, 14.5), 1),
        Decision(interval_rule(0, 19.0, 25.5), 2),
    }
    common_kwargs = dict(n_select=2, alpha_val=0.0, lambda_val=0.2)

    fresh = _fitted_objective(X, y, decision_set, **common_kwargs)
    fresh_selected = fresh.select()

    path = tmp_path / "precomputed_uncompressed.pkl"
    fresh.save_precomputed(path, compress = False)

    reloaded = CoverageMistakeObjective(precomputed_path = path, **common_kwargs)
    reloaded.initialize_data(X, y)
    reloaded.initialize_decision_set(decision_set)
    reloaded_selected = reloaded.select()

    assert reloaded_selected == fresh_selected
