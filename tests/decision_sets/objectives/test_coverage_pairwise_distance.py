import numpy as np
import pytest

from intercluster import Rule, Decision
from intercluster.rules import LinearCondition
from intercluster.decision_sets.objectives import (
    CoveragePairwiseDistanceObjective,
    TotalCoveragePairwiseDistanceObjective,
)


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


def build_decision_set(rules_and_labels):
    return {Decision(rule, label) for rule, label in rules_and_labels}


def run_select(obj_cls, X, y, decision_set, n_select, alpha_val, lambda_val, selection_algorithm, **kwargs):
    obj = obj_cls(
        n_select=n_select,
        alpha_val=alpha_val,
        lambda_val=lambda_val,
        selection_algorithm=selection_algorithm,
        **kwargs,
    )
    obj.initialize_data(X, y)
    obj.initialize_decision_set(decision_set)
    selected = obj.select()
    return obj, selected


####################################################################################################
# Fixtures: same 3-cluster, 1D dataset as test_heap_distorted_greedy.py, sized (4, 5, 6) so
# cluster sizes are all distinct -- important here since cost() is weighted by cluster size.
####################################################################################################


@pytest.fixture
def three_cluster_dataset():
    X = np.array([[0.0], [1.0], [2.0], [3.0],
                  [10.0], [11.0], [12.0], [13.0], [14.0],
                  [20.0], [21.0], [22.0], [23.0], [24.0], [25.0]])
    y = [{0}] * 4 + [{1}] * 5 + [{2}] * 6
    return X, y


@pytest.fixture
def three_cluster_decisions():
    rules_and_labels = [
        (interval_rule(0, -1.0, 3.5), 0),    # covers all of cluster 0 (4 points)
        (interval_rule(0, -1.0, 1.5), 0),    # covers part of cluster 0 (2 points)
        (interval_rule(0, 9.0, 14.5), 1),    # covers all of cluster 1 (5 points)
        (interval_rule(0, 9.0, 12.5), 1),    # covers part of cluster 1 (3 points)
        (interval_rule(0, 19.0, 25.5), 2),   # covers all of cluster 2 (6 points)
        (interval_rule(0, 3.0, 11.0), 0),    # crosses cluster boundary -- makes mistakes
    ]
    return build_decision_set(rules_and_labels)


####################################################################################################
# Heap vs. plain distorted-greedy equivalence -- CoveragePairwiseDistanceObjective isn't covered
# by test_heap_distorted_greedy.py at all (only CoverageMistakeObjective/CoverageCostObjective are).
####################################################################################################


class TestHeapEquivalence:

    def test_small_pool_no_ties(self, three_cluster_dataset, three_cluster_decisions):
        X, y = three_cluster_dataset
        params = dict(
            X=X, y=y, decision_set=three_cluster_decisions,
            n_select=3, alpha_val=0.0, lambda_val=0.1,
        )
        obj_plain, sel_plain = run_select(
            CoveragePairwiseDistanceObjective, selection_algorithm='distorted-greedy', **params
        )
        obj_heap, sel_heap = run_select(
            CoveragePairwiseDistanceObjective, selection_algorithm='heap-distorted-greedy', **params
        )
        assert sel_heap == sel_plain
        assert obj_heap.objective_value == pytest.approx(obj_plain.objective_value)

    def test_randomized_property(self):
        rng = np.random.default_rng(2024)
        centers = np.array([[0.0, 0.0], [10.0, 10.0], [20.0, 20.0]])

        for trial in range(5):
            n_per_cluster = 8
            X = np.concatenate([
                centers[c] + rng.normal(scale=0.5, size=(n_per_cluster, 2))
                for c in range(3)
            ])
            y = [{c} for c in range(3) for _ in range(n_per_cluster)]

            rules_and_labels = []
            for _ in range(15):
                feature = int(rng.integers(0, 2))
                cluster = int(rng.integers(0, 3))
                center_val = centers[cluster, feature]
                low = center_val - rng.uniform(1.0, 6.0)
                high = center_val + rng.uniform(1.0, 6.0)
                label = int(rng.integers(0, 3))
                rules_and_labels.append((interval_rule(feature, low, high), label))
            decision_set = build_decision_set(rules_and_labels)
            if len(decision_set) < 2:
                continue

            pool_size = len(decision_set)
            for n_select in {1, max(1, pool_size // 2), pool_size}:
                for lambda_val in [0.0, 0.1, 5.0]:
                    params = dict(
                        X=X, y=y, decision_set=decision_set,
                        n_select=n_select, alpha_val=0.0, lambda_val=lambda_val,
                    )
                    obj_plain, _ = run_select(
                        CoveragePairwiseDistanceObjective, selection_algorithm='distorted-greedy', **params
                    )
                    obj_heap, _ = run_select(
                        CoveragePairwiseDistanceObjective, selection_algorithm='heap-distorted-greedy', **params
                    )
                    assert obj_heap.objective_value == pytest.approx(
                        obj_plain.objective_value, rel=1e-9, abs=1e-9
                    ), f"trial={trial} n_select={n_select} lambda_val={lambda_val}"


####################################################################################################
# cost() correctness
####################################################################################################


def test_cost_matches_hand_computed_weighted_mistakes(three_cluster_dataset):
    """Rule covering x in (9.5, 10.5]: covers only point x=10 (cluster 1, size 5) correctly if
    labeled 1 (0 mistakes -> cost 0), but if labeled 0 it makes a single mistake on point 10 --
    weighted by (home_cluster_size - 1) + assigned_cluster_size = (5 - 1) + 4 = 8."""
    X, y = three_cluster_dataset
    rule = interval_rule(0, 9.5, 10.5)  # covers only point at x=10, in cluster 1 (size 5)
    decision_wrong = Decision(rule, 0)  # mislabel it as cluster 0 (size 4)

    obj = CoveragePairwiseDistanceObjective(n_select=1, alpha_val=0.0, lambda_val=0.0)
    obj.initialize_data(X, y)
    obj.initialize_decision_set({decision_wrong})

    info = obj.decision_info_dict[decision_wrong]
    cost = obj.cost({decision_wrong: info})
    assert cost == pytest.approx((5 - 1) + 4)

    decision_correct = Decision(rule, 1)
    obj2 = CoveragePairwiseDistanceObjective(n_select=1, alpha_val=0.0, lambda_val=0.0)
    obj2.initialize_data(X, y)
    obj2.initialize_decision_set({decision_correct})
    info2 = obj2.decision_info_dict[decision_correct]
    assert obj2.cost({decision_correct: info2}) == 0.0


def test_cost_includes_alpha_length_penalty(three_cluster_dataset):
    X, y = three_cluster_dataset
    rule = interval_rule(0, -1.0, 3.5)  # covers all of cluster 0, correctly labeled -> 0 mistakes
    decision = Decision(rule, 0)

    obj = CoveragePairwiseDistanceObjective(n_select=1, alpha_val=2.0, lambda_val=0.0)
    obj.initialize_data(X, y)
    obj.initialize_decision_set({decision})
    info = obj.decision_info_dict[decision]
    assert info['length'] == 2  # two conditions (lower + upper bound)
    assert obj.cost({decision: info}) == pytest.approx(2.0 * 2)


####################################################################################################
# Empty label set edge case: CoveragePairwiseDistanceObjective already guards
# `next(iter(s)) if len(s) > 0 else 0`; TotalCoveragePairwiseDistanceObjective now mirrors that
# guard (previously it would raise StopIteration on an empty label set).
####################################################################################################


def test_cost_handles_empty_label_set_coverage_pairwise():
    """The unlabeled point (empty label set) is never a true member of any cluster's
    membership matrix, so a rule covering it still counts it as a "mistake" for cost() --
    the guard only prevents a crash computing its *weight* (treating its home cluster as 0
    for that purpose alone), it does not exempt it from being a mistake when covered.
    cluster_sizes[0] = 2 (points 0, 2); weight_by_sample for the unlabeled point (guarded to
    label 0) = 2 - 1 = 1; assigned_cluster_size (label 0) = 2 -> mistake weight = 1 + 2 = 3."""
    X = np.array([[0.0], [1.0], [2.0]])
    y = [{0}, set(), {0}]  # point 1 is unlabeled
    rule = interval_rule(0, -1.0, 2.5)
    decision = Decision(rule, 0)

    obj = CoveragePairwiseDistanceObjective(n_select=1, alpha_val=0.0, lambda_val=0.0)
    obj.initialize_data(X, y)
    obj.initialize_decision_set({decision})
    info = obj.decision_info_dict[decision]
    # Should not raise (this is the guarded path), and gives a well-defined, finite cost.
    assert obj.cost({decision: info}) == 3.0


def test_cost_handles_empty_label_set_total_coverage_pairwise():
    X = np.array([[0.0], [1.0], [2.0]])
    y = [{0}, set(), {0}]
    rule = interval_rule(0, -1.0, 2.5)
    decision = Decision(rule, 0)

    obj = TotalCoveragePairwiseDistanceObjective(n_select=1, alpha_val=0.0, lambda_val=0.0)
    obj.initialize_data(X, y)
    obj.initialize_decision_set({decision})
    info = obj.decision_info_dict[decision]
    # Previously raised AttributeError (missing self.cluster_sizes init) before StopIteration
    # was even reachable; now fixed alongside the guard, so this should not raise either.
    assert obj.cost({decision: info}) == 3.0
