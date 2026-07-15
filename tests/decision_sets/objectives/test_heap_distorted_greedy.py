import sys
from pathlib import Path

import numpy as np
import pytest

from intercluster import Rule, Decision
from intercluster.rules import LinearCondition
from intercluster.utils import labels_format, unique_labels
from intercluster.decision_trees import DecisionTree
from intercluster.decision_sets import PEC
from intercluster.decision_sets.mining import TreeMiner
from intercluster.decision_sets.objectives import (
    CoverageMistakeObjective,
    CoverageCostObjective,
)

# Mirrors the sys.path bootstrap in experiments/*.py: `data/` is a top-level, non-installed
# package (not part of the `intercluster` distribution), so it's only importable once the repo
# root is on sys.path.
_HERE = Path(__file__).resolve()
_PROJECT_ROOT = next((p for p in _HERE.parents if (p / "data").is_dir()), None)
if _PROJECT_ROOT is not None and str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data.preprocessing import load_preprocessed_ansio  # noqa: E402


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


def interval_rule(feature, low, high):
    """Rule matching low < x[feature] <= high."""
    return Rule([
        make_condition(feature, low, 1),
        make_condition(feature, high, -1),
    ])


def run_select(obj_cls, X, y, decision_set, n_select, alpha_val, lambda_val, selection_algorithm, **kwargs):
    """Builds and runs an Objective of the given class/algorithm against identical inputs."""
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
# Fixtures: a small 1D, 3-cluster dataset with a varied rule pool.
####################################################################################################


@pytest.fixture
def three_cluster_dataset():
    """15 points along one axis in 3 well-separated clusters of *different* sizes (4, 5, 6), so
    that "cover the whole cluster" rules for different clusters never have identical coverage."""
    X = np.array([[0.0], [1.0], [2.0], [3.0],
                  [10.0], [11.0], [12.0], [13.0], [14.0],
                  [20.0], [21.0], [22.0], [23.0], [24.0], [25.0]])
    y = [{0}] * 4 + [{1}] * 5 + [{2}] * 6
    return X, y


@pytest.fixture
def three_cluster_decisions():
    """A pool of rules with varied coverage/cost -- no two share the same (coverage, cost, label),
    so distorted_greedy_select never hits an exact scoring tie against this pool."""
    rules_and_labels = [
        (interval_rule(0, -1.0, 3.5), 0),    # covers all of cluster 0 (4 points)
        (interval_rule(0, -1.0, 1.5), 0),    # covers part of cluster 0 (2 points)
        (interval_rule(0, 9.0, 14.5), 1),    # covers all of cluster 1 (5 points)
        (interval_rule(0, 9.0, 12.5), 1),    # covers part of cluster 1 (3 points)
        (interval_rule(0, 19.0, 25.5), 2),   # covers all of cluster 2 (6 points)
        (interval_rule(0, 3.0, 11.0), 0),    # crosses cluster boundary -- makes mistakes
    ]
    return {Decision(rule, label) for rule, label in rules_and_labels}


def build_decision_set(rules_and_labels):
    return {Decision(rule, label) for rule, label in rules_and_labels}


@pytest.fixture(scope="module")
def aniso_dataset():
    """The repo's small (1000-point, 2D, 8-blob) synthetic 'aniso' dataset (see
    data/preprocessing.py:load_preprocessed_ansio and CLAUDE.md's Data section), with a modest
    pool of mined rules -- one Decision per (rule, cluster label) pair, mirroring how PEC's
    set_labels builds a decision set when rule_labels=None. Real mined rules and a real dataset
    exercise coverage/cost patterns a hand-built pool doesn't: this is what caught a genuine
    correctness bug in heap_distorted_greedy_select's discard branch (see git history) that no
    hand-built test case reproduced."""
    X, y_raw, _, _ = load_preprocessed_ansio()
    y = labels_format(y_raw)
    n_labels = len(set(y_raw.tolist()))
    cluster_centers = np.array([X[y_raw == c].mean(axis=0) for c in range(n_labels)])

    tree = DecisionTree(max_depth=4, random_state=0)
    rules, _ = TreeMiner(tree, leaf_rules=False).fit(X, y)

    ulabels = unique_labels(y)
    decision_set = {Decision(rule, u) for rule in rules for u in ulabels}
    return X, y, cluster_centers, decision_set


####################################################################################################
# Objective-level equivalence tests
####################################################################################################


class TestHeapDistortedGreedyEquivalence:

    def test_small_pool_no_ties(self, three_cluster_dataset, three_cluster_decisions):
        """With a non-tied pool, heap-distorted-greedy should select exactly the same set as
        the plain distorted-greedy scan, and reach the same objective value."""
        X, y = three_cluster_dataset
        params = dict(
            X=X, y=y, decision_set=three_cluster_decisions,
            n_select=3, alpha_val=0.0, lambda_val=0.5,
        )
        obj_plain, sel_plain = run_select(
            CoverageMistakeObjective, selection_algorithm='distorted-greedy', **params
        )
        obj_heap, sel_heap = run_select(
            CoverageMistakeObjective, selection_algorithm='heap-distorted-greedy', **params
        )

        assert sel_heap == sel_plain
        assert obj_heap.objective_value == pytest.approx(obj_plain.objective_value)
        assert obj_heap.n_available_decisions == obj_plain.n_available_decisions

    def test_ties_match_objective_value(self, three_cluster_dataset):
        """Decisions with literally identical coverage/cost can legitimately be resolved
        differently by the two algorithms (see plan notes on tie-break direction), so only
        objective value and solution size are compared here, not set equality."""
        X, y = three_cluster_dataset
        rule = interval_rule(0, -1.0, 3.5)
        # Three distinct Rule objects (different condition instances) that are functionally
        # identical in coverage/cost/label -- a genuine tie in (a, b) at every iteration.
        tied_rules_and_labels = [
            (interval_rule(0, -1.0, 3.5), 0),
            (interval_rule(0, -1.0, 3.5 + 1e-12), 0),
            (interval_rule(0, -1.0 - 1e-12, 3.5), 0),
        ]
        decision_set = build_decision_set(tied_rules_and_labels)

        params = dict(X=X, y=y, decision_set=decision_set, n_select=2, alpha_val=0.0, lambda_val=0.1)
        obj_plain, sel_plain = run_select(
            CoverageMistakeObjective, selection_algorithm='distorted-greedy', **params
        )
        obj_heap, sel_heap = run_select(
            CoverageMistakeObjective, selection_algorithm='heap-distorted-greedy', **params
        )

        assert len(sel_heap) == len(sel_plain)
        assert obj_heap.objective_value == pytest.approx(obj_plain.objective_value)

    def test_high_lambda_zero_gate(self, three_cluster_dataset, three_cluster_decisions):
        """A very high lambda pushes every decision's initial gate negative; both algorithms
        should agree on the (empty) result and on n_available_decisions. alpha_val > 0 is
        needed here so that every decision has strictly positive cost (via its rule length
        penalty) even when it makes no coverage mistakes -- otherwise a zero-mistake,
        zero-length-penalty rule would have zero cost and its gate could never be driven
        negative by lambda alone."""
        X, y = three_cluster_dataset
        params = dict(
            X=X, y=y, decision_set=three_cluster_decisions,
            n_select=3, alpha_val=1.0, lambda_val=1000.0,
        )
        obj_plain, sel_plain = run_select(
            CoverageMistakeObjective, selection_algorithm='distorted-greedy', **params
        )
        obj_heap, sel_heap = run_select(
            CoverageMistakeObjective, selection_algorithm='heap-distorted-greedy', **params
        )

        assert sel_heap == sel_plain == set()
        assert obj_heap.n_available_decisions == obj_plain.n_available_decisions == 0
        assert obj_heap.objective_value == pytest.approx(obj_plain.objective_value)

    def test_n_select_one(self, three_cluster_dataset, three_cluster_decisions):
        """n_select=1 exercises t_0 = (1 - 1/1)**(1-1) == 0.0**0 == 1.0."""
        X, y = three_cluster_dataset
        params = dict(
            X=X, y=y, decision_set=three_cluster_decisions,
            n_select=1, alpha_val=0.0, lambda_val=0.2,
        )
        obj_plain, sel_plain = run_select(
            CoverageMistakeObjective, selection_algorithm='distorted-greedy', **params
        )
        obj_heap, sel_heap = run_select(
            CoverageMistakeObjective, selection_algorithm='heap-distorted-greedy', **params
        )

        assert sel_heap == sel_plain
        assert len(sel_heap) <= 1
        assert obj_heap.objective_value == pytest.approx(obj_plain.objective_value)

    def test_more_decisions_than_n_select(self, three_cluster_dataset, three_cluster_decisions):
        """The ordinary case: pool bigger than the selection budget."""
        X, y = three_cluster_dataset
        params = dict(
            X=X, y=y, decision_set=three_cluster_decisions,
            n_select=3, alpha_val=0.05, lambda_val=0.3,
        )
        obj_plain, sel_plain = run_select(
            CoverageMistakeObjective, selection_algorithm='distorted-greedy', **params
        )
        obj_heap, sel_heap = run_select(
            CoverageMistakeObjective, selection_algorithm='heap-distorted-greedy', **params
        )

        assert len(three_cluster_decisions) > 3
        assert sel_heap == sel_plain
        assert obj_heap.objective_value == pytest.approx(obj_plain.objective_value)

    def test_fewer_available_than_n_select(self, three_cluster_dataset):
        """Only 2 decisions ever pass the positivity gate, but n_select=5: exercises the
        heap-exhaustion early-exit path without raising."""
        X, y = three_cluster_dataset
        decision_set = build_decision_set([
            (interval_rule(0, -1.0, 3.5), 0),
            (interval_rule(0, 9.0, 14.5), 1),
        ])
        params = dict(
            X=X, y=y, decision_set=decision_set,
            n_select=5, alpha_val=0.0, lambda_val=0.1,
        )
        obj_plain, sel_plain = run_select(
            CoverageMistakeObjective, selection_algorithm='distorted-greedy', **params
        )
        obj_heap, sel_heap = run_select(
            CoverageMistakeObjective, selection_algorithm='heap-distorted-greedy', **params
        )

        assert sel_heap == sel_plain == decision_set
        assert obj_heap.objective_value == pytest.approx(obj_plain.objective_value)

    def test_randomized_property(self):
        """Randomized property test: for many small synthetic 3-cluster decision pools and a
        spread of n_select/lambda_val values, heap-distorted-greedy must reach the same
        objective value as distorted-greedy."""
        rng = np.random.default_rng(12345)
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
                        cluster_centers=centers,
                    )
                    obj_plain, _ = run_select(
                        CoverageCostObjective, selection_algorithm='distorted-greedy', **params
                    )
                    obj_heap, _ = run_select(
                        CoverageCostObjective, selection_algorithm='heap-distorted-greedy', **params
                    )
                    assert obj_heap.objective_value == pytest.approx(
                        obj_plain.objective_value, rel=1e-9, abs=1e-9
                    ), f"trial={trial} n_select={n_select} lambda_val={lambda_val}"

    @pytest.mark.parametrize("n_select", [1, 3, 8, 20, 30])
    @pytest.mark.parametrize("lambda_val", [0.0, 0.1, 0.5, 2.0, 50.0])
    def test_aniso_dataset_equivalence(self, aniso_dataset, n_select, lambda_val):
        """Real dataset, real mined rules: covers both a small n_select (few iterations, t_i
        reaches ~1 quickly) and n_select close to or above the pool's per-rule budget (many
        iterations at small t_i) -- the latter is what originally caught a bug where a decision
        popped with a non-positive *distorted* score was discarded forever, even though its
        *undistorted* score (distorted_greedy_select's real permanent-discard criterion) was
        still positive and it should have stayed eligible for a later, larger t_i."""
        X, y, cluster_centers, decision_set = aniso_dataset
        params = dict(
            X=X, y=y, decision_set=decision_set,
            n_select=n_select, alpha_val=0.0, lambda_val=lambda_val,
            cluster_centers=cluster_centers,
        )
        obj_plain, sel_plain = run_select(
            CoverageCostObjective, selection_algorithm='distorted-greedy', **params
        )
        obj_heap, sel_heap = run_select(
            CoverageCostObjective, selection_algorithm='heap-distorted-greedy', **params
        )

        assert obj_heap.objective_value == pytest.approx(obj_plain.objective_value, abs=1e-6)
        assert obj_heap.n_available_decisions == obj_plain.n_available_decisions
        # lambda_val == 0.0 produces genuine (a, b) ties across many same-label decisions with
        # equal marginal gain, which the two algorithms may break differently (see the tie-break
        # direction note above) while still reaching the same objective value and solution size.
        if lambda_val != 0.0:
            assert sel_heap == sel_plain
        else:
            assert len(sel_heap) == len(sel_plain)


####################################################################################################
# PEC-level integration tests
####################################################################################################


class TestPECHeapDistortedGreedy:

    def test_pec_end_to_end(self, three_cluster_dataset):
        X, y = three_cluster_dataset
        cluster_centers = np.array([[1.5], [12.0], [22.5]])
        rules = [
            interval_rule(0, -1.0, 3.5),
            interval_rule(0, 9.0, 14.5),
            interval_rule(0, 19.0, 25.5),
        ]
        pec = PEC(
            rules=rules,
            objective_type='coverage-cost',
            selection_algorithm='heap-distorted-greedy',
            n_select=3,
            alpha_val=0.0,
            lambda_val=0.1,
            cluster_centers=cluster_centers,
        )
        pec.fit(X, y)

        assert isinstance(pec.decision_set, list)
        assert all(isinstance(d, Decision) for d in pec.decision_set)
        assert {d.rule for d in pec.decision_set}.issubset(set(rules))
        assert not np.isnan(pec.n_available_decisions)

    def test_pec_matches_distorted_greedy(self, three_cluster_dataset):
        X, y = three_cluster_dataset
        cluster_centers = np.array([[1.5], [12.0], [22.5]])
        rules = [
            interval_rule(0, -1.0, 3.5),
            interval_rule(0, 9.0, 14.5),
            interval_rule(0, 19.0, 25.5),
            interval_rule(0, 3.0, 11.0),
        ]
        common_kwargs = dict(
            rules=rules,
            objective_type='coverage-cost',
            n_select=3,
            alpha_val=0.0,
            lambda_val=0.1,
            cluster_centers=cluster_centers,
        )

        pec_plain = PEC(selection_algorithm='distorted-greedy', **common_kwargs)
        pec_plain.fit(X, y)

        pec_heap = PEC(selection_algorithm='heap-distorted-greedy', **common_kwargs)
        pec_heap.fit(X, y)

        assert pec_heap.objective.objective_value == pytest.approx(pec_plain.objective.objective_value)

    def test_pec_rejects_unknown_selection_algorithm(self, three_cluster_dataset):
        X, y = three_cluster_dataset
        with pytest.raises(AssertionError):
            PEC(
                rules=[interval_rule(0, -1.0, 3.5)],
                objective_type='coverage-cost',
                selection_algorithm='not-a-real-algorithm',
                n_select=1,
                cluster_centers=np.array([[1.5], [12.0], [22.5]]),
            )
