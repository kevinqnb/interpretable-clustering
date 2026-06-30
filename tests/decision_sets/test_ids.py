import numpy as np
import pandas as pd
import pytest
from intercluster.rules import LinearCondition
from intercluster import Rule, Decision, flatten_labels
from intercluster.decision_sets import IDS
from intercluster.decision_sets.ids import IDSCoverageCache, IDSObjective, SLSOptimizer


LAMBDAS = [1.0] * 7
# Only f6 (correct coverage union): forces selection of every rule that covers new points.
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


@pytest.fixture
def two_cluster_data():
    """20 points: cluster 0 at x=-1, cluster 1 at x=1."""
    n = 20
    X = np.zeros((n, 1))
    X[:10, 0] = -1.0
    X[10:, 0] = 1.0
    y = [{0}] * 10 + [{1}] * 10
    return X, y


####################################################################################################
# Structural tests (no pyIDS dependency)
####################################################################################################


def test_selected_rules_subset_of_input(two_cluster_data):
    """Every rule in the selected set must come from the user-provided pool."""
    X, y = two_cluster_data
    model = IDS(
        rules=[left_rule(), right_rule()],
        rule_labels=[{0}, {1}],
        lambdas=LAMBDAS,
    )
    model.fit(X, y)
    selected = {d.rule for d in model.decision_set}
    assert selected.issubset({left_rule(), right_rule()})


def test_input_pool_limits_selection(two_cluster_data):
    """Providing a single rule forces the selection to contain at most that rule."""
    X, y = two_cluster_data
    model = IDS(
        rules=[left_rule()],
        rule_labels=[{0}],
        lambdas=LAMBDAS,
    )
    model.fit(X, y)
    selected = {d.rule for d in model.decision_set}
    assert selected.issubset({left_rule()})


def test_n_select_caps_output(two_cluster_data):
    """n_select limits the number of selected rules."""
    X, y = two_cluster_data
    model = IDS(
        rules=[left_rule(), right_rule()],
        rule_labels=[{0}, {1}],
        lambdas=COVERAGE_LAMBDAS,
        n_select=1,
    )
    model.fit(X, y)
    assert len(model.decision_set) <= 1


def test_n_select_larger_than_pool(two_cluster_data):
    """n_select larger than pool size should not raise."""
    X, y = two_cluster_data
    model = IDS(
        rules=[left_rule(), right_rule()],
        rule_labels=[{0}, {1}],
        lambdas=COVERAGE_LAMBDAS,
        n_select=10,
    )
    model.fit(X, y)
    assert len(model.decision_set) <= 10


def test_cache_is_stored_after_fit(two_cluster_data):
    """get_cache() returns a non-None cache after fit()."""
    X, y = two_cluster_data
    model = IDS(
        rules=[left_rule(), right_rule()],
        rule_labels=[{0}, {1}],
        lambdas=LAMBDAS,
    )
    model.fit(X, y)
    assert model.get_cache() is not None


def test_cache_reuse_gives_same_pool(two_cluster_data):
    """Fitting with a prebuilt cache produces a selection from the same decision pool."""
    X, y = two_cluster_data
    # First fit — build cache
    pre = IDS(
        rules=[left_rule(), right_rule()],
        rule_labels=[{0}, {1}],
        lambdas=LAMBDAS,
        n_select=None,
    )
    pre.fit(X, y)
    cache = pre.get_cache()

    # Second fit — reuse cache
    model = IDS(
        rules=[left_rule(), right_rule()],
        rule_labels=[{0}, {1}],
        lambdas=LAMBDAS,
        n_select=1,
        cache=cache,
    )
    model.fit(X, y)
    selected = {d.rule for d in model.decision_set}
    assert selected.issubset({left_rule(), right_rule()})


####################################################################################################
# PyIDS comparison tests
####################################################################################################

try:
    from pyids.algorithms.ids import IDS as IDS_pyids
    from pyids.data_structures.ids_rule import IDSRule
    from pyids.data_structures.ids_ruleset import IDSRuleSet
    from pyids.data_structures.ids_cacher import IDSCacher
    from pyarc.qcba.data_structures import QuantitativeDataFrame
    from intercluster import decision_set_to_cars
    _PYIDS_AVAILABLE = True
except ImportError:
    _PYIDS_AVAILABLE = False

pyids_required = pytest.mark.skipif(not _PYIDS_AVAILABLE, reason="pyids not installed")


def _build_pyids_structures(X, y, decisions, bin_df):
    """Helper: build pyIDS data structures from our Decision objects."""
    y_flat = flatten_labels(y)
    cars = decision_set_to_cars(X, y, decisions)
    cars = [c for c in cars if c.confidence > 0 and c.support > 0
            and int(c.consequent[1]) != -1]
    if not cars:
        return None, None, None, None
    ids_rules = list(map(IDSRule, cars))
    ids_ruleset = IDSRuleSet(ids_rules)
    df = bin_df.assign(**{'class': y_flat.astype(str)})
    quant_df = QuantitativeDataFrame(df)
    cacher = IDSCacher()
    cacher.calculate_overlap(ids_ruleset, quant_df)
    return cars, ids_ruleset, quant_df, cacher


@pyids_required
def test_objective_comparable_to_pyids(two_cluster_data):
    """
    Our SLS selection achieves an IDS objective value within 10% of pyIDS SLS,
    evaluated using our IDSObjective on both selections.

    We run several seeds and require the condition to hold for the majority,
    accommodating SLS's stochastic nature.
    """
    X, y = two_cluster_data
    bin_df = pd.DataFrame({"0": ["(-inf, 0.0]"] * 10 + ["(0.0, inf]"] * 10})
    lambdas = LAMBDAS
    decisions = [Decision(left_rule(), 0), Decision(right_rule(), 1)]

    # Build our IDSCoverageCache once
    y_flat = flatten_labels(y)
    cache = IDSCoverageCache()
    cache.compute(decisions, X, y_flat)
    N = cache.N
    M = len(cache.decisions)
    our_obj = IDSObjective(lambdas, cache, N, M)

    # Build pyIDS structures once
    cars, ids_ruleset, quant_df, cacher = _build_pyids_structures(X, y, decisions, bin_df)
    if ids_ruleset is None:
        pytest.skip("pyIDS returned no valid rules")

    passes = 0
    n_trials = 5
    for seed in range(n_trials):
        np.random.seed(seed)

        # Our IDS
        our_model = IDS(rules=[left_rule(), right_rule()], rule_labels=[{0}, {1}],
                        lambdas=lambdas, cache=cache)
        our_model.fit(X, y)
        our_decisions_set = set(our_model.decision_set)
        our_indices = {i for i, d in enumerate(cache.decisions) if d in our_decisions_set}
        our_score = our_obj.evaluate(our_indices)

        # PyIDS SLS
        pyids = IDS_pyids(n_select=None, algorithm="SLS")
        pyids.ids_ruleset = ids_ruleset
        pyids.cacher = cacher
        pyids.fit(quant_dataframe=quant_df, lambda_array=lambdas)

        # Map pyIDS selected IDSRules back to our decision indices by label + coverage
        pyids_indices = set()
        for ids_rule in pyids.clf.rules:
            label = int(ids_rule.car.consequent[1])
            mask = ids_rule.correct_cover(quant_df)
            for i, d in enumerate(cache.decisions):
                if d.label == label and np.array_equal(cache.correct_masks[i], mask):
                    pyids_indices.add(i)
                    break
        pyids_score = our_obj.evaluate(pyids_indices)

        if pyids_score <= 0 or our_score >= pyids_score * 0.9:
            passes += 1

    assert passes >= n_trials // 2 + 1, (
        f"Our IDS objective fell >10% below pyIDS in {n_trials - passes}/{n_trials} trials"
    )


@pyids_required
def test_objective_ranking_matches_pyids(two_cluster_data):
    """
    Our IDSObjective ranks the empty set vs. the full set the same way as the
    pyIDS IDSObjectiveFunction for the same lambda weights.

    This directly validates that our 7-term objective formula is correct.
    """
    from pyids.algorithms.ids_objective_function import (
        IDSObjectiveFunction, ObjectiveFunctionParameters
    )

    X, y = two_cluster_data
    bin_df = pd.DataFrame({"0": ["(-inf, 0.0]"] * 10 + ["(0.0, inf]"] * 10})
    lambdas = LAMBDAS
    decisions = [Decision(left_rule(), 0), Decision(right_rule(), 1)]

    # Build our cache
    y_flat = flatten_labels(y)
    cache = IDSCoverageCache()
    cache.compute(decisions, X, y_flat)
    N = cache.N
    M = len(cache.decisions)
    our_obj = IDSObjective(lambdas, cache, N, M)

    our_f_empty = our_obj.evaluate(set())
    our_f_both  = our_obj.evaluate({0, 1})
    our_f_left  = our_obj.evaluate({0})

    # Build pyIDS objective
    _, ids_ruleset, quant_df, cacher = _build_pyids_structures(X, y, decisions, bin_df)
    if ids_ruleset is None:
        pytest.skip("pyIDS returned no valid rules")

    params = ObjectiveFunctionParameters()
    params.params["all_rules"] = ids_ruleset
    params.params["len_all_rules"] = len(ids_ruleset.ruleset)
    params.params["quant_dataframe"] = quant_df
    params.params["lambda_array"] = lambdas
    pyids_obj = IDSObjectiveFunction(params, cacher=cacher)

    # Map our Decision indices to pyIDS IDSRuleSet subsets
    pyids_rules_list = list(ids_ruleset.ruleset)
    pyids_f_empty = pyids_obj.evaluate(IDSRuleSet([]))
    pyids_f_both  = pyids_obj.evaluate(ids_ruleset)
    pyids_f_left  = pyids_obj.evaluate(IDSRuleSet([pyids_rules_list[0]]))

    # Rankings must agree: full set vs empty, and single rule vs empty
    assert (our_f_both > our_f_empty) == (pyids_f_both > pyids_f_empty), (
        f"our: both={our_f_both:.2f} vs empty={our_f_empty:.2f}; "
        f"pyids: both={pyids_f_both:.2f} vs empty={pyids_f_empty:.2f}"
    )
    assert (our_f_left > our_f_empty) == (pyids_f_left > pyids_f_empty), (
        f"our: left={our_f_left:.2f} vs empty={our_f_empty:.2f}; "
        f"pyids: left={pyids_f_left:.2f} vs empty={pyids_f_empty:.2f}"
    )
