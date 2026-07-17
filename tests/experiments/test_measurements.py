import numpy as np
import pytest
from sklearn.metrics.pairwise import pairwise_distances
from intercluster.measurements import (
    TotalCoverage,
    TotalCoverageSet,
    ClusterCoverage,
    ClusterCoverageSet,
    Overlap,
    Mistakes,
    ClusteringCost,
    RuleClusteringCost,
    PairwiseDistance,
    RulePairwiseDistance,
    Silhouette,
)


####################################################################################################
# Shared fixture-style setup, reused across most measurement classes below.
#
# 4 points, 2 clusters, 2 rules. Ground truth (baseline): points 0,1 -> cluster 0;
# points 2,3 -> cluster 1. Rules: rule 0 covers points 0,1,2 and is assigned to cluster 0
# (so point 2 is a "mistake"); rule 1 covers point 3 and is assigned to cluster 1.
# The predicted data_to_cluster_assignment (D) is what those rules produce: points 0,1,2
# land in cluster 0, point 3 lands in cluster 1.
####################################################################################################

X = np.array([[0.], [0.], [10.], [10.]])
BASELINE = np.array([
    [1, 0],
    [1, 0],
    [0, 1],
    [0, 1],
])
DATA_TO_RULE = np.array([
    [1, 0],
    [1, 0],
    [1, 0],
    [0, 1],
])
RULE_TO_CLUSTER = np.array([
    [1, 0],
    [0, 1],
])
DATA_TO_CLUSTER = np.array([
    [1, 0],
    [1, 0],
    [1, 0],
    [0, 1],
])


####################################################################################################


def test_total_coverage():
    fn = TotalCoverage()
    assert fn(DATA_TO_RULE, RULE_TO_CLUSTER, DATA_TO_CLUSTER) == 4
    assert np.isnan(fn(DATA_TO_RULE, RULE_TO_CLUSTER, None))

    weighted = TotalCoverage(weights = np.array([2., 1., 1., 1.]))
    assert weighted(DATA_TO_RULE, RULE_TO_CLUSTER, DATA_TO_CLUSTER) == 5


def test_total_coverage_set():
    fn = TotalCoverageSet()
    assert fn(DATA_TO_RULE, RULE_TO_CLUSTER, DATA_TO_CLUSTER) == [True, True, True, True]
    assert np.isnan(fn(DATA_TO_RULE, RULE_TO_CLUSTER, None))


def test_cluster_coverage():
    # Point 2 was mis-clustered by the rules (rule 0's cluster 0 vs its true cluster 1),
    # so it doesn't count as "covered within its true cluster".
    fn = ClusterCoverage(baseline_assignment = BASELINE)
    assert fn(DATA_TO_RULE, RULE_TO_CLUSTER, DATA_TO_CLUSTER) == 3
    assert np.isnan(fn(DATA_TO_RULE, RULE_TO_CLUSTER, None))


def test_cluster_coverage_set():
    fn = ClusterCoverageSet(baseline_assignment = BASELINE)
    assert fn(DATA_TO_RULE, RULE_TO_CLUSTER, DATA_TO_CLUSTER) == [True, True, False, True]


def test_overlap():
    fn = Overlap()
    # No point is assigned to more than one cluster here, so overlap is exactly 1.
    assert fn(DATA_TO_RULE, RULE_TO_CLUSTER, DATA_TO_CLUSTER) == 1.0
    assert np.isnan(fn(DATA_TO_RULE, RULE_TO_CLUSTER, None))


def test_mistakes():
    # Rule 0 covers 3 points (0,1,2) but only 2 of them (0,1) are truly in its assigned
    # cluster 0 -> 1 mistake. Rule 1 covers 1 point (3), correctly -> 0 mistakes.
    fn = Mistakes(baseline_assignment = BASELINE)
    assert fn(DATA_TO_RULE, RULE_TO_CLUSTER, DATA_TO_CLUSTER) == 1
    assert np.isnan(fn(None, RULE_TO_CLUSTER, DATA_TO_CLUSTER))
    assert np.isnan(fn(DATA_TO_RULE, None, DATA_TO_CLUSTER))

    with pytest.raises(AssertionError):
        # A rule assigned to two clusters violates the "exactly one cluster" contract.
        fn(DATA_TO_RULE, np.array([[1, 1], [0, 1]]), DATA_TO_CLUSTER)


def test_clustering_cost():
    fn = ClusteringCost(data = X)
    assert np.isclose(fn(DATA_TO_RULE, RULE_TO_CLUSTER, DATA_TO_CLUSTER), 40/3)
    assert np.isnan(fn(DATA_TO_RULE, RULE_TO_CLUSTER, None))

    fn_medians = ClusteringCost(data = X, method = 'kmedians')
    assert fn_medians(DATA_TO_RULE, RULE_TO_CLUSTER, DATA_TO_CLUSTER) == 10.0

    fn_normalized = ClusteringCost(data = X, normalize = True)
    assert np.isclose(fn_normalized(DATA_TO_RULE, RULE_TO_CLUSTER, DATA_TO_CLUSTER), 40/3/4)

    with pytest.raises(ValueError):
        ClusteringCost(data = X, method = 'bad-method')


def test_rule_clustering_cost():
    # With centers auto-computed from data_to_cluster_assignment, this should agree
    # exactly with ClusteringCost on the same (rule-consistent) partition.
    fn_auto = RuleClusteringCost(data = X)
    assert np.isclose(
        fn_auto(DATA_TO_RULE, RULE_TO_CLUSTER, DATA_TO_CLUSTER), 40/3
    )

    fixed_centers = np.array([[0.], [10.]])
    fn_fixed = RuleClusteringCost(data = X, cluster_centers = fixed_centers)
    assert fn_fixed(DATA_TO_RULE, RULE_TO_CLUSTER, DATA_TO_CLUSTER) == 10.0
    assert np.isnan(fn_fixed(None, RULE_TO_CLUSTER, DATA_TO_CLUSTER))


def test_pairwise_distance():
    fn = PairwiseDistance(baseline_assignment = BASELINE)
    # Point 2's mis-clustering changes which pairs agree/disagree relative to baseline:
    # pairs (0,2),(1,2) now falsely agree (both in predicted cluster 0) while pair
    # (2,3) now falsely disagrees (predicted apart, baseline together) -> 3 discordant pairs.
    assert fn(DATA_TO_RULE, RULE_TO_CLUSTER, DATA_TO_CLUSTER) == 3
    assert np.isnan(fn(DATA_TO_RULE, RULE_TO_CLUSTER, None))

    # A perfect match against its own baseline has zero pairwise distance.
    fn_self = PairwiseDistance(baseline_assignment = BASELINE)
    assert fn_self(DATA_TO_RULE, RULE_TO_CLUSTER, BASELINE) == 0


def test_rule_pairwise_distance():
    fn = RulePairwiseDistance(baseline_assignment = BASELINE)
    assert fn(DATA_TO_RULE, RULE_TO_CLUSTER, DATA_TO_CLUSTER) == 24.0
    assert np.isnan(fn(None, RULE_TO_CLUSTER, DATA_TO_CLUSTER))

    with pytest.raises(AssertionError):
        fn(DATA_TO_RULE, np.array([[1, 1], [0, 1]]), DATA_TO_CLUSTER)


def test_silhouette():
    distances = pairwise_distances(X)
    fn = Silhouette(distances = distances)
    assert np.isclose(fn(DATA_TO_RULE, RULE_TO_CLUSTER, DATA_TO_CLUSTER), 0.25)
    assert np.isnan(fn(DATA_TO_RULE, RULE_TO_CLUSTER, None))
