import pytest
import numpy as np
from intercluster.decision_sets.objectives.objectives import (
    Objective,
    CoverageMistakeObjective,
    TotalCoverageMistakeObjective,
    CoverageCostObjective,
    TotalCoverageCostObjective
)


####################################################################################################
# Fixtures
####################################################################################################


@pytest.fixture
def simple_data():
    """Simple 2D dataset with 10 points."""
    return np.array([
        [0, 0], [1, 1], [2, 2],  # Cluster 0
        [10, 10], [11, 11], [12, 12],  # Cluster 1
        [20, 20], [21, 21], [22, 22], [23, 23]  # Cluster 2
    ], dtype=np.float64)


@pytest.fixture
def simple_data_to_cluster():
    """Data to cluster assignment for simple_data."""
    # 3 clusters, 10 points
    return np.array([
        [1, 0, 0],  # point 0 -> cluster 0
        [1, 0, 0],  # point 1 -> cluster 0
        [1, 0, 0],  # point 2 -> cluster 0
        [0, 1, 0],  # point 3 -> cluster 1
        [0, 1, 0],  # point 4 -> cluster 1
        [0, 1, 0],  # point 5 -> cluster 1
        [0, 0, 1],  # point 6 -> cluster 2
        [0, 0, 1],  # point 7 -> cluster 2
        [0, 0, 1],  # point 8 -> cluster 2
        [0, 0, 1],  # point 9 -> cluster 2
    ], dtype=bool)


@pytest.fixture
def simple_rule_to_cluster():
    """Rule to cluster assignment."""
    # 5 rules, 3 clusters
    return np.array([
        [1, 0, 0],  # rule 0 -> cluster 0
        [1, 0, 0],  # rule 1 -> cluster 0
        [0, 1, 0],  # rule 2 -> cluster 1
        [0, 0, 1],  # rule 3 -> cluster 2
        [0, 0, 1],  # rule 4 -> cluster 2
    ], dtype=bool)


@pytest.fixture
def simple_data_to_rules():
    """Data to rules assignment for simple_data."""
    # 10 points, 5 rules
    return np.array([
        [1, 0, 0, 0, 0],  # point 0 -> rule 0
        [1, 1, 0, 0, 0],  # point 1 -> rules 0, 1
        [1, 0, 0, 0, 0],  # point 2 -> rule 0
        [0, 0, 1, 0, 0],  # point 3 -> rule 2
        [0, 0, 1, 0, 0],  # point 4 -> rule 2
        [0, 0, 0, 0, 0],  # point 5 -> no rules (uncovered)
        [0, 0, 0, 1, 1],  # point 6 -> rules 3, 4
        [0, 0, 0, 0, 1],  # point 7 -> rule 4
        [0, 0, 0, 0, 1],  # point 8 -> rule 4
        [0, 0, 0, 0, 0],  # point 9 -> no rules (uncovered)
    ], dtype=bool)


@pytest.fixture
def mistake_data_to_rules():
    """Data to rules assignment with mistakes."""
    # 10 points, 5 rules
    return np.array([
        [1, 0, 0, 1, 1],  # point 0 -> rule 0, rule 3, rule 4
        [1, 1, 0, 0, 0],  # point 1 -> rules 0, 1
        [1, 0, 0, 0, 0],  # point 2 -> rule 0
        [1, 0, 1, 0, 0],  # point 3 -> rule 0, rule 2
        [0, 0, 1, 0, 0],  # point 4 -> rule 2
        [0, 0, 0, 0, 0],  # point 5 -> no rules (uncovered)
        [0, 0, 1, 1, 1],  # point 6 -> rules 2, 3, 4
        [0, 0, 0, 0, 1],  # point 7 -> rule 4
        [0, 1, 0, 0, 1],  # point 8 -> rule 1, rule 4
        [0, 0, 0, 0, 0],  # point 9 -> no rules (uncovered)
    ], dtype=bool)


@pytest.fixture
def tied_mistake_data_to_rules():
    """Data to rules assignment for simple_data."""
    # 10 points, 5 rules
    return np.array([
        [1, 0, 0, 0, 0],  # point 0 -> rule 0
        [1, 1, 0, 0, 0],  # point 1 -> rules 0, 1
        [1, 0, 0, 0, 0],  # point 2 -> rule 0
        [0, 0, 1, 0, 0],  # point 3 -> rule 2
        [0, 0, 1, 0, 0],  # point 4 -> rule 2
        [0, 0, 0, 0, 0],  # point 5 -> no rules (uncovered)
        [0, 0, 0, 1, 1],  # point 6 -> rules 3, 4
        [0, 0, 0, 0, 1],  # point 7 -> rule 4
        [0, 0, 0, 1, 0],  # point 8 -> rule 3
        [0, 0, 0, 0, 0],  # point 9 -> no rules (uncovered)
    ], dtype=bool)


@pytest.fixture
def overlapping_data_to_cluster():
    """Data to cluster assignment with overlaps."""
    # Some points belong to multiple clusters
    return np.array([
        [1, 1, 0],  # point 0 -> clusters 0, 1
        [1, 0, 0],  # point 1 -> cluster 0
        [1, 0, 0],  # point 2 -> cluster 0
        [0, 1, 0],  # point 3 -> cluster 1
        [0, 1, 0],  # point 4 -> cluster 1
        [0, 1, 1],  # point 5 -> clusters 1, 2
        [0, 0, 1],  # point 6 -> cluster 2
        [0, 0, 1],  # point 7 -> cluster 2
        [0, 0, 1],  # point 8 -> cluster 2
        [0, 0, 1],  # point 9 -> cluster 2
    ], dtype=bool)


@pytest.fixture
def cluster_centers():
    """Cluster centers for cost-based objectives."""
    return np.array([
        [1, 1],    # center of cluster 0
        [11, 11],  # center of cluster 1
        [21, 21],  # center of cluster 2
    ], dtype=np.float64)


####################################################################################################
# Base Objective Tests
####################################################################################################


class TestObjectiveBase:
    """Tests for the base Objective class."""
        
    def test_select_shape_assertions(
        self, 
        simple_data, 
        simple_data_to_cluster,
        simple_rule_to_cluster,
        simple_data_to_rules
    ):
        """Test that select method validates input shapes."""
        obj = CoverageMistakeObjective(n_rules=3)
        
        # Mismatched data and data_to_cluster
        with pytest.raises(AssertionError):
            bad_data = simple_data[:5]  # Only 5 points
            obj.select(
                bad_data,
                simple_data_to_cluster,  # Expects 10 points
                simple_rule_to_cluster,
                simple_data_to_rules
            )
            
        # Mismatched clusters between data and rules
        with pytest.raises(AssertionError):
            bad_rule_to_cluster = simple_rule_to_cluster[:, :2]  # Only 2 clusters
            obj.select(
                simple_data,
                simple_data_to_cluster,  # 3 clusters
                bad_rule_to_cluster,
                simple_data_to_rules
            )
            
    def test_select_rule_assignment_assertion(
        self,
        simple_data,
        simple_data_to_cluster,
        simple_data_to_rules
    ):
        """Test that select validates rules are assigned to exactly one cluster."""
        obj = CoverageMistakeObjective(n_rules=3)
        
        # Rules assigned to multiple clusters
        bad_rule_to_cluster = np.array([
            [1, 1, 0],  # rule 0 -> clusters 0, 1 (invalid!)
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 1],
        ], dtype=bool)
        
        with pytest.raises(AssertionError):
            obj.select(
                simple_data,
                simple_data_to_cluster,
                bad_rule_to_cluster,
                simple_data_to_rules
            )

    def test_clustser_centers_assertion(
        self,
        simple_data,
        simple_data_to_cluster,
        simple_rule_to_cluster,
        simple_data_to_rules,
        cluster_centers
    ):        
        # Mismatched number of cluster centers
        bad_cluster_centers = cluster_centers[:2]  # Only 2 centers instead of 3
        
        obj_bad = CoverageCostObjective(
            n_rules=3,
            lambda_val=1.0,
            cluster_centers=bad_cluster_centers,
            method="kmeans"
        )
        
        with pytest.raises(AssertionError):
            obj_bad.select(
                simple_data,
                simple_data_to_cluster,
                simple_rule_to_cluster,
                simple_data_to_rules
            )


####################################################################################################
# CoverageMistakeObjective Tests
####################################################################################################


class TestCoverageMistakeObjective:
    """Tests for CoverageMistakeObjective class."""
        
    def test_marginal_gain_new_coverage(self):
        """Test marginal gain calculation with new coverage."""
        obj = CoverageMistakeObjective(n_rules=3)
        
        rule_points = {0: {0, 1, 2}}
        rule_cluster_coverage = {0: {0, 1, 2}}
        selected_cluster_coverage = {0: set()}
        
        gain = obj.marginal_gain(0, 0, rule_points, rule_cluster_coverage, selected_cluster_coverage)
        assert gain == 3  # All 3 points are new coverage
        
    def test_marginal_gain_partial_coverage(self):
        """Test marginal gain with partial existing coverage."""
        obj = CoverageMistakeObjective(n_rules=3)
        
        rule_points = {1: {1, 2, 3}}
        rule_cluster_coverage = {1: {1, 2, 3}}
        selected_cluster_coverage = {0: {1, 2}}  # Already covered points 1, 2
        
        gain = obj.marginal_gain(1, 0, rule_points, rule_cluster_coverage, selected_cluster_coverage)
        assert gain == 1  # Only point 3 is new
        
    def test_marginal_gain_no_new_coverage(self):
        """Test marginal gain when all points already covered."""
        obj = CoverageMistakeObjective(n_rules=3)
        
        rule_points = {0: {0, 1}}
        rule_cluster_coverage = {0: {0, 1}}
        selected_cluster_coverage = {0: {0, 1, 2}}  # All points already covered
        
        gain = obj.marginal_gain(0, 0, rule_points, rule_cluster_coverage, selected_cluster_coverage)
        assert gain == 0
        
    def test_marginal_cost_no_mistakes(self):
        """Test marginal cost when rule makes no mistakes."""
        obj = CoverageMistakeObjective(n_rules=3)
        
        rule_points = {0: {0, 1, 2}}
        rule_cluster_coverage = {0: {0, 1, 2}}  # All rule points are in cluster
        selected_cluster_coverage = {0: set()}
        
        cost = obj.marginal_cost(0, 0, rule_points, rule_cluster_coverage, selected_cluster_coverage)
        assert cost == 0
        
    def test_marginal_cost_with_mistakes(self):
        """Test marginal cost when rule covers points outside cluster."""
        obj = CoverageMistakeObjective(n_rules=3)
        
        rule_points = {0: {0, 1, 2, 3, 4}}
        rule_cluster_coverage = {0: {0, 1}}  # Only 2 points in cluster
        selected_cluster_coverage = {0: set()}
        
        cost = obj.marginal_cost(0, 0, rule_points, rule_cluster_coverage, selected_cluster_coverage)
        assert cost == 3  # 3 mistakes (points 2, 3, 4)
        
    def test_select_basic(
        self,
        simple_data,
        simple_data_to_cluster,
        simple_rule_to_cluster,
        simple_data_to_rules
    ):
        """Test basic rule selection."""
        obj = CoverageMistakeObjective(n_rules=3, lambda_val=1.0)
        
        selected = obj.select(
            simple_data,
            simple_data_to_cluster,
            simple_rule_to_cluster,
            simple_data_to_rules
        )
        
        assert isinstance(selected, np.ndarray)
        assert len(selected) <= 3  # At most n_rules selected
        assert all(0 <= r < 5 for r in selected)  # Valid rule indices
        assert set(list(selected)) == {0,2,4}
        
    def test_select_returns_empty_when_no_positive_scores(
        self,
        simple_data,
        simple_data_to_cluster,
        simple_rule_to_cluster,
        mistake_data_to_rules
    ):
        """Test that selection returns empty when no rules have positive scores."""
        # High lambda makes all scores negative
        obj = CoverageMistakeObjective(n_rules=3, lambda_val=1000.0)
        
        selected = obj.select(
            simple_data,
            simple_data_to_cluster,
            simple_rule_to_cluster,
            mistake_data_to_rules
        )
        
        assert len(selected) == 0
        
    def test_select_with_low_lambda(
        self,
        simple_data,
        simple_data_to_cluster,
        simple_rule_to_cluster,
        mistake_data_to_rules
    ):
        """Test selection with low lambda (prioritizes coverage)."""
        obj = CoverageMistakeObjective(n_rules=3, lambda_val=0.1)
        
        selected = obj.select(
            simple_data,
            simple_data_to_cluster,
            simple_rule_to_cluster,
            mistake_data_to_rules
        )
        
        assert len(selected) > 0
        
    def test_select_with_overlapping_clusters(
        self,
        simple_data,
        overlapping_data_to_cluster,
        simple_rule_to_cluster,
        simple_data_to_rules
    ):
        """Test selection with overlapping cluster assignments."""
        obj = CoverageMistakeObjective(n_rules=3, lambda_val=1.0)
        
        selected = obj.select(
            simple_data,
            overlapping_data_to_cluster,
            simple_rule_to_cluster,
            simple_data_to_rules
        )
        
        assert isinstance(selected, np.ndarray)
        assert len(selected) <= 3


    def test_select_ties(
        self,
        simple_data,
        simple_data_to_cluster,
        simple_rule_to_cluster,
        tied_mistake_data_to_rules
    ):
        """Test basic rule selection."""
        obj = CoverageMistakeObjective(n_rules=3, lambda_val=1.0)
        
        selected = obj.select(
            simple_data,
            simple_data_to_cluster,
            simple_rule_to_cluster,
            tied_mistake_data_to_rules
        )
        
        assert set(list(selected)) == {0,2,3}


####################################################################################################
# TotalCoverageMistakeObjective Tests
####################################################################################################


class TestTotalCoverageMistakeObjective:
    """Tests for TotalCoverageMistakeObjective class."""
        
    def test_marginal_gain_total_coverage(self):
        """Test marginal gain considers total coverage across all clusters."""
        obj = TotalCoverageMistakeObjective(n_rules=3)
        
        rule_points = {0: {0, 1, 2}}
        rule_cluster_coverage = {0: {0, 1}}  # Only 2 in cluster
        selected_cluster_coverage = {
            0: set(),
            1: {2}  # Point 2 covered in different cluster
        }
        
        gain = obj.marginal_gain(0, 0, rule_points, rule_cluster_coverage, selected_cluster_coverage)
        assert gain == 2  # Points 0, 1 are new (point 2 already covered globally)
        
    def test_marginal_gain_all_clusters_covered(self):
        """Test gain when points covered across multiple clusters."""
        obj = TotalCoverageMistakeObjective(n_rules=3)
        
        rule_points = {0: {0, 1, 2}}
        rule_cluster_coverage = {0: {0, 1, 2}}
        selected_cluster_coverage = {
            0: {0},
            1: {1},
            2: {2}
        }
        
        gain = obj.marginal_gain(0, 0, rule_points, rule_cluster_coverage, selected_cluster_coverage)
        assert gain == 0  # All points already covered globally
        
    def test_marginal_cost_same_as_coverage_mistake(self):
        """Test that marginal cost is same as CoverageMistakeObjective."""
        obj = TotalCoverageMistakeObjective(n_rules=3)
        
        rule_points = {0: {0, 1, 2, 3}}
        rule_cluster_coverage = {0: {0, 1}}
        selected_cluster_coverage = {0: set()}
        
        cost = obj.marginal_cost(0, 0, rule_points, rule_cluster_coverage, selected_cluster_coverage)
        assert cost == 2  # Points 2, 3 are mistakes
        
    def test_select_basic(
        self,
        simple_data,
        simple_data_to_cluster,
        simple_rule_to_cluster,
        simple_data_to_rules
    ):
        """Test basic rule selection with total coverage."""
        obj = TotalCoverageMistakeObjective(n_rules=3, lambda_val=1.0)
        
        selected = obj.select(
            simple_data,
            simple_data_to_cluster,
            simple_rule_to_cluster,
            simple_data_to_rules
        )
        
        assert isinstance(selected, np.ndarray)
        assert len(selected) <= 3
        assert set(list(selected)) == {0,2,4}


####################################################################################################
# CoverageCostObjective Tests
####################################################################################################


class TestCoverageCostObjective:
    """Tests for CoverageCostObjective class."""
        
    def test_init_invalid_method(self, cluster_centers):
        """Test that invalid method raises ValueError."""
        method = "invalid_method"
        error_msg = f"Method {method} not supported. Supported methods are 'kmeans' and 'kmedians'."
        with pytest.raises(ValueError, match=error_msg):
            CoverageCostObjective(
                n_rules=5,
                lambda_val=2.0,
                cluster_centers=cluster_centers,
                method="invalid_method"
            )
            
    def test_marginal_gain_same_as_coverage_mistake(self):
        """Test that marginal gain is same as CoverageMistakeObjective."""
        cluster_centers = np.array([[1, 1], [10, 10]], dtype=np.float64)
        obj = CoverageCostObjective(
            n_rules=3,
            lambda_val=1.0,
            cluster_centers=cluster_centers,
            method="kmeans"
        )
        
        rule_points = {0: {0, 1, 2}}
        rule_cluster_coverage = {0: {0, 1, 2}}
        selected_cluster_coverage = {0: set(), 1: set()}
        
        gain = obj.marginal_gain(0, 0, rule_points, rule_cluster_coverage, selected_cluster_coverage)
        assert gain == 3
        
    def test_marginal_cost_kmeans(self, cluster_centers, simple_data):
        """Test marginal cost calculation with kmeans."""
        obj = CoverageCostObjective(
            n_rules=3,
            lambda_val=1.0,
            cluster_centers=cluster_centers,
            method="kmeans"
        )
        obj.data = simple_data  # Set data for cost calculation
        
        # Rule covers points 0, 1 from cluster 0
        rule_points = {0: {0, 1}}
        rule_cluster_coverage = {0: {0, 1}}
        selected_cluster_coverage = {0: set(), 1: set(), 2: set()}
        
        cost = obj.marginal_cost(0, 0, rule_points, rule_cluster_coverage, selected_cluster_coverage)
        
        # Expected: sum of squared distances from points to cluster center
        expected_cost = np.sum((simple_data[[0, 1]] - cluster_centers[0])**2)
        assert np.isclose(cost, expected_cost)
        
    def test_marginal_cost_kmedians(self, cluster_centers, simple_data):
        """Test marginal cost calculation with kmedians."""
        obj = CoverageCostObjective(
            n_rules=3,
            lambda_val=1.0,
            cluster_centers=cluster_centers,
            method="kmedians"
        )
        obj.data = simple_data  # Set data for cost calculation
        
        # Rule covers points 0, 1 from cluster 0
        rule_points = {0: {0, 1}}
        rule_cluster_coverage = {0: {0, 1}}
        selected_cluster_coverage = {0: set(), 1: set(), 2: set()}
        
        cost = obj.marginal_cost(0, 0, rule_points, rule_cluster_coverage, selected_cluster_coverage)
        
        # Expected: sum of absolute distances from points to cluster center
        expected_cost = np.sum(np.abs(simple_data[[0, 1]] - cluster_centers[0]))
        assert np.isclose(cost, expected_cost)


####################################################################################################
# TotalCoverageCostObjective Tests
####################################################################################################


class TestTotalCoverageCostObjective:
    """Tests for TotalCoverageCostObjective class."""
        
    def test_init_invalid_method(self, cluster_centers):
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError, match="not supported"):
            TotalCoverageCostObjective(
                n_rules=5,
                lambda_val=2.0,
                cluster_centers=cluster_centers,
                method="bad_method"
            )
            
    def test_marginal_gain_total_coverage(self):
        """Test that marginal gain considers total coverage."""
        cluster_centers = np.array([[1, 1], [10, 10]], dtype=np.float64)
        obj = TotalCoverageCostObjective(
            n_rules=3,
            lambda_val=1.0,
            cluster_centers=cluster_centers,
            method="kmeans"
        )
        
        rule_points = {0: {0, 1, 2}}
        rule_cluster_coverage = {0: {0, 1}}
        selected_cluster_coverage = {
            0: set(),
            1: {2}  # Point 2 covered in another cluster
        }
        
        gain = obj.marginal_gain(0, 0, rule_points, rule_cluster_coverage, selected_cluster_coverage)
        assert gain == 2  # Only points 0, 1 are new globally
        
    def test_marginal_cost_kmeans(self, cluster_centers, simple_data):
        """Test marginal cost with kmeans method."""
        obj = TotalCoverageCostObjective(
            n_rules=3,
            lambda_val=1.0,
            cluster_centers=cluster_centers,
            method="kmeans"
        )
        obj.data = simple_data  # Set data for cost calculation
        
        rule_points = {0: {0, 1}}
        rule_cluster_coverage = {0: {0, 1}}
        selected_cluster_coverage = {0: set(), 1: set(), 2: set()}
        
        cost = obj.marginal_cost(0, 0, rule_points, rule_cluster_coverage, selected_cluster_coverage)
        
        expected_cost = np.sum((simple_data[[0, 1]] - cluster_centers[0])**2)
        assert np.isclose(cost, expected_cost)
        
    def test_marginal_cost_kmedians(self, cluster_centers, simple_data):
        """Test marginal cost with kmedians method."""
        obj = TotalCoverageCostObjective(
            n_rules=3,
            lambda_val=1.0,
            cluster_centers=cluster_centers,
            method="kmedians"
        )
        obj.data = simple_data  # Set data for cost calculation
        
        rule_points = {0: {0, 1}}
        rule_cluster_coverage = {0: {0, 1}}
        selected_cluster_coverage = {0: set(), 1: set(), 2: set()}
        
        cost = obj.marginal_cost(0, 0, rule_points, rule_cluster_coverage, selected_cluster_coverage)
        
        expected_cost = np.sum(np.abs(simple_data[[0, 1]] - cluster_centers[0]))
        assert np.isclose(cost, expected_cost)


####################################################################################################
# Edge Cases
####################################################################################################


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_single_rule_selection(
        self,
        simple_data,
        simple_data_to_cluster,
        simple_rule_to_cluster,
        simple_data_to_rules
    ):
        """Test selecting only one rule."""
        obj = CoverageMistakeObjective(n_rules=1, lambda_val=0.5)
        
        selected = obj.select(
            simple_data,
            simple_data_to_cluster,
            simple_rule_to_cluster,
            simple_data_to_rules
        )
        
        assert len(selected) <= 1
        
    def test_empty_rule_coverage(self):
        """Test with rules that cover no points."""
        data = np.array([[0, 0], [1, 1]], dtype=np.float64)
        data_to_cluster = np.array([[1, 0], [0, 1]], dtype=bool)
        rule_to_cluster = np.array([[1, 0], [0, 1]], dtype=bool)
        data_to_rules = np.array([[0, 0], [0, 0]], dtype=bool)  # No coverage
        
        obj = CoverageMistakeObjective(n_rules=2, lambda_val=1.0)
        
        selected = obj.select(
            data,
            data_to_cluster,
            rule_to_cluster,
            data_to_rules
        )
        
        assert len(selected) == 0  # No rules with positive gain
        
    def test_all_points_covered_initially(self):
        """Test when all points in a cluster are already covered."""
        obj = CoverageMistakeObjective(n_rules=3)
        
        rule_points = {0: {0, 1, 2}}
        rule_cluster_coverage = {0: {0, 1, 2}}
        selected_cluster_coverage = {0: {0, 1, 2}}  # All covered
        
        gain = obj.marginal_gain(0, 0, rule_points, rule_cluster_coverage, selected_cluster_coverage)
        assert gain == 0
