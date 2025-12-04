import numpy as np
import pytest
from intercluster.utils import oned_cluster


class TestClusterValidation:
    """Test input validation and error handling."""
    
    def test_empty_array_raises_error(self):
        """Test that empty array raises ValueError."""
        x = np.array([])
        with pytest.raises(ValueError, match="Input data cannot be empty"):
            oned_cluster(x)
    
    def test_multidimensional_array_raises_error(self):
        """Test that 2D array raises ValueError."""
        x = np.array([[1, 2], [3, 4]])
        with pytest.raises(ValueError):
            oned_cluster(x)
    
    def test_cluster_cost_below_zero_raises_error(self):
        """Test that cluster_cost < 0 raises ValueError."""
        x = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="Cluster cost must be between 0 and 1"):
            oned_cluster(x, cluster_cost=-0.1)
    
    def test_cluster_cost_above_one_raises_error(self):
        """Test that cluster_cost > 1 raises ValueError."""
        x = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="Cluster cost must be between 0 and 1"):
            oned_cluster(x, cluster_cost=1.1)
    
    def test_invalid_method_raises_error(self):
        """Test that unsupported method raises ValueError."""
        x = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="Unsupported clustering method"):
            oned_cluster(x, method="invalid_method")


class TestClusterKMeans:
    """Test k-means clustering functionality."""
    
    def test_single_element(self):
        """Test clustering with a single element."""
        x = np.array([5.0])
        labels = oned_cluster(x, cluster_cost=0.0, method="kmeans")
        assert len(labels) == 1
        assert labels[0] == 0
    
    def test_identical_values(self):
        """Test clustering with identical values."""
        x = np.array([3.0, 3.0, 3.0, 3.0])
        labels = oned_cluster(x, cluster_cost=0.0, method="kmeans")
        # All identical values should be in one cluster
        assert len(np.unique(labels)) == 1
    
    def test_two_clear_clusters(self):
        """Test clustering with two well-separated groups."""
        x = np.array([1.0, 1.1, 1.2, 10.0, 10.1, 10.2])
        labels = oned_cluster(x, cluster_cost=0.01, method="kmeans")
        # Should identify two clusters
        assert len(np.unique(labels)) == 2
        # Check that nearby values are in the same cluster
        assert labels[0] == labels[1] == labels[2]
        assert labels[3] == labels[4] == labels[5]
        assert labels[0] != labels[3]
    
    def test_high_cluster_cost_single_cluster(self):
        """Test that high cluster cost results in fewer clusters."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        labels = oned_cluster(x, cluster_cost=1.0, method="kmeans")
        # With maximum cluster cost, should have one cluster
        assert len(np.unique(labels)) == 1
    
    def test_zero_cluster_cost_many_clusters(self):
        """Test that zero cluster cost can result in more clusters."""
        x = np.array([1.0, 5.0, 10.0, 15.0, 20.0])
        labels = oned_cluster(x, cluster_cost=0.0, method="kmeans")
        # With zero cost, algorithm decides based on data structure
        assert len(labels) == 5
    
    def test_three_clusters(self):
        """Test clustering into three groups."""
        x = np.array([5.0, 1.0, 5.1, 10.0, 10.1, 1.1])
        labels = oned_cluster(x, cluster_cost=0.01, method="kmeans")
        # Should identify three clusters
        assert len(np.unique(labels)) == 3
        assert labels[0] == labels[2]  # 5.0 and 5.1
        assert labels[1] == labels[5]  # 1.0 and 1.1
        assert labels[3] == labels[4]  # 10.0 and 10.1


class TestClusterKMedians:
    """Test k-medians clustering functionality."""
    
    def test_kmedians_single_element(self):
        """Test k-medians with a single element."""
        x = np.array([5.0])
        labels = oned_cluster(x, cluster_cost=0.0, method="kmedians")
        assert len(labels) == 1
        assert labels[0] == 0
    
    def test_kmedians_identical_values(self):
        """Test k-medians with identical values."""
        x = np.array([3.0, 3.0, 3.0, 3.0])
        labels = oned_cluster(x, cluster_cost=0.0, method="kmedians")
        # All identical values should be in one cluster
        assert len(np.unique(labels)) == 1
    
    def test_kmedians_two_clusters(self):
        """Test k-medians with two well-separated groups."""
        x = np.array([1.0, 1.1, 1.2, 10.0, 10.1, 10.2])
        labels = oned_cluster(x, cluster_cost=0.01, method="kmedians")
        # Should identify two clusters
        assert len(np.unique(labels)) == 2
        # Check that nearby values are in the same cluster
        assert labels[0] == labels[1] == labels[2]
        assert labels[3] == labels[4] == labels[5]    


class TestClusterProperties:
    """Test output properties."""
    
    def test_methods_produce_valid_labels(self):
        """Test that both methods produce valid label arrays."""
        x = np.array([1.0, 2.0, 5.0, 6.0, 10.0, 11.0])
        
        labels_kmeans = oned_cluster(x, cluster_cost=0.1, method="kmeans")
        labels_kmedians = oned_cluster(x, cluster_cost=0.1, method="kmedians")
        
        # Both should return arrays of same length
        assert len(labels_kmeans) == len(x)
        assert len(labels_kmedians) == len(x)
        
        # Labels should be non-negative integers
        assert np.all(labels_kmeans >= 0)
        assert np.all(labels_kmedians >= 0)
        
        # Labels should be contiguous (0, 1, 2, ... k-1)
        unique_kmeans = np.unique(labels_kmeans)
        unique_kmedians = np.unique(labels_kmedians)
        assert np.array_equal(unique_kmeans, np.arange(len(unique_kmeans)))
        assert np.array_equal(unique_kmedians, np.arange(len(unique_kmedians)))
    
    def test_labels_are_integers(self):
        """Test that labels are integers."""
        x = np.array([1.0, 2.0, 3.0, 10.0, 11.0])
        labels = oned_cluster(x, cluster_cost=0.1, method="kmeans")
        assert labels.dtype == np.int64
    
    def test_labels_start_from_zero(self):
        """Test that labels start from 0."""
        x = np.array([1.0, 2.0, 10.0, 11.0])
        labels = oned_cluster(x, cluster_cost=0.1, method="kmeans")
        assert np.min(labels) == 0
    
    def test_labels_are_contiguous(self):
        """Test that labels are contiguous without gaps."""
        x = np.array([1.0, 2.0, 3.0, 10.0, 11.0, 12.0])
        labels = oned_cluster(x, cluster_cost=0.1, method="kmeans")
        unique_labels = np.unique(labels)
        expected = np.arange(len(unique_labels))
        assert np.array_equal(unique_labels, expected)