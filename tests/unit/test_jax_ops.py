"""
Unit tests for JAX operations module.

Tests numerical correctness and performance of JAX-based operations.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.core import jax_ops


class TestBasicArrayOperations:
    """Test basic array operations."""
    
    def test_invert_pattern(self):
        """Test pattern inversion."""
        pattern = jnp.array([1.0, 0.0, 1.0, 0.0])
        expected = jnp.array([0.0, 1.0, 0.0, 1.0])
        
        result = jax_ops.invert_pattern(pattern)
        
        np.testing.assert_array_equal(result, expected)
    
    def test_invert_matrix(self):
        """Test matrix inversion."""
        matrix = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        expected = jnp.array([[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
        
        result = jax_ops.invert_matrix(matrix)
        
        np.testing.assert_array_equal(result, expected)
    
    def test_normalize_matrix(self):
        """Test matrix normalization."""
        matrix = jnp.array([[1.0, 10.0], [5.0, 20.0], [3.0, 15.0]])
        
        result = jax_ops.normalize_matrix(matrix)
        
        # Check each column is normalized to [0, 1]
        assert result.min() >= 0.0
        assert result.max() <= 1.0
        np.testing.assert_allclose(result[:, 0].min(), 0.0, atol=1e-6)
        np.testing.assert_allclose(result[:, 0].max(), 1.0, atol=1e-6)
        np.testing.assert_allclose(result[:, 1].min(), 0.0, atol=1e-6)
        np.testing.assert_allclose(result[:, 1].max(), 1.0, atol=1e-6)
    
    def test_normalize_matrix_constant_column(self):
        """Test normalization with constant column."""
        matrix = jnp.array([[1.0, 5.0], [1.0, 10.0], [1.0, 7.0]])
        
        result = jax_ops.normalize_matrix(matrix)
        
        # Constant column should become all zeros (or epsilon-scaled)
        assert result[:, 0].std() < 0.1  # Very small variance
    
    def test_compute_incident_matrix_upper(self):
        """Test incident matrix computation with upper threshold."""
        values = jnp.array([[1.0, 5.0], [3.0, 7.0], [2.0, 4.0]])
        theta = 4.0
        
        result = jax_ops.compute_incident_matrix(values, theta, "upper")
        
        expected = jnp.array([[0, 1], [0, 1], [0, 1]], dtype=jnp.int32)
        np.testing.assert_array_equal(result, expected)
    
    def test_compute_incident_matrix_lower(self):
        """Test incident matrix computation with lower threshold."""
        values = jnp.array([[1.0, 5.0], [3.0, 7.0], [2.0, 4.0]])
        theta = 4.0
        
        result = jax_ops.compute_incident_matrix(values, theta, "lower")
        
        expected = jnp.array([[1, 0], [1, 0], [1, 1]], dtype=jnp.int32)
        np.testing.assert_array_equal(result, expected)


class TestDistanceCalculations:
    """Test distance calculation functions."""
    
    def test_euclidean_distance_pairwise(self):
        """Test pairwise Euclidean distance calculation."""
        coords1 = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        coords2 = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        
        result = jax_ops.euclidean_distance_pairwise(coords1, coords2)
        
        # Distance from (0,0) to (1,0) should be 1.0
        assert result.shape == (2, 2)
        np.testing.assert_allclose(result[0, 0], 1.0, atol=1e-6)
        # Distance from (1,1) to (1,0) should be 1.0
        np.testing.assert_allclose(result[1, 0], 1.0, atol=1e-6)
        # Distance from (1,1) to (0,1) should be 1.0
        np.testing.assert_allclose(result[1, 1], 1.0, atol=1e-6)
    
    def test_euclidean_distance_self(self):
        """Test Euclidean distance of points to themselves."""
        coords = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        
        result = jax_ops.euclidean_distance_pairwise(coords, coords)
        
        # Diagonal should be zero (distance to self)
        np.testing.assert_allclose(jnp.diag(result), 0.0, atol=1e-6)
    
    def test_haversine_distance_basic(self):
        """Test Haversine distance calculation."""
        # Houston coordinates
        lat1 = jnp.array([29.7604])
        lon1 = jnp.array([-95.3698])
        lat2 = jnp.array([29.7604])
        lon2 = jnp.array([-95.3698])
        
        result = jax_ops.haversine_distance(lat1, lon1, lat2, lon2)
        
        # Distance to self should be ~0
        np.testing.assert_allclose(result[0], 0.0, atol=1.0)  # Within 1 meter
    
    def test_haversine_distance_known(self):
        """Test Haversine with known distance."""
        # Two points approximately 1 degree apart in longitude
        lat1 = jnp.array([29.0])
        lon1 = jnp.array([-95.0])
        lat2 = jnp.array([29.0])
        lon2 = jnp.array([-94.0])
        
        result = jax_ops.haversine_distance(lat1, lon1, lat2, lon2)
        
        # 1 degree longitude at this latitude is ~95-100 km
        assert 90000 < result[0] < 105000  # Between 90 and 105 km
    
    def test_haversine_distance_matrix(self):
        """Test pairwise Haversine distance matrix."""
        coords1 = jnp.array([[29.7604, -95.3698], [29.7605, -95.3699]])
        coords2 = jnp.array([[29.7604, -95.3698]])
        
        result = jax_ops.haversine_distance_matrix(coords1, coords2)
        
        assert result.shape == (2, 1)
        # First point to itself should be ~0
        np.testing.assert_allclose(result[0, 0], 0.0, atol=1.0)
        # Second point should be close but not zero
        assert result[1, 0] > 0


class TestConnectivityOperations:
    """Test connectivity operations."""
    
    def test_build_adjacency_matrix(self):
        """Test adjacency matrix construction."""
        distances = jnp.array([[0.0, 1.0, 5.0],
                               [1.0, 0.0, 2.0],
                               [5.0, 2.0, 0.0]])
        threshold = 3.0
        
        result = jax_ops.build_adjacency_matrix(distances, threshold)
        
        expected = jnp.array([[1, 1, 0],
                              [1, 1, 1],
                              [0, 1, 1]], dtype=jnp.int32)
        np.testing.assert_array_equal(result, expected)
    
    def test_build_adjacency_matrix_all_connected(self):
        """Test adjacency with high threshold."""
        distances = jnp.array([[0.0, 1.0],
                               [1.0, 0.0]])
        threshold = 10.0
        
        result = jax_ops.build_adjacency_matrix(distances, threshold)
        
        # All nodes should be connected
        expected = jnp.ones((2, 2), dtype=jnp.int32)
        np.testing.assert_array_equal(result, expected)
    
    def test_compute_connectivity(self):
        """Test flood connectivity computation."""
        elevations = jnp.array([5.0, 10.0, 8.0])
        adjacency = jnp.ones((3, 3), dtype=jnp.int32)
        water_level = 9.0
        
        result = jax_ops.compute_connectivity(elevations, adjacency, water_level)
        
        # Parcels 0 and 2 are flooded (< 9.0), parcel 1 is not
        # So connectivity should be high between 0 and 2
        assert result[0, 2] == 1  # Both flooded
        assert result[1, 0] == 0  # One not flooded
        assert result[1, 2] == 0  # One not flooded
    
    def test_compute_connectivity_no_flooding(self):
        """Test connectivity with no flooding."""
        elevations = jnp.array([10.0, 12.0, 15.0])
        adjacency = jnp.ones((3, 3), dtype=jnp.int32)
        water_level = 5.0
        
        result = jax_ops.compute_connectivity(elevations, adjacency, water_level)
        
        # No parcels flooded, so no connectivity
        np.testing.assert_array_equal(result, jnp.zeros((3, 3), dtype=jnp.int32))


class TestStatisticalOperations:
    """Test statistical operations."""
    
    def test_weighted_average(self):
        """Test weighted average calculation."""
        values = jnp.array([1.0, 2.0, 3.0])
        weights = jnp.array([0.5, 0.3, 0.2])
        
        result = jax_ops.weighted_average(values, weights)
        
        expected = (1.0 * 0.5 + 2.0 * 0.3 + 3.0 * 0.2) / (0.5 + 0.3 + 0.2)
        np.testing.assert_allclose(result, expected, atol=1e-6)
    
    def test_weighted_average_equal_weights(self):
        """Test weighted average with equal weights."""
        values = jnp.array([1.0, 2.0, 3.0])
        weights = jnp.ones(3)
        
        result = jax_ops.weighted_average(values, weights)
        
        # Should equal simple average
        expected = 2.0
        np.testing.assert_allclose(result, expected, atol=1e-6)
    
    def test_compute_percentile_mask(self):
        """Test percentile mask computation."""
        values = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        result = jax_ops.compute_percentile_mask(values, 60)
        
        # 60th percentile of [1,2,3,4,5] is 3.4, so 4 and 5 should be above
        expected = jnp.array([0, 0, 0, 1, 1], dtype=jnp.int32)
        np.testing.assert_array_equal(result, expected)


class TestImpactPropagation:
    """Test impact propagation functions."""
    
    def test_propagate_impact(self):
        """Test basic impact propagation."""
        values = jnp.array([100.0, 100.0, 100.0])
        connectivity = jnp.array([[1, 1, 0],
                                  [1, 1, 1],
                                  [0, 1, 1]], dtype=jnp.int32)
        impact_factor = 0.1
        
        result = jax_ops.propagate_impact(values, connectivity, impact_factor)
        
        # Middle parcel has 3 connections, so loses 30%
        assert result[1] < values[1]
        # All values should be non-negative
        assert jnp.all(result >= 0)
    
    def test_propagate_impact_no_connectivity(self):
        """Test impact with no connections."""
        values = jnp.array([100.0, 100.0])
        connectivity = jnp.eye(2, dtype=jnp.int32)  # Only self-connections
        impact_factor = 0.5
        
        result = jax_ops.propagate_impact(values, connectivity, impact_factor)
        
        # Each parcel should lose 50% (1 connection × 0.5)
        expected = jnp.array([50.0, 50.0])
        np.testing.assert_allclose(result, expected, atol=1e-6)
    
    def test_compute_cumulative_impact(self):
        """Test cumulative impact over time steps."""
        initial_values = jnp.array([100.0, 100.0])
        # Two time steps with same connectivity
        connectivity_matrices = jnp.array([
            [[1, 1], [1, 1]],
            [[1, 1], [1, 1]]
        ], dtype=jnp.int32)
        impact_factor = 0.1
        
        result = jax_ops.compute_cumulative_impact(
            initial_values, connectivity_matrices, impact_factor
        )
        
        # Values should decrease over time
        assert jnp.all(result < initial_values)
        assert jnp.all(result >= 0)


class TestRandomGeneration:
    """Test random number generation."""
    
    def test_generate_random_events(self):
        """Test random event generation."""
        key = jax.random.PRNGKey(42)
        n_events = 10
        intensity_range = (1.0, 10.0)
        
        result = jax_ops.generate_random_events(key, n_events, intensity_range)
        
        assert result.shape == (n_events,)
        assert jnp.all(result >= intensity_range[0])
        assert jnp.all(result <= intensity_range[1])
    
    def test_generate_random_events_reproducible(self):
        """Test that same key produces same results."""
        key = jax.random.PRNGKey(42)
        n_events = 5
        intensity_range = (0.0, 1.0)
        
        result1 = jax_ops.generate_random_events(key, n_events, intensity_range)
        result2 = jax_ops.generate_random_events(key, n_events, intensity_range)
        
        np.testing.assert_array_equal(result1, result2)


class TestArrayUpdates:
    """Test immutable array update operations."""
    
    def test_update_array_at_indices(self):
        """Test updating array at specific indices."""
        array = jnp.array([1.0, 2.0, 3.0, 4.0])
        indices = jnp.array([0, 2])
        values = jnp.array([10.0, 30.0])
        
        result = jax_ops.update_array_at_indices(array, indices, values)
        
        expected = jnp.array([10.0, 2.0, 30.0, 4.0])
        np.testing.assert_array_equal(result, expected)
        # Original array should be unchanged (immutability)
        np.testing.assert_array_equal(array, jnp.array([1.0, 2.0, 3.0, 4.0]))
    
    def test_multiply_array_at_indices(self):
        """Test multiplying array values at indices."""
        array = jnp.array([1.0, 2.0, 3.0, 4.0])
        indices = jnp.array([1, 3])
        multiplier = 2.0
        
        result = jax_ops.multiply_array_at_indices(array, indices, multiplier)
        
        expected = jnp.array([1.0, 4.0, 3.0, 8.0])
        np.testing.assert_array_equal(result, expected)


class TestVectorizedOperations:
    """Test vectorized batch operations."""
    
    def test_haversine_distance_vectorized(self):
        """Test vectorized Haversine distance."""
        n_points = 5
        lat1 = jnp.ones(n_points) * 29.0
        lon1 = jnp.ones(n_points) * -95.0
        lat2 = jnp.ones(n_points) * 29.0
        lon2 = jnp.ones(n_points) * -94.0
        
        result = jax_ops.haversine_distance_vectorized(lat1, lon1, lat2, lon2)
        
        assert result.shape == (n_points,)
        # All distances should be similar (same coordinates)
        assert result.std() < 1.0  # Very small variation
    
    def test_invert_pattern_vectorized(self):
        """Test vectorized pattern inversion."""
        patterns = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        
        result = jax_ops.invert_pattern_vectorized(patterns)
        
        expected = jnp.array([[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
        np.testing.assert_array_equal(result, expected)


class TestJITCompilation:
    """Test that JIT compilation works correctly."""
    
    def test_jit_compilation_runs(self):
        """Test that JIT-compiled functions execute."""
        # This would fail if JIT compilation had issues
        pattern = jnp.array([1.0, 0.0])
        result = jax_ops.invert_pattern(pattern)
        assert result is not None
    
    def test_jit_multiple_calls(self):
        """Test that JIT compilation caches correctly."""
        pattern1 = jnp.array([1.0, 0.0])
        pattern2 = jnp.array([0.0, 1.0])
        
        result1 = jax_ops.invert_pattern(pattern1)
        result2 = jax_ops.invert_pattern(pattern2)
        
        # Both should complete successfully
        assert result1 is not None
        assert result2 is not None


# Performance benchmark (not run by default)
@pytest.mark.skip(reason="Performance test, run manually")
class TestPerformance:
    """Performance benchmarks for JAX operations."""
    
    def test_distance_calculation_performance(self):
        """Benchmark distance calculations."""
        import time
        
        n_points = 1000
        coords = jnp.array(np.random.rand(n_points, 2) * 10)
        
        # Warm up JIT
        _ = jax_ops.euclidean_distance_pairwise(coords[:10], coords[:10])
        
        start = time.time()
        result = jax_ops.euclidean_distance_pairwise(coords, coords)
        result.block_until_ready()  # Wait for JAX async execution
        elapsed = time.time() - start
        
        print(f"\nDistance calculation for {n_points} points: {elapsed:.4f}s")
        assert result.shape == (n_points, n_points)

