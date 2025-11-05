"""
Tests for the hyper_graph module
"""
import numpy as np
import pytest
from dias.core.hyper_graph import (
    compute_incident,
    invert_pattern,
    normalize,
    sparse_graph,
    compute_classes,
    simple_ecc,
)


class TestHyperGraph:
    """Test cases for hyper_graph functions"""
    
    def test_compute_incident_upper(self):
        """Test compute_incident with upper slice type"""
        value_matrix = np.array([[1, 2, 3], [4, 5, 6]])
        theta = 3
        result = compute_incident(value_matrix, theta, 'upper')
        expected = np.array([[0, 0, 1], [1, 1, 1]])
        np.testing.assert_array_equal(result, expected)
    
    def test_compute_incident_lower(self):
        """Test compute_incident with lower slice type"""
        value_matrix = np.array([[1, 2, 3], [4, 5, 6]])
        theta = 3
        result = compute_incident(value_matrix, theta, 'lower')
        expected = np.array([[1, 1, 1], [0, 0, 0]])
        np.testing.assert_array_equal(result, expected)
    
    def test_invert_pattern(self):
        """Test pattern inversion"""
        pattern = np.array([0.0, 1.0, 0.0, 1.0])
        result = invert_pattern(pattern)
        expected = np.array([1.0, 0.0, 1.0, 0.0])
        np.testing.assert_array_equal(result, expected)
    
    def test_normalize(self):
        """Test matrix normalization"""
        matrix = np.array([[1, 2], [3, 4]])
        result = normalize(matrix)
        # First column: (1-1)/(3-1) = 0, (3-1)/(3-1) = 1
        # Second column: (2-2)/(4-2) = 0, (4-2)/(4-2) = 1
        expected = np.array([[0.0, 0.0], [1.0, 1.0]])
        np.testing.assert_array_equal(result, expected)
    
    def test_sparse_graph_basic(self):
        """Test sparse graph creation"""
        incidence = np.array([[1, 0, 1], [0, 1, 1]])
        hyperedge_list = [0, 1, 2]
        theta = 1
        result = sparse_graph(incidence, hyperedge_list, theta)
        # Should return edges where incidence >= theta
        expected_edges = [(0, 0, 1), (0, 2, 1), (1, 1, 1), (1, 2, 1)]
        assert len(result) == len(expected_edges)
        for edge in expected_edges:
            assert edge in result
    
    def test_simple_ecc(self):
        """Test simple eccentricity calculation"""
        # Create a simple array for testing
        array = np.array([1, 3, 2, 1])
        result = simple_ecc(array)
        # qhat = max(array) - 1 = 3 - 1 = 2
        # qbottom = max(array without max element) - 1 = 2 - 1 = 1
        # ecc = (2 - 1) / (1 + 1) = 1/2 = 0.5
        expected = 0.5
        assert result == expected


class TestHyperGraphIntegration:
    """Integration tests for hyper_graph functions"""
    
    def test_basic_workflow(self):
        """Test a basic workflow with multiple functions"""
        # Create test data
        value_matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        theta = 5
        
        # Compute incident matrix
        incident = compute_incident(value_matrix, theta, 'upper')
        
        # Create sparse graph
        edges = sparse_graph(incident, range(len(incident[0])), 1)
        
        # Should have some edges
        assert len(edges) > 0
        
        # All edge values should be >= 1 (our theta for sparse_graph)
        for edge in edges:
            assert edge[2] >= 1


class TestErrorHandling:
    """Test error handling in hyper_graph functions"""
    
    def test_empty_arrays(self):
        """Test functions with empty arrays"""
        empty_array = np.array([])
        
        # invert_pattern should handle empty arrays
        result = invert_pattern(empty_array)
        assert len(result) == 0
    
    def test_invalid_slice_type(self):
        """Test compute_incident with invalid slice type"""
        value_matrix = np.array([[1, 2], [3, 4]])
        theta = 2
        
        # Should default to lower slice when slice_type is not 'upper'
        result = compute_incident(value_matrix, theta, 'invalid')
        expected = np.array([[1, 1], [0, 0]])
        np.testing.assert_array_equal(result, expected)