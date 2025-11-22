"""Tests for abstract phenomenon base class."""

import pytest
import numpy as np
from typing import Dict, List, Any

from src.core.base.phenomenon import SpatialPhenomenon


class ConcretePhenomenon(SpatialPhenomenon):
    """Minimal concrete implementation for testing."""
    
    def compute_zones(self, scenario_params: Dict[str, Any]) -> List[np.ndarray]:
        """Dummy zone computation."""
        n = len(self.entity_ids)
        self.zones = [np.ones(n), np.zeros(n)]
        return self.zones
    
    def compute_impact(
        self, zones: List[np.ndarray], scenario_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Dummy impact computation."""
        self.impact_metrics = {"total_impact": 100.0}
        return self.impact_metrics
    
    def get_phenomenon_type(self) -> str:
        """Return test type."""
        return "test"


class TestSpatialPhenomenonAbstract:
    """Test abstract base class behavior."""
    
    def test_cannot_instantiate_abstract_class(self):
        """Test that abstract class cannot be instantiated directly."""
        with pytest.raises(TypeError):
            SpatialPhenomenon([], np.array([]), np.array([]), {})
    
    def test_subclass_must_implement_compute_zones(self):
        """Test that subclass must implement compute_zones."""
        class IncompletePhenomenon1(SpatialPhenomenon):
            def compute_impact(self, zones, params):
                return {}
            
            def get_phenomenon_type(self):
                return "incomplete"
        
        with pytest.raises(TypeError):
            IncompletePhenomenon1([], np.array([]), np.array([]), {})
    
    def test_subclass_must_implement_compute_impact(self):
        """Test that subclass must implement compute_impact."""
        class IncompletePhenomenon2(SpatialPhenomenon):
            def compute_zones(self, params):
                return []
            
            def get_phenomenon_type(self):
                return "incomplete"
        
        with pytest.raises(TypeError):
            IncompletePhenomenon2([], np.array([]), np.array([]), {})
    
    def test_subclass_must_implement_get_phenomenon_type(self):
        """Test that subclass must implement get_phenomenon_type."""
        class IncompletePhenomenon3(SpatialPhenomenon):
            def compute_zones(self, params):
                return []
            
            def compute_impact(self, zones, params):
                return {}
        
        with pytest.raises(TypeError):
            IncompletePhenomenon3([], np.array([]), np.array([]), {})


class TestSpatialPhenomenonConcrete:
    """Test concrete implementation of base class."""
    
    @pytest.fixture
    def sample_phenomenon(self):
        """Create sample phenomenon for testing."""
        entity_ids = ["E001", "E002", "E003"]
        coordinates = np.array([[29.76, -95.37], [29.77, -95.38], [29.78, -95.39]])
        adjacency = np.array([[1, 1, 0], [1, 1, 1], [0, 1, 1]])
        attributes = {
            "attr1": np.array([1.0, 2.0, 3.0]),
            "attr2": np.array([10.0, 20.0, 30.0]),
        }
        
        return ConcretePhenomenon(entity_ids, coordinates, adjacency, attributes)
    
    def test_initialization(self, sample_phenomenon):
        """Test phenomenon initialization."""
        assert len(sample_phenomenon.entity_ids) == 3
        assert sample_phenomenon.coordinates.shape == (3, 2)
        assert sample_phenomenon.adjacency_matrix.shape == (3, 3)
        assert len(sample_phenomenon.attributes) == 2
        assert sample_phenomenon.zones is None
        assert sample_phenomenon.impact_metrics is None
    
    def test_compute_zones(self, sample_phenomenon):
        """Test zone computation."""
        zones = sample_phenomenon.compute_zones({"param": "value"})
        assert len(zones) == 2
        assert sample_phenomenon.zones is not None
    
    def test_compute_impact(self, sample_phenomenon):
        """Test impact computation."""
        zones = sample_phenomenon.compute_zones({})
        impact = sample_phenomenon.compute_impact(zones, {})
        assert "total_impact" in impact
        assert sample_phenomenon.impact_metrics is not None
    
    def test_get_phenomenon_type(self, sample_phenomenon):
        """Test phenomenon type identifier."""
        assert sample_phenomenon.get_phenomenon_type() == "test"
    
    def test_to_dict(self, sample_phenomenon):
        """Test dictionary export."""
        data = sample_phenomenon.to_dict()
        
        assert data["phenomenon_type"] == "test"
        assert data["n_entities"] == 3
        assert data["has_zones"] is False
        assert data["has_impacts"] is False
        assert "coordinate_bounds" in data
    
    def test_to_dict_with_zones_and_impacts(self, sample_phenomenon):
        """Test dictionary export with computed data."""
        sample_phenomenon.compute_zones({})
        sample_phenomenon.compute_impact(sample_phenomenon.zones, {})
        
        data = sample_phenomenon.to_dict()
        assert data["has_zones"] is True
        assert data["has_impacts"] is True
    
    def test_to_dataframe(self, sample_phenomenon):
        """Test DataFrame export."""
        df = sample_phenomenon.to_dataframe()
        
        assert len(df) == 3
        assert "entity_id" in df.columns
        assert "lat" in df.columns
        assert "lon" in df.columns
        assert "attr1" in df.columns
        assert "attr2" in df.columns
    
    def test_to_dataframe_with_zones(self, sample_phenomenon):
        """Test DataFrame export with zones."""
        sample_phenomenon.compute_zones({})
        df = sample_phenomenon.to_dataframe()
        
        assert "zone_0" in df.columns
        assert "zone_1" in df.columns
    
    def test_get_summary(self, sample_phenomenon):
        """Test summary statistics."""
        summary = sample_phenomenon.get_summary()
        
        assert summary["phenomenon_type"] == "test"
        assert summary["n_entities"] == 3
        assert summary["n_zones"] == 0
        assert "attr1_range" in summary
        assert "attr1_mean" in summary
        assert summary["attr1_range"] == (1.0, 3.0)
        assert summary["attr1_mean"] == 2.0
    
    def test_get_summary_with_impacts(self, sample_phenomenon):
        """Test summary with impact metrics."""
        sample_phenomenon.compute_zones({})
        sample_phenomenon.compute_impact(sample_phenomenon.zones, {})
        
        summary = sample_phenomenon.get_summary()
        assert "impact_metrics" in summary
        assert summary["n_zones"] == 2
    
    def test_coordinate_bounds(self, sample_phenomenon):
        """Test coordinate bounds calculation."""
        bounds = sample_phenomenon._get_coordinate_bounds()
        
        assert bounds["min_lat"] == 29.76
        assert bounds["max_lat"] == 29.78
        assert bounds["min_lon"] == -95.39
        assert bounds["max_lon"] == -95.37
    
    def test_repr(self, sample_phenomenon):
        """Test string representation."""
        repr_str = repr(sample_phenomenon)
        
        assert "ConcretePhenomenon" in repr_str
        assert "type=test" in repr_str
        assert "entities=3" in repr_str

