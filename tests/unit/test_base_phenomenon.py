"""Tests for abstract sp_event base class."""

import pytest
import numpy as np
from typing import Dict, List, Any

from src.core.base.sp_event import SpatialEvent


class ConcreteEvent(SpatialEvent):
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
    
    def get_event_type(self) -> str:
        """Return test type."""
        return "test"


class TestSpatialEventAbstract:
    """Test abstract base class behavior."""
    
    def test_cannot_instantiate_abstract_class(self):
        """Test that abstract class cannot be instantiated directly."""
        with pytest.raises(TypeError):
            SpatialEvent([], np.array([]), np.array([]), {})
    
    def test_subclass_must_implement_compute_zones(self):
        """Test that subclass must implement compute_zones."""
        class IncompleteEvent1(SpatialEvent):
            def compute_impact(self, zones, params):
                return {}
            
            def get_event_type(self):
                return "incomplete"
        
        with pytest.raises(TypeError):
            IncompleteEvent1([], np.array([]), np.array([]), {})
    
    def test_subclass_must_implement_compute_impact(self):
        """Test that subclass must implement compute_impact."""
        class IncompleteEvent2(SpatialEvent):
            def compute_zones(self, params):
                return []
            
            def get_event_type(self):
                return "incomplete"
        
        with pytest.raises(TypeError):
            IncompleteEvent2([], np.array([]), np.array([]), {})
    
    def test_subclass_must_implement_get_event_type(self):
        """Test that subclass must implement get_event_type."""
        class IncompleteEvent3(SpatialEvent):
            def compute_zones(self, params):
                return []
            
            def compute_impact(self, zones, params):
                return {}
        
        with pytest.raises(TypeError):
            IncompleteEvent3([], np.array([]), np.array([]), {})


class TestSpatialEventConcrete:
    """Test concrete implementation of base class."""
    
    @pytest.fixture
    def sample_sp_event(self):
        """Create sample sp_event for testing."""
        entity_ids = ["E001", "E002", "E003"]
        coordinates = np.array([[29.76, -95.37], [29.77, -95.38], [29.78, -95.39]])
        adjacency = np.array([[1, 1, 0], [1, 1, 1], [0, 1, 1]])
        attributes = {
            "attr1": np.array([1.0, 2.0, 3.0]),
            "attr2": np.array([10.0, 20.0, 30.0]),
        }
        
        return ConcreteEvent(entity_ids, coordinates, adjacency, attributes)
    
    def test_initialization(self, sample_sp_event):
        """Test sp_event initialization."""
        assert len(sample_sp_event.entity_ids) == 3
        assert sample_sp_event.coordinates.shape == (3, 2)
        assert sample_sp_event.adjacency_matrix.shape == (3, 3)
        assert len(sample_sp_event.attributes) == 2
        assert sample_sp_event.zones is None
        assert sample_sp_event.impact_metrics is None
    
    def test_compute_zones(self, sample_sp_event):
        """Test zone computation."""
        zones = sample_sp_event.compute_zones({"param": "value"})
        assert len(zones) == 2
        assert sample_sp_event.zones is not None
    
    def test_compute_impact(self, sample_sp_event):
        """Test impact computation."""
        zones = sample_sp_event.compute_zones({})
        impact = sample_sp_event.compute_impact(zones, {})
        assert "total_impact" in impact
        assert sample_sp_event.impact_metrics is not None
    
    def test_get_event_type(self, sample_sp_event):
        """Test sp_event type identifier."""
        assert sample_sp_event.get_event_type() == "test"
    
    def test_to_dict(self, sample_sp_event):
        """Test dictionary export."""
        data = sample_sp_event.to_dict()
        
        assert data["event_type"] == "test"
        assert data["n_entities"] == 3
        assert data["has_zones"] is False
        assert data["has_impacts"] is False
        assert "coordinate_bounds" in data
    
    def test_to_dict_with_zones_and_impacts(self, sample_sp_event):
        """Test dictionary export with computed data."""
        sample_sp_event.compute_zones({})
        sample_sp_event.compute_impact(sample_sp_event.zones, {})
        
        data = sample_sp_event.to_dict()
        assert data["has_zones"] is True
        assert data["has_impacts"] is True
    
    def test_to_dataframe(self, sample_sp_event):
        """Test DataFrame export."""
        df = sample_sp_event.to_dataframe()
        
        assert len(df) == 3
        assert "entity_id" in df.columns
        assert "lat" in df.columns
        assert "lon" in df.columns
        assert "attr1" in df.columns
        assert "attr2" in df.columns
    
    def test_to_dataframe_with_zones(self, sample_sp_event):
        """Test DataFrame export with zones."""
        sample_sp_event.compute_zones({})
        df = sample_sp_event.to_dataframe()
        
        assert "zone_0" in df.columns
        assert "zone_1" in df.columns
    
    def test_get_summary(self, sample_sp_event):
        """Test summary statistics."""
        summary = sample_sp_event.get_summary()
        
        assert summary["event_type"] == "test"
        assert summary["n_entities"] == 3
        assert summary["n_zones"] == 0
        assert "attr1_range" in summary
        assert "attr1_mean" in summary
        assert summary["attr1_range"] == (1.0, 3.0)
        assert summary["attr1_mean"] == 2.0
    
    def test_get_summary_with_impacts(self, sample_sp_event):
        """Test summary with impact metrics."""
        sample_sp_event.compute_zones({})
        sample_sp_event.compute_impact(sample_sp_event.zones, {})
        
        summary = sample_sp_event.get_summary()
        assert "impact_metrics" in summary
        assert summary["n_zones"] == 2
    
    def test_coordinate_bounds(self, sample_sp_event):
        """Test coordinate bounds calculation."""
        bounds = sample_sp_event._get_coordinate_bounds()
        
        assert bounds["min_lat"] == 29.76
        assert bounds["max_lat"] == 29.78
        assert bounds["min_lon"] == -95.39
        assert bounds["max_lon"] == -95.37
    
    def test_repr(self, sample_sp_event):
        """Test string representation."""
        repr_str = repr(sample_sp_event)
        
        assert "ConcreteEvent" in repr_str
        assert "type=test" in repr_str
        assert "entities=3" in repr_str

