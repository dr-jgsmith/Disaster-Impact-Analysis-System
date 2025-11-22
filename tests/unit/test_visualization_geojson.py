"""Tests for GeoJSON visualization utilities."""

import pytest
import numpy as np
import tempfile
import json
import os

from src.core.phenomena.flood import FloodPhenomenon
from src.core.visualization.geojson import (
    phenomenon_to_geojson,
    phenomenon_to_geojson_with_impacts,
    get_zone_bounds,
    get_zone_statistics,
    export_all_scenarios,
)


class TestPhenomenonToGeoJSON:
    """Test basic GeoJSON conversion."""
    
    @pytest.fixture
    def sample_flood(self):
        """Create sample flood phenomenon."""
        parcel_ids = ["P001", "P002", "P003"]
        coordinates = np.array([[29.76, -95.37], [29.77, -95.38], [29.78, -95.39]])
        adjacency = np.ones((3, 3))
        elevations = np.array([5.0, 10.0, 8.0])
        land_values = np.array([100000, 150000, 120000])
        building_values = np.array([200000, 250000, 220000])
        
        return FloodPhenomenon(
            parcel_ids, coordinates, adjacency,
            elevations, land_values, building_values
        )
    
    def test_basic_geojson_structure(self, sample_flood):
        """Test basic GeoJSON structure."""
        geojson = phenomenon_to_geojson(sample_flood)
        
        # Check top-level structure
        assert geojson["type"] == "FeatureCollection"
        assert "features" in geojson
        assert "metadata" in geojson
        
        # Check features
        assert len(geojson["features"]) == 3
    
    def test_feature_geometry(self, sample_flood):
        """Test feature geometry format."""
        geojson = phenomenon_to_geojson(sample_flood)
        
        feature = geojson["features"][0]
        
        # Check geometry
        assert feature["geometry"]["type"] == "Point"
        assert len(feature["geometry"]["coordinates"]) == 2
        
        # Coordinates should be [lon, lat] (GeoJSON standard)
        lon, lat = feature["geometry"]["coordinates"]
        assert lon == -95.37
        assert lat == 29.76
    
    def test_feature_properties(self, sample_flood):
        """Test feature properties."""
        geojson = phenomenon_to_geojson(sample_flood)
        
        feature = geojson["features"][0]
        props = feature["properties"]
        
        # Check base properties
        assert props["id"] == "P001"
        assert props["phenomenon_type"] == "flood"
        assert "entity_index" in props
        
        # Check attributes (by default included)
        assert "elevations" in props
        assert "land_values" in props
        assert "building_values" in props
    
    def test_metadata(self, sample_flood):
        """Test metadata."""
        geojson = phenomenon_to_geojson(sample_flood)
        
        metadata = geojson["metadata"]
        
        assert metadata["phenomenon_type"] == "flood"
        assert metadata["n_entities"] == 3
        assert metadata["has_zones"] is False
        assert "coordinate_bounds" in metadata
    
    def test_exclude_attributes(self, sample_flood):
        """Test excluding attributes from properties."""
        geojson = phenomenon_to_geojson(
            sample_flood,
            include_attributes=False
        )
        
        props = geojson["features"][0]["properties"]
        
        assert "elevations" not in props
        assert "land_values" not in props
        assert "id" in props  # Base properties still included
    
    def test_include_zones_without_computation(self, sample_flood):
        """Test zone inclusion when zones not computed."""
        geojson = phenomenon_to_geojson(
            sample_flood,
            include_zones=True
        )
        
        # Should work but zones won't be in properties
        props = geojson["features"][0]["properties"]
        assert "zone" not in props
    
    def test_include_all_zones(self, sample_flood):
        """Test including all zones."""
        # Compute zones
        sample_flood.compute_zones({
            "min_water_level": 6.0,
            "max_water_level": 8.0,
        })
        
        geojson = phenomenon_to_geojson(
            sample_flood,
            include_zones=True,
            zone_index=None  # All zones
        )
        
        props = geojson["features"][0]["properties"]
        
        # Should have zone_0, zone_1, zone_2
        assert "zone_0" in props
        assert "zone_1" in props
        assert "zone_2" in props
    
    def test_include_specific_zone(self, sample_flood):
        """Test including specific zone."""
        sample_flood.compute_zones({
            "min_water_level": 6.0,
            "max_water_level": 8.0,
        })
        
        geojson = phenomenon_to_geojson(
            sample_flood,
            include_zones=True,
            zone_index=1
        )
        
        props = geojson["features"][0]["properties"]
        
        # Should have single "zone" property
        assert "zone" in props
        assert "zone_0" not in props
        assert "zone_1" not in props


class TestPhenomenonToGeoJSONWithImpacts:
    """Test GeoJSON conversion with impact data."""
    
    @pytest.fixture
    def sample_flood_with_impacts(self):
        """Create flood with computed zones and impacts."""
        parcel_ids = ["P001", "P002", "P003"]
        coordinates = np.array([[29.76, -95.37], [29.77, -95.38], [29.78, -95.39]])
        adjacency = np.ones((3, 3))
        elevations = np.array([5.0, 10.0, 8.0])
        land_values = np.array([100000, 150000, 120000])
        building_values = np.array([200000, 250000, 220000])
        
        flood = FloodPhenomenon(
            parcel_ids, coordinates, adjacency,
            elevations, land_values, building_values
        )
        
        zones = flood.compute_zones({
            "min_water_level": 6.0,
            "max_water_level": 8.0,
        })
        
        flood.compute_impact(zones, {
            "loss_percent": 0.8,
            "min_water_level": 6.0,
        })
        
        return flood
    
    def test_impact_metadata(self, sample_flood_with_impacts):
        """Test impact metrics in metadata."""
        geojson = phenomenon_to_geojson_with_impacts(sample_flood_with_impacts)
        
        metadata = geojson["metadata"]
        
        assert "impact_metrics" in metadata
        assert "total_property_loss" in metadata["impact_metrics"]
        assert "affected_parcels" in metadata["impact_metrics"]
    
    def test_impact_properties_specific_zone(self, sample_flood_with_impacts):
        """Test impact in feature properties for specific zone."""
        geojson = phenomenon_to_geojson_with_impacts(
            sample_flood_with_impacts,
            zone_index=0
        )
        
        props = geojson["features"][0]["properties"]
        
        assert "impact_intensity" in props
        assert isinstance(props["impact_intensity"], float)
    
    def test_impact_properties_all_zones(self, sample_flood_with_impacts):
        """Test impact in properties for all zones."""
        geojson = phenomenon_to_geojson_with_impacts(
            sample_flood_with_impacts,
            zone_index=None
        )
        
        props = geojson["features"][0]["properties"]
        
        # Should have impact_0, impact_1, impact_2
        assert "impact_0" in props
        assert "impact_1" in props
        assert "impact_2" in props


class TestGetZoneBounds:
    """Test zone bounding box calculation."""
    
    @pytest.fixture
    def sample_flood_with_zones(self):
        """Create flood with zones."""
        parcel_ids = [f"P{i:03d}" for i in range(10)]
        coordinates = np.random.rand(10, 2) * 0.1 + np.array([29.76, -95.37])
        adjacency = np.ones((10, 10))
        elevations = np.random.rand(10) * 10 + 5
        land_values = np.random.rand(10) * 100000 + 50000
        building_values = np.random.rand(10) * 200000 + 100000
        
        flood = FloodPhenomenon(
            parcel_ids, coordinates, adjacency,
            elevations, land_values, building_values
        )
        
        flood.compute_zones({
            "min_water_level": 6.0,
            "max_water_level": 12.0,
        })
        
        return flood
    
    def test_get_bounds_valid_zone(self, sample_flood_with_zones):
        """Test getting bounds for valid zone."""
        bounds = get_zone_bounds(sample_flood_with_zones, zone_index=0)
        
        if bounds is not None:  # May be None if no affected entities
            assert "min_lat" in bounds
            assert "max_lat" in bounds
            assert "min_lon" in bounds
            assert "max_lon" in bounds
            
            # Sanity checks
            assert bounds["min_lat"] <= bounds["max_lat"]
            assert bounds["min_lon"] <= bounds["max_lon"]
    
    def test_get_bounds_invalid_zone(self, sample_flood_with_zones):
        """Test getting bounds for invalid zone."""
        bounds = get_zone_bounds(sample_flood_with_zones, zone_index=999)
        assert bounds is None
    
    def test_get_bounds_no_zones(self):
        """Test getting bounds when no zones computed."""
        flood = FloodPhenomenon(
            ["P001"], np.array([[29.76, -95.37]]), np.array([[1]]),
            np.array([5.0]), np.array([100000]), np.array([200000])
        )
        
        bounds = get_zone_bounds(flood, zone_index=0)
        assert bounds is None


class TestGetZoneStatistics:
    """Test zone statistics calculation."""
    
    @pytest.fixture
    def sample_flood_with_zones(self):
        """Create flood with zones."""
        parcel_ids = ["P001", "P002", "P003", "P004"]
        coordinates = np.array([
            [29.76, -95.37],
            [29.77, -95.38],
            [29.78, -95.39],
            [29.79, -95.40]
        ])
        adjacency = np.ones((4, 4))
        elevations = np.array([5.0, 10.0, 8.0, 12.0])
        land_values = np.array([100000, 150000, 120000, 180000])
        building_values = np.array([200000, 250000, 220000, 280000])
        
        flood = FloodPhenomenon(
            parcel_ids, coordinates, adjacency,
            elevations, land_values, building_values
        )
        
        flood.compute_zones({
            "min_water_level": 7.0,
            "max_water_level": 9.0,
        })
        
        return flood
    
    def test_get_statistics_valid_zone(self, sample_flood_with_zones):
        """Test getting statistics for valid zone."""
        stats = get_zone_statistics(sample_flood_with_zones, zone_index=0)
        
        if stats is not None:  # May be None if no affected entities
            assert "zone_index" in stats
            assert "n_entities" in stats
            assert "percent_affected" in stats
            assert "unique_zones" in stats
            
            # Check attribute statistics
            assert "elevations_affected" in stats
            assert "land_values_affected" in stats
    
    def test_get_statistics_invalid_zone(self, sample_flood_with_zones):
        """Test statistics for invalid zone."""
        stats = get_zone_statistics(sample_flood_with_zones, zone_index=999)
        assert stats is None


class TestExportAllScenarios:
    """Test exporting all scenarios to files."""
    
    def test_export_creates_files(self):
        """Test that export creates GeoJSON files."""
        # Create simple flood
        flood = FloodPhenomenon(
            ["P001", "P002"],
            np.array([[29.76, -95.37], [29.77, -95.38]]),
            np.ones((2, 2)),
            np.array([5.0, 10.0]),
            np.array([100000, 150000]),
            np.array([200000, 250000])
        )
        
        flood.compute_zones({
            "min_water_level": 6.0,
            "max_water_level": 8.0,
        })
        
        # Export to temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            files = export_all_scenarios(flood, tmpdir, prefix="test")
            
            # Check files were created
            assert len(files) == 3  # 3 scenarios
            
            for filepath in files:
                assert os.path.exists(filepath)
                assert filepath.endswith(".geojson")
                
                # Check file content is valid JSON
                with open(filepath, "r") as f:
                    data = json.load(f)
                    assert data["type"] == "FeatureCollection"
    
    def test_export_without_zones(self):
        """Test export without zones returns empty list."""
        flood = FloodPhenomenon(
            ["P001"],
            np.array([[29.76, -95.37]]),
            np.ones((1, 1)),
            np.array([5.0]),
            np.array([100000]),
            np.array([200000])
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            files = export_all_scenarios(flood, tmpdir)
            assert len(files) == 0

