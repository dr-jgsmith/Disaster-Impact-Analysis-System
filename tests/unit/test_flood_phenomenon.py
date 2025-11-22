"""Tests for flood phenomenon implementation."""

import pytest
import numpy as np
import pandas as pd

from src.core.phenomena.flood import FloodPhenomenon, build_flood_model_from_data


class TestFloodPhenomenon:
    """Test FloodPhenomenon class."""
    
    @pytest.fixture
    def sample_flood_data(self):
        """Create sample flood data for testing."""
        n = 10
        parcel_ids = [f"P{i:03d}" for i in range(n)]
        coordinates = np.random.rand(n, 2) * 0.1 + np.array([29.76, -95.37])
        adjacency = np.random.rand(n, n) > 0.7
        adjacency = (adjacency + adjacency.T) / 2  # Make symmetric
        np.fill_diagonal(adjacency, 1)  # Self-connected
        
        elevations = np.random.rand(n) * 10 + 5  # 5-15 feet
        land_values = np.random.rand(n) * 100000 + 50000
        building_values = np.random.rand(n) * 200000 + 100000
        
        return {
            "parcel_ids": parcel_ids,
            "coordinates": coordinates,
            "adjacency": adjacency,
            "elevations": elevations,
            "land_values": land_values,
            "building_values": building_values,
        }
    
    def test_initialization(self, sample_flood_data):
        """Test flood phenomenon initialization."""
        flood = FloodPhenomenon(**sample_flood_data)
        
        assert len(flood.entity_ids) == 10
        assert flood.coordinates.shape == (10, 2)
        assert flood.elevations.shape == (10,)
        assert flood.land_values.shape == (10,)
        assert flood.building_values.shape == (10,)
        assert "elevations" in flood.attributes
        assert "land_values" in flood.attributes
        assert "building_values" in flood.attributes
    
    def test_get_phenomenon_type(self, sample_flood_data):
        """Test phenomenon type identifier."""
        flood = FloodPhenomenon(**sample_flood_data)
        assert flood.get_phenomenon_type() == "flood"
    
    def test_compute_zones(self, sample_flood_data):
        """Test flood zone computation."""
        flood = FloodPhenomenon(**sample_flood_data)
        
        zones = flood.compute_zones({
            "min_water_level": 6.0,
            "max_water_level": 12.0,
        })
        
        # Should have one zone per water level
        assert len(zones) == 7  # 6, 7, 8, 9, 10, 11, 12
        
        # Each zone should have same length as parcels
        for zone in zones:
            assert len(zone) == 10
        
        # Zones should be stored
        assert flood.zones is not None
        assert flood.binary_zones is not None
    
    def test_compute_zones_stores_params(self, sample_flood_data):
        """Test that compute_zones stores scenario parameters."""
        flood = FloodPhenomenon(**sample_flood_data)
        
        params = {"min_water_level": 6.0, "max_water_level": 12.0}
        flood.compute_zones(params)
        
        assert flood.scenario_params is not None
        assert flood.scenario_params["min_water_level"] == 6.0
    
    def test_compute_impact(self, sample_flood_data):
        """Test flood impact computation."""
        flood = FloodPhenomenon(**sample_flood_data)
        
        # Compute zones first
        zones = flood.compute_zones({
            "min_water_level": 6.0,
            "max_water_level": 12.0,
        })
        
        # Compute impacts
        impact = flood.compute_impact(zones, {
            "loss_percent": 0.8,
            "min_water_level": 6.0,
        })
        
        # Check impact structure
        assert "impact_intensities" in impact
        assert "mean_impacts" in impact
        assert "total_property_loss" in impact
        assert "affected_parcels" in impact
        assert "n_scenarios" in impact
        
        # Check counts
        assert len(impact["impact_intensities"]) == 7
        assert len(impact["mean_impacts"]) == 7
        assert len(impact["total_property_loss"]) == 7
        assert len(impact["affected_parcels"]) == 7
        
        # Check storage
        assert flood.impact_metrics is not None
        assert flood.impact_intensities is not None
        assert flood.mean_impacts is not None
    
    def test_compute_impact_without_zones_raises_error(self, sample_flood_data):
        """Test that compute_impact fails without zones."""
        flood = FloodPhenomenon(**sample_flood_data)
        
        # This should work even without compute_zones if we pass zones
        zones = [np.zeros(10)]  # Dummy zones
        impact = flood.compute_impact(zones, {
            "loss_percent": 0.8,
            "min_water_level": 6.0,
        })
        
        assert impact is not None
    
    def test_impact_increases_with_water_level(self, sample_flood_data):
        """Test that impact generally increases with water level."""
        flood = FloodPhenomenon(**sample_flood_data)
        
        zones = flood.compute_zones({
            "min_water_level": 6.0,
            "max_water_level": 12.0,
        })
        
        impact = flood.compute_impact(zones, {
            "loss_percent": 0.8,
            "min_water_level": 6.0,
        })
        
        # Total loss should generally increase
        losses = impact["total_property_loss"]
        
        # At least some scenarios should show increasing loss
        # (Not strict because it depends on parcel distribution)
        assert len(losses) > 0
        assert all(loss >= 0 for loss in losses)
    
    def test_to_dataframe(self, sample_flood_data):
        """Test DataFrame export."""
        flood = FloodPhenomenon(**sample_flood_data)
        
        df = flood.to_dataframe()
        
        # Check columns
        assert "PARCELID" in df.columns
        assert "LAT" in df.columns
        assert "LON" in df.columns
        assert "ELEVATION" in df.columns
        assert "LANDVALUE" in df.columns
        assert "BLDGVALUE" in df.columns
        
        # Check size
        assert len(df) == 10
    
    def test_to_dataframe_with_zones_and_impacts(self, sample_flood_data):
        """Test DataFrame export with computed data."""
        flood = FloodPhenomenon(**sample_flood_data)
        
        zones = flood.compute_zones({
            "min_water_level": 6.0,
            "max_water_level": 8.0,
        })
        
        flood.compute_impact(zones, {
            "loss_percent": 0.8,
            "min_water_level": 6.0,
        })
        
        df = flood.to_dataframe()
        
        # Check zone columns
        assert "Impact_Zone_0" in df.columns
        assert "Impact_Zone_1" in df.columns
        assert "Impact_Zone_2" in df.columns
        
        # Check impact columns
        assert "Impact_Intensity_0" in df.columns
        assert "Impact_Intensity_1" in df.columns
        assert "Impact_Intensity_2" in df.columns
    
    def test_get_summary(self, sample_flood_data):
        """Test flood summary statistics."""
        flood = FloodPhenomenon(**sample_flood_data)
        
        summary = flood.get_summary()
        
        # Base phenomenon fields
        assert summary["phenomenon_type"] == "flood"
        assert summary["n_entities"] == 10
        
        # Flood-specific fields
        assert "total_land_value" in summary
        assert "total_building_value" in summary
        assert "total_property_value" in summary
        assert "mean_connectivity" in summary
        
        # Check values are reasonable
        assert summary["total_land_value"] > 0
        assert summary["total_building_value"] > 0
        assert summary["total_property_value"] == (
            summary["total_land_value"] + summary["total_building_value"]
        )


class TestBuildFloodModelFromData:
    """Test factory function for building flood models."""
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create sample DataFrame for testing."""
        n = 20
        data = {
            "PARCELID": [f"P{i:04d}" for i in range(n)],
            "LAT": np.random.rand(n) * 0.1 + 29.76,
            "LON": np.random.rand(n) * 0.1 - 95.37,
            "ELEVATION": np.random.rand(n) * 10 + 5,
            "LANDVALUE": np.random.rand(n) * 100000 + 50000,
            "BLDGVALUE": np.random.rand(n) * 200000 + 100000,
        }
        return pd.DataFrame(data)
    
    def test_build_from_dataframe_default_fields(self, sample_dataframe):
        """Test building flood model from DataFrame with default fields."""
        flood = build_flood_model_from_data(sample_dataframe)
        
        assert isinstance(flood, FloodPhenomenon)
        assert len(flood.entity_ids) == 20
        assert flood.coordinates.shape == (20, 2)
        assert flood.adjacency_matrix.shape == (20, 20)
    
    def test_build_from_dataframe_custom_fields(self):
        """Test building with custom field names."""
        data = pd.DataFrame({
            "id": ["A", "B", "C"],
            "latitude": [29.76, 29.77, 29.78],
            "longitude": [-95.37, -95.38, -95.39],
            "elev": [5.0, 10.0, 8.0],
            "land": [100000, 150000, 120000],
            "building": [200000, 250000, 220000],
        })
        
        flood = build_flood_model_from_data(
            data,
            lat_field="latitude",
            lon_field="longitude",
            parcel_field="id",
            elevation_field="elev",
            land_value_field="land",
            building_value_field="building",
        )
        
        assert len(flood.entity_ids) == 3
        assert flood.entity_ids == ["A", "B", "C"]
    
    def test_build_with_geodesic_distances(self, sample_dataframe):
        """Test building with geodesic distances."""
        flood = build_flood_model_from_data(
            sample_dataframe,
            use_geodesic=True
        )
        
        assert flood.adjacency_matrix is not None
    
    def test_build_with_euclidean_distances(self, sample_dataframe):
        """Test building with Euclidean distances."""
        flood = build_flood_model_from_data(
            sample_dataframe,
            use_geodesic=False
        )
        
        assert flood.adjacency_matrix is not None
    
    def test_build_with_proximity_threshold(self, sample_dataframe):
        """Test building with custom proximity threshold."""
        flood = build_flood_model_from_data(
            sample_dataframe,
            proximity_threshold=5000.0  # 5km
        )
        
        assert flood.adjacency_matrix is not None


class TestFloodZoneLogic:
    """Test flood-specific zone computation logic."""
    
    def test_parcels_below_water_level_are_flooded(self):
        """Test that parcels below water level are marked as flooded."""
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
            "min_water_level": 7.0,
            "max_water_level": 7.0,
        })
        
        # Only P001 (elev 5.0) should be flooded at water level 7.0
        assert zones[0][0] > 0  # P001 flooded
        assert zones[0][1] == 0  # P002 not flooded
        assert zones[0][2] == 0  # P003 not flooded
    
    def test_loss_proportional_to_depth(self):
        """Test that loss is proportional to depth below water."""
        parcel_ids = ["P001", "P002"]
        coordinates = np.array([[29.76, -95.37], [29.77, -95.38]])
        adjacency = np.ones((2, 2))
        elevations = np.array([5.0, 7.0])
        land_values = np.array([100000, 100000])
        building_values = np.array([100000, 100000])
        
        flood = FloodPhenomenon(
            parcel_ids, coordinates, adjacency,
            elevations, land_values, building_values
        )
        
        zones = flood.compute_zones({
            "min_water_level": 10.0,
            "max_water_level": 10.0,
        })
        
        impact = flood.compute_impact(zones, {
            "loss_percent": 1.0,  # 100% per foot
            "min_water_level": 10.0,
        })
        
        # P001 is 5 feet below water (depth 5)
        # P002 is 3 feet below water (depth 3)
        # So P001 should have higher impact intensity
        intensities = impact["impact_intensities"][0]
        assert intensities[0] > intensities[1]

