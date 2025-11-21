"""
Unit tests for core model building functions.

Tests the disaster impact model construction and zone computation.
"""

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from src.core import model


class TestConnectivityMatrix:
    """Test connectivity matrix construction."""
    
    def test_build_connectivity_euclidean(self):
        """Test Euclidean connectivity matrix."""
        coords = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 10.0]])
        
        adj_matrix = model.build_connectivity_matrix(
            coords, use_geodesic=False, proximity_threshold=2.0
        )
        
        # Points 0 and 1 should be connected (distance 1.0)
        # Point 2 should not be connected to others (distance >10)
        assert adj_matrix[0, 1] == 1
        assert adj_matrix[1, 0] == 1
        assert adj_matrix[0, 2] == 0
        assert adj_matrix[2, 0] == 0
    
    def test_build_connectivity_geodesic(self):
        """Test geodesic connectivity matrix."""
        # Houston area coordinates
        coords = np.array([
            [29.7604, -95.3698],
            [29.7605, -95.3699],
            [30.0000, -96.0000]  # Far away
        ])
        
        adj_matrix = model.build_connectivity_matrix(
            coords, use_geodesic=True, proximity_threshold=500.0
        )
        
        # First two points should be connected (very close)
        assert adj_matrix[0, 1] == 1
        # Third point should not be connected
        assert adj_matrix[0, 2] == 0
    
    def test_build_connectivity_auto_threshold(self):
        """Test automatic threshold computation."""
        coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        
        adj_matrix = model.build_connectivity_matrix(
            coords, use_geodesic=False, proximity_threshold=None
        )
        
        # Should automatically determine threshold
        assert adj_matrix.shape == (3, 3)
        # Diagonal should be all 1s (self-connected)
        assert np.all(np.diag(adj_matrix) == 1)


class TestImpactZones:
    """Test impact zone computation."""
    
    def test_compute_impact_zones_basic(self):
        """Test basic impact zone computation."""
        elevations = np.array([5.0, 10.0, 8.0, 6.0])
        adjacency = np.ones((4, 4))  # All connected
        
        zones = model.compute_impact_zones(elevations, adjacency, 3, 12)
        
        # Should have one zone array per water level
        assert len(zones) == 10  # 12 - 3 + 1
        
        # At water level 7, parcels 0 and 3 should be flooded
        zone_7 = zones[4]  # Index 4 = water level 7
        assert zone_7[0] > 0  # Parcel 0 flooded (elevation 5)
        assert zone_7[1] == 0  # Parcel 1 not flooded (elevation 10)
        assert zone_7[3] > 0  # Parcel 3 flooded (elevation 6)
    
    def test_compute_impact_zones_no_flooding(self):
        """Test zones with no flooding."""
        elevations = np.array([10.0, 12.0, 15.0])
        adjacency = np.ones((3, 3))
        
        zones = model.compute_impact_zones(elevations, adjacency, 3, 8)
        
        # No parcels should be flooded (all elevations > max water level)
        for zone in zones:
            assert np.all(zone == 0)
    
    def test_compute_impact_zones_all_flooded(self):
        """Test zones where all parcels flood."""
        elevations = np.array([2.0, 3.0, 4.0])
        adjacency = np.ones((3, 3))
        
        zones = model.compute_impact_zones(elevations, adjacency, 5, 10)
        
        # All parcels should be flooded at every water level
        for zone in zones:
            assert np.all(zone > 0)


class TestBinarizeZones:
    """Test zone binarization."""
    
    def test_binarize_zones(self):
        """Test converting zones to binary."""
        zones = [
            np.array([0.0, 1.0, 1.0, 0.0]),
            np.array([0.0, 2.0, 2.0, 1.0]),
            np.array([1.0, 1.0, 2.0, 2.0])
        ]
        
        binary_zones = model.binarize_zones(zones)
        
        assert len(binary_zones) == 3
        
        # First zone: [0, 1, 1, 0]
        np.testing.assert_array_equal(
            binary_zones[0],
            jnp.array([0.0, 1.0, 1.0, 0.0])
        )
        
        # Second zone: [0, 1, 1, 1] (all non-zero become 1)
        np.testing.assert_array_equal(
            binary_zones[1],
            jnp.array([0.0, 1.0, 1.0, 1.0])
        )


class TestLossMultiplier:
    """Test loss multiplier computation."""
    
    def test_compute_loss_multiplier_basic(self):
        """Test basic loss multiplier calculation."""
        elevations = np.array([5.0, 10.0, 8.0])
        binary_zone = jnp.array([1.0, 0.0, 1.0])
        water_level = 9.0
        loss_percent = 0.8
        
        multiplier = model.compute_loss_multiplier(
            elevations, binary_zone, water_level, loss_percent
        )
        
        # Parcel 0: 5ft elevation, 4ft below water = 0.8 * 4 = 3.2
        assert multiplier[0] > 3.0
        # Parcel 1: not flooded = 0
        assert multiplier[1] == 0.0
        # Parcel 2: 8ft elevation, 1ft below water = 0.8 * 1 = 0.8
        assert multiplier[2] > 0.5
    
    def test_compute_loss_multiplier_no_flooding(self):
        """Test loss with no flooding."""
        elevations = np.array([10.0, 12.0])
        binary_zone = jnp.array([0.0, 0.0])
        
        multiplier = model.compute_loss_multiplier(
            elevations, binary_zone, 5.0, 0.8
        )
        
        # No flooding = no loss
        np.testing.assert_array_equal(multiplier, jnp.array([0.0, 0.0]))


class TestImpactIntensities:
    """Test impact intensity computation."""
    
    def test_compute_impact_intensities(self):
        """Test impact intensity calculation."""
        elevations = np.array([5.0, 10.0, 8.0])
        binary_zones = [
            jnp.array([1.0, 0.0, 0.0]),
            jnp.array([1.0, 0.0, 1.0]),
        ]
        loss_percent = 0.8
        min_water_level = 6.0
        
        impacts, means = model.compute_impact_intensities(
            elevations, binary_zones, loss_percent, min_water_level
        )
        
        assert len(impacts) == 2
        assert len(means) == 2
        
        # Means should be >= 0
        assert all(m >= 0 for m in means)
        
        # Second scenario should have higher mean (more flooding)
        assert means[1] >= means[0]


class TestDisasterImpactModel:
    """Test complete model class."""
    
    @pytest.fixture
    def sample_model(self):
        """Create sample model for testing."""
        parcel_ids = ["P001", "P002", "P003", "P004"]
        coordinates = np.array([
            [29.76, -95.37],
            [29.77, -95.38],
            [29.78, -95.39],
            [29.79, -95.40]
        ])
        elevations = np.array([5.0, 10.0, 8.0, 6.0])
        land_values = np.array([50000, 60000, 55000, 52000])
        building_values = np.array([200000, 250000, 225000, 210000])
        
        return model.DisasterImpactModel(
            parcel_ids=parcel_ids,
            coordinates=coordinates,
            elevations=elevations,
            land_values=land_values,
            building_values=building_values,
            use_geodesic=True,
        )
    
    def test_model_initialization(self, sample_model):
        """Test model initializes correctly."""
        assert len(sample_model.parcel_ids) == 4
        assert sample_model.adjacency_matrix.shape == (4, 4)
        assert sample_model.zones is None  # Not computed yet
    
    def test_compute_zones(self, sample_model):
        """Test zone computation."""
        sample_model.compute_zones(3, 12)
        
        assert sample_model.zones is not None
        assert len(sample_model.zones) == 10  # 12 - 3 + 1
        assert sample_model.binary_zones is not None
        assert len(sample_model.binary_zones) == 10
    
    def test_compute_impacts(self, sample_model):
        """Test impact computation."""
        sample_model.compute_zones(3, 12)
        sample_model.compute_impacts(3, 0.8)
        
        assert sample_model.impact_intensities is not None
        assert sample_model.mean_impacts is not None
        assert len(sample_model.mean_impacts) == 10
    
    def test_compute_impacts_before_zones_fails(self, sample_model):
        """Test that computing impacts without zones fails."""
        with pytest.raises(ValueError, match="Must call compute_zones"):
            sample_model.compute_impacts(3, 0.8)
    
    def test_get_summary(self, sample_model):
        """Test model summary generation."""
        summary = sample_model.get_summary()
        
        assert summary["n_parcels"] == 4
        assert "elevation_range" in summary
        assert "total_land_value" in summary
        assert "total_building_value" in summary
        assert summary["total_land_value"] == 217000  # Sum of land values
    
    def test_to_dataframe(self, sample_model):
        """Test DataFrame export."""
        sample_model.compute_zones(3, 12)
        sample_model.compute_impacts(3, 0.8)
        
        df = sample_model.to_dataframe()
        
        assert len(df) == 4
        assert "PARCELID" in df.columns
        assert "ELEVATION" in df.columns
        # Should have zone columns
        assert "Impact_Zone_0" in df.columns
        # Should have intensity columns
        assert "Impact_Intensity_0" in df.columns


class TestBuildModelFromData:
    """Test building model from DataFrame."""
    
    @pytest.fixture
    def sample_dataframe(self):
        """Create sample DataFrame."""
        return pd.DataFrame({
            "PARCELID": ["P001", "P002", "P003"],
            "LAT": [29.76, 29.77, 29.78],
            "LON": [-95.37, -95.38, -95.39],
            "ELEVATION": [5.0, 10.0, 8.0],
            "LANDVALUE": [50000, 60000, 55000],
            "BLDGVALUE": [200000, 250000, 225000],
        })
    
    def test_build_model_from_data(self, sample_dataframe):
        """Test model building from DataFrame."""
        built_model = model.build_model_from_data(sample_dataframe)
        
        assert len(built_model.parcel_ids) == 3
        assert built_model.coordinates.shape == (3, 2)
        assert len(built_model.elevations) == 3
    
    def test_build_model_custom_fields(self, sample_dataframe):
        """Test model building with custom field names."""
        # Rename columns
        df = sample_dataframe.rename(columns={
            "LAT": "Latitude",
            "LON": "Longitude",
            "ELEVATION": "Elev"
        })
        
        built_model = model.build_model_from_data(
            df,
            lat_field="Latitude",
            lon_field="Longitude",
            elevation_field="Elev"
        )
        
        assert len(built_model.parcel_ids) == 3
    
    def test_full_workflow(self, sample_dataframe):
        """Test complete workflow from DataFrame to results."""
        # Build model
        built_model = model.build_model_from_data(sample_dataframe)
        
        # Compute zones
        built_model.compute_zones(3, 12)
        
        # Compute impacts
        built_model.compute_impacts(3, 0.8)
        
        # Get summary
        summary = built_model.get_summary()
        assert summary["n_parcels"] == 3
        assert summary["n_zones"] == 10
        
        # Export to DataFrame
        result_df = built_model.to_dataframe()
        assert len(result_df) == 3
        assert "Impact_Zone_0" in result_df.columns


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_single_parcel(self):
        """Test model with single parcel."""
        model_single = model.DisasterImpactModel(
            parcel_ids=["P001"],
            coordinates=np.array([[29.76, -95.37]]),
            elevations=np.array([5.0]),
            land_values=np.array([50000]),
            building_values=np.array([200000]),
        )
        
        model_single.compute_zones(3, 10)
        model_single.compute_impacts(3, 0.8)
        
        summary = model_single.get_summary()
        assert summary["n_parcels"] == 1
    
    def test_all_same_elevation(self):
        """Test model with constant elevation."""
        elevations = np.array([10.0, 10.0, 10.0])
        adjacency = np.ones((3, 3))
        
        zones = model.compute_impact_zones(elevations, adjacency, 5, 12)
        
        # All parcels should have same flooding status at each level
        for zone in zones:
            unique_nonzero = np.unique(zone[zone > 0])
            # Either all flooded (one zone) or none flooded
            assert len(unique_nonzero) <= 1
    
    def test_zero_loss_percent(self):
        """Test with zero loss percentage."""
        elevations = np.array([5.0, 10.0])
        binary_zone = jnp.array([1.0, 1.0])
        
        multiplier = model.compute_loss_multiplier(
            elevations, binary_zone, 12.0, 0.0
        )
        
        # Zero loss = zero multiplier
        np.testing.assert_array_equal(multiplier, jnp.array([0.0, 0.0]))


# Integration test with real test data
@pytest.mark.skip(reason="Requires test data files, run manually")
class TestWithRealData:
    """Integration tests with generated test data."""
    
    def test_with_test_data(self):
        """Test model with generated test data."""
        # This would require test data to be generated first
        import os
        
        if not os.path.exists("test_data/parcels_test.csv"):
            pytest.skip("Test data not generated")
        
        # Load test data
        df = pd.read_csv("test_data/parcels_test.csv")
        elevations_df = pd.read_csv("test_data/elevations_test.csv")
        
        # Merge elevation data
        df = df.merge(elevations_df[["PARCELID", "ELEVATION"]], on="PARCELID")
        
        # Build model
        test_model = model.build_model_from_data(df)
        
        # Run analysis
        test_model.compute_zones(0, 20)
        test_model.compute_impacts(0, 0.8)
        
        # Verify results
        summary = test_model.get_summary()
        assert summary["n_parcels"] > 0
        assert summary["n_zones"] == 21  # 20 - 0 + 1

