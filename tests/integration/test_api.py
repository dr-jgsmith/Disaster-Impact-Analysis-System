"""
Integration tests for DIAS REST API.

These tests verify the complete API workflow including
phenomenon creation, computation, and visualization.
"""

import pytest
from fastapi.testclient import TestClient
import numpy as np

from src.api.main import app
from src.api.storage import PhenomenonStorage


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def sample_flood_data():
    """Create sample flood data for testing."""
    return {
        "phenomenon_type": "flood",
        "data": {
            "entity_ids": ["P001", "P002", "P003", "P004"],
            "coordinates": [
                [29.76, -95.37],
                [29.77, -95.38],
                [29.78, -95.39],
                [29.79, -95.40]
            ],
            "adjacency_matrix": [
                [1, 1, 0, 0],
                [1, 1, 1, 0],
                [0, 1, 1, 1],
                [0, 0, 1, 1]
            ],
            "attributes": {
                "elevations": [5.0, 10.0, 8.0, 12.0],
                "land_values": [100000, 150000, 120000, 180000],
                "building_values": [200000, 250000, 220000, 280000]
            }
        },
        "options": {
            "use_geodesic": True
        }
    }


class TestHealthEndpoints:
    """Test health and info endpoints."""
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert data["service"] == "DIAS API"
    
    def test_api_info(self, client):
        """Test API info endpoint."""
        response = client.get("/info")
        assert response.status_code == 200
        
        data = response.json()
        assert "DIAS" in data["name"]
        assert data["version"] == "2.0.0"
        assert "capabilities" in data
        assert "flood" in data["capabilities"]["phenomena_types"]
    
    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["name"] == "DIAS API"
        assert "documentation" in data


class TestPhenomenonCRUD:
    """Test phenomenon CRUD operations."""
    
    def test_create_phenomenon(self, client, sample_flood_data):
        """Test creating a phenomenon."""
        response = client.post("/api/v1/phenomena", json=sample_flood_data)
        assert response.status_code == 201
        
        data = response.json()
        assert "id" in data
        assert data["phenomenon_type"] == "flood"
        assert data["n_entities"] == 4
        assert data["status"] == "ready"
        assert "links" in data
    
    def test_get_phenomenon(self, client, sample_flood_data):
        """Test retrieving a phenomenon."""
        # Create phenomenon
        create_response = client.post("/api/v1/phenomena", json=sample_flood_data)
        phenom_id = create_response.json()["id"]
        
        # Get phenomenon
        response = client.get(f"/api/v1/phenomena/{phenom_id}")
        assert response.status_code == 200
        
        data = response.json()
        assert data["id"] == phenom_id
        assert data["phenomenon_type"] == "flood"
        assert data["n_entities"] == 4
        assert data["has_zones"] is False
        assert data["has_impacts"] is False
    
    def test_get_nonexistent_phenomenon(self, client):
        """Test retrieving non-existent phenomenon."""
        response = client.get("/api/v1/phenomena/nonexistent_id")
        assert response.status_code == 404
        
        data = response.json()
        assert "error" in data or "detail" in data
    
    def test_delete_phenomenon(self, client, sample_flood_data):
        """Test deleting a phenomenon."""
        # Create phenomenon
        create_response = client.post("/api/v1/phenomena", json=sample_flood_data)
        phenom_id = create_response.json()["id"]
        
        # Delete phenomenon
        response = client.delete(f"/api/v1/phenomena/{phenom_id}")
        assert response.status_code == 204
        
        # Verify deletion
        get_response = client.get(f"/api/v1/phenomena/{phenom_id}")
        assert get_response.status_code == 404
    
    def test_list_phenomena(self, client, sample_flood_data):
        """Test listing phenomena."""
        # Create a few phenomena
        for _ in range(3):
            client.post("/api/v1/phenomena", json=sample_flood_data)
        
        # List phenomena
        response = client.get("/api/v1/phenomena?page=1&page_size=10")
        assert response.status_code == 200
        
        data = response.json()
        assert "phenomena" in data
        assert "total" in data
        assert len(data["phenomena"]) >= 3


class TestComputation:
    """Test computation endpoints."""
    
    def test_compute_zones(self, client, sample_flood_data):
        """Test computing zones."""
        # Create phenomenon
        create_response = client.post("/api/v1/phenomena", json=sample_flood_data)
        phenom_id = create_response.json()["id"]
        
        # Compute zones
        zones_request = {
            "scenario_params": {
                "min_water_level": 6.0,
                "max_water_level": 11.0
            }
        }
        response = client.post(
            f"/api/v1/phenomena/{phenom_id}/zones",
            json=zones_request
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["phenomenon_id"] == phenom_id
        assert data["n_scenarios"] == 6  # 6, 7, 8, 9, 10, 11
        assert data["scenarios_computed"] is True
        assert "computation_time_ms" in data
    
    def test_compute_impact(self, client, sample_flood_data):
        """Test computing impact."""
        # Create phenomenon
        create_response = client.post("/api/v1/phenomena", json=sample_flood_data)
        phenom_id = create_response.json()["id"]
        
        # Compute zones first
        zones_request = {
            "scenario_params": {
                "min_water_level": 6.0,
                "max_water_level": 11.0
            }
        }
        client.post(f"/api/v1/phenomena/{phenom_id}/zones", json=zones_request)
        
        # Compute impact
        impact_request = {
            "scenario_params": {
                "loss_percent": 0.8,
                "min_water_level": 6.0
            }
        }
        response = client.post(
            f"/api/v1/phenomena/{phenom_id}/impact",
            json=impact_request
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["phenomenon_id"] == phenom_id
        assert "impact_metrics" in data
        assert "computation_time_ms" in data
    
    def test_compute_impact_without_zones(self, client, sample_flood_data):
        """Test that computing impact without zones fails."""
        # Create phenomenon
        create_response = client.post("/api/v1/phenomena", json=sample_flood_data)
        phenom_id = create_response.json()["id"]
        
        # Try to compute impact without zones
        impact_request = {
            "scenario_params": {
                "loss_percent": 0.8
            }
        }
        response = client.post(
            f"/api/v1/phenomena/{phenom_id}/impact",
            json=impact_request
        )
        assert response.status_code == 409  # Conflict


class TestVisualization:
    """Test visualization endpoints."""
    
    def test_get_geojson(self, client, sample_flood_data):
        """Test getting GeoJSON representation."""
        # Create and prepare phenomenon
        create_response = client.post("/api/v1/phenomena", json=sample_flood_data)
        phenom_id = create_response.json()["id"]
        
        # Compute zones
        zones_request = {
            "scenario_params": {
                "min_water_level": 6.0,
                "max_water_level": 8.0
            }
        }
        client.post(f"/api/v1/phenomena/{phenom_id}/zones", json=zones_request)
        
        # Get GeoJSON
        response = client.get(f"/api/v1/phenomena/{phenom_id}/geojson")
        assert response.status_code == 200
        
        geojson = response.json()
        assert geojson["type"] == "FeatureCollection"
        assert "features" in geojson
        assert len(geojson["features"]) == 4
        assert "metadata" in geojson
    
    def test_get_geojson_specific_scenario(self, client, sample_flood_data):
        """Test getting GeoJSON for specific scenario."""
        # Create and prepare phenomenon
        create_response = client.post("/api/v1/phenomena", json=sample_flood_data)
        phenom_id = create_response.json()["id"]
        
        # Compute zones
        zones_request = {
            "scenario_params": {
                "min_water_level": 6.0,
                "max_water_level": 8.0
            }
        }
        client.post(f"/api/v1/phenomena/{phenom_id}/zones", json=zones_request)
        
        # Get GeoJSON for scenario 1
        response = client.get(f"/api/v1/phenomena/{phenom_id}/geojson?scenario=1")
        assert response.status_code == 200
        
        geojson = response.json()
        # Check that zone property exists (not zone_0, zone_1, etc.)
        props = geojson["features"][0]["properties"]
        assert "zone" in props or "zone_0" in props
    
    def test_get_summary(self, client, sample_flood_data):
        """Test getting phenomenon summary."""
        # Create phenomenon
        create_response = client.post("/api/v1/phenomena", json=sample_flood_data)
        phenom_id = create_response.json()["id"]
        
        # Get summary
        response = client.get(f"/api/v1/phenomena/{phenom_id}/summary")
        assert response.status_code == 200
        
        data = response.json()
        assert data["phenomenon_id"] == phenom_id
        assert data["phenomenon_type"] == "flood"
        assert data["n_entities"] == 4
        assert "coordinate_bounds" in data
        assert "attribute_statistics" in data
    
    def test_get_zone_bounds(self, client, sample_flood_data):
        """Test getting zone bounds."""
        # Create and prepare phenomenon
        create_response = client.post("/api/v1/phenomena", json=sample_flood_data)
        phenom_id = create_response.json()["id"]
        
        # Compute zones
        zones_request = {
            "scenario_params": {
                "min_water_level": 6.0,
                "max_water_level": 8.0
            }
        }
        client.post(f"/api/v1/phenomena/{phenom_id}/zones", json=zones_request)
        
        # Get zone bounds
        response = client.get(f"/api/v1/phenomena/{phenom_id}/zones/0/bounds")
        assert response.status_code == 200
        
        data = response.json()
        assert data["phenomenon_id"] == phenom_id
        assert data["zone_index"] == 0
    
    def test_get_zone_stats(self, client, sample_flood_data):
        """Test getting zone statistics."""
        # Create and prepare phenomenon
        create_response = client.post("/api/v1/phenomena", json=sample_flood_data)
        phenom_id = create_response.json()["id"]
        
        # Compute zones
        zones_request = {
            "scenario_params": {
                "min_water_level": 6.0,
                "max_water_level": 8.0
            }
        }
        client.post(f"/api/v1/phenomena/{phenom_id}/zones", json=zones_request)
        
        # Get zone stats
        response = client.get(f"/api/v1/phenomena/{phenom_id}/zones/0/stats")
        assert response.status_code == 200
        
        data = response.json()
        assert data["phenomenon_id"] == phenom_id
        assert data["zone_index"] == 0


class TestCompleteWorkflow:
    """Test complete API workflow."""
    
    def test_full_flood_analysis_workflow(self, client, sample_flood_data):
        """Test complete flood analysis workflow."""
        # 1. Create phenomenon
        create_response = client.post("/api/v1/phenomena", json=sample_flood_data)
        assert create_response.status_code == 201
        phenom_id = create_response.json()["id"]
        
        # 2. Verify creation
        get_response = client.get(f"/api/v1/phenomena/{phenom_id}")
        assert get_response.status_code == 200
        assert get_response.json()["has_zones"] is False
        
        # 3. Compute zones
        zones_request = {
            "scenario_params": {
                "min_water_level": 6.0,
                "max_water_level": 11.0
            }
        }
        zones_response = client.post(
            f"/api/v1/phenomena/{phenom_id}/zones",
            json=zones_request
        )
        assert zones_response.status_code == 200
        assert zones_response.json()["n_scenarios"] == 6
        
        # 4. Compute impact
        impact_request = {
            "scenario_params": {
                "loss_percent": 0.8,
                "min_water_level": 6.0
            }
        }
        impact_response = client.post(
            f"/api/v1/phenomena/{phenom_id}/impact",
            json=impact_request
        )
        assert impact_response.status_code == 200
        
        # 5. Get GeoJSON
        geojson_response = client.get(f"/api/v1/phenomena/{phenom_id}/geojson")
        assert geojson_response.status_code == 200
        geojson = geojson_response.json()
        assert geojson["type"] == "FeatureCollection"
        assert len(geojson["features"]) == 4
        
        # 6. Get summary
        summary_response = client.get(f"/api/v1/phenomena/{phenom_id}/summary")
        assert summary_response.status_code == 200
        summary = summary_response.json()
        assert summary["n_entities"] == 4
        assert summary["n_scenarios"] == 6
        
        # 7. Clean up
        delete_response = client.delete(f"/api/v1/phenomena/{phenom_id}")
        assert delete_response.status_code == 204

