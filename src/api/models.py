"""
Pydantic models for API request/response validation.

This module defines all data models used in the DIAS REST API,
ensuring type safety and automatic validation.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, ConfigDict


# ============================================================================
# Enums
# ============================================================================


class PhenomenonType(str, Enum):
    """Supported phenomenon types."""
    FLOOD = "flood"
    CONTAGION = "contagion"
    SUPPLY_CHAIN = "supply_chain"


class PhenomenonStatus(str, Enum):
    """Phenomenon computation status."""
    READY = "ready"
    COMPUTING_ZONES = "computing_zones"
    ZONES_COMPUTED = "zones_computed"
    COMPUTING_IMPACT = "computing_impact"
    COMPLETE = "complete"
    ERROR = "error"


# ============================================================================
# Request Models
# ============================================================================


class CreatePhenomenonRequest(BaseModel):
    """Request to create a new phenomenon."""
    
    phenomenon_type: PhenomenonType = Field(
        ...,
        description="Type of spatial phenomenon"
    )
    
    data: Dict[str, Any] = Field(
        ...,
        description="Phenomenon data (entity_ids, coordinates, adjacency_matrix, attributes)"
    )
    
    options: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional configuration (use_geodesic, proximity_threshold, etc.)"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "phenomenon_type": "flood",
                "data": {
                    "entity_ids": ["P001", "P002", "P003"],
                    "coordinates": [[29.76, -95.37], [29.77, -95.38], [29.78, -95.39]],
                    "adjacency_matrix": [[1, 1, 0], [1, 1, 1], [0, 1, 1]],
                    "attributes": {
                        "elevations": [5.0, 10.0, 8.0],
                        "land_values": [100000, 150000, 120000],
                        "building_values": [200000, 250000, 220000]
                    }
                },
                "options": {
                    "use_geodesic": True,
                    "proximity_threshold": None
                }
            }
        }
    )


class ComputeZonesRequest(BaseModel):
    """Request to compute zones/scenarios."""
    
    scenario_params: Dict[str, Any] = Field(
        ...,
        description="Scenario parameters (phenomenon-specific)"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "scenario_params": {
                    "min_water_level": 3.0,
                    "max_water_level": 14.0
                }
            }
        }
    )


class ComputeImpactRequest(BaseModel):
    """Request to compute impact metrics."""
    
    scenario_params: Dict[str, Any] = Field(
        ...,
        description="Impact parameters (phenomenon-specific)"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "scenario_params": {
                    "loss_percent": 0.8,
                    "min_water_level": 3.0
                }
            }
        }
    )


# ============================================================================
# Response Models
# ============================================================================


class PhenomenonLinks(BaseModel):
    """HATEOAS links for phenomenon resource."""
    
    self: str
    compute_zones: str
    compute_impact: str
    geojson: str
    summary: str


class CreatePhenomenonResponse(BaseModel):
    """Response after creating a phenomenon."""
    
    id: str = Field(..., description="Unique phenomenon identifier")
    phenomenon_type: PhenomenonType = Field(..., description="Type of phenomenon")
    n_entities: int = Field(..., description="Number of spatial entities")
    created_at: datetime = Field(..., description="Creation timestamp")
    status: PhenomenonStatus = Field(..., description="Computation status")
    links: PhenomenonLinks = Field(..., description="HATEOAS links")


class PhenomenonInfo(BaseModel):
    """Detailed phenomenon information."""
    
    id: str
    phenomenon_type: PhenomenonType
    n_entities: int
    created_at: datetime
    status: PhenomenonStatus
    has_zones: bool
    has_impacts: bool
    n_scenarios: Optional[int] = None
    links: PhenomenonLinks


class ComputeZonesResponse(BaseModel):
    """Response after computing zones."""
    
    phenomenon_id: str
    n_scenarios: int
    scenarios_computed: bool
    computation_time_ms: float
    links: Dict[str, str]


class ComputeImpactResponse(BaseModel):
    """Response after computing impact."""
    
    phenomenon_id: str
    impact_metrics: Dict[str, Any]
    computation_time_ms: float
    links: Dict[str, str]


class PhenomenonSummary(BaseModel):
    """Summary statistics for phenomenon."""
    
    phenomenon_id: str
    phenomenon_type: PhenomenonType
    n_entities: int
    n_scenarios: int
    coordinate_bounds: Dict[str, float]
    attribute_statistics: Dict[str, Any]
    impact_metrics: Optional[Dict[str, Any]] = None


class ZoneBounds(BaseModel):
    """Bounding box for a specific zone."""
    
    phenomenon_id: str
    zone_index: int
    bounds: Optional[Dict[str, float]]


class ZoneStatistics(BaseModel):
    """Statistics for a specific zone."""
    
    phenomenon_id: str
    zone_index: int
    statistics: Optional[Dict[str, Any]]


class PhenomenonList(BaseModel):
    """Paginated list of phenomena."""
    
    phenomena: List[PhenomenonInfo]
    total: int
    page: int
    page_size: int
    total_pages: int


# ============================================================================
# Error Models
# ============================================================================


class ErrorDetail(BaseModel):
    """Detailed error information."""
    
    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional error details"
    )


class ErrorResponse(BaseModel):
    """Standard error response."""
    
    error: ErrorDetail

