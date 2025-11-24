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


class EventType(str, Enum):
    """Supported sp_event types."""
    FLOOD = "flood"
    CONTAGION = "contagion"
    SUPPLY_CHAIN = "supply_chain"


class EventStatus(str, Enum):
    """Event computation status."""
    READY = "ready"
    COMPUTING_ZONES = "computing_zones"
    ZONES_COMPUTED = "zones_computed"
    COMPUTING_IMPACT = "computing_impact"
    COMPLETE = "complete"
    ERROR = "error"


# ============================================================================
# Request Models
# ============================================================================


class CreateEventRequest(BaseModel):
    """Request to create a new sp_event."""
    
    event_type: EventType = Field(
        ...,
        description="Type of spatial sp_event"
    )
    
    data: Dict[str, Any] = Field(
        ...,
        description="Event data (entity_ids, coordinates, adjacency_matrix, attributes)"
    )
    
    options: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional configuration (use_geodesic, proximity_threshold, etc.)"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "event_type": "flood",
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
        description="Scenario parameters (sp_event-specific)"
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
        description="Impact parameters (sp_event-specific)"
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


class EventLinks(BaseModel):
    """HATEOAS links for sp_event resource."""
    
    self: str
    compute_zones: str
    compute_impact: str
    geojson: str
    summary: str


class CreateEventResponse(BaseModel):
    """Response after creating a sp_event."""
    
    id: str = Field(..., description="Unique sp_event identifier")
    event_type: EventType = Field(..., description="Type of sp_event")
    n_entities: int = Field(..., description="Number of spatial entities")
    created_at: datetime = Field(..., description="Creation timestamp")
    status: EventStatus = Field(..., description="Computation status")
    links: EventLinks = Field(..., description="HATEOAS links")


class EventInfo(BaseModel):
    """Detailed sp_event information."""
    
    id: str
    event_type: EventType
    n_entities: int
    created_at: datetime
    status: EventStatus
    has_zones: bool
    has_impacts: bool
    n_scenarios: Optional[int] = None
    links: EventLinks


class ComputeZonesResponse(BaseModel):
    """Response after computing zones."""
    
    event_id: str
    n_scenarios: int
    scenarios_computed: bool
    computation_time_ms: float
    links: Dict[str, str]


class ComputeImpactResponse(BaseModel):
    """Response after computing impact."""
    
    event_id: str
    impact_metrics: Dict[str, Any]
    computation_time_ms: float
    links: Dict[str, str]


class EventSummary(BaseModel):
    """Summary statistics for sp_event."""
    
    event_id: str
    event_type: EventType
    n_entities: int
    n_scenarios: int
    coordinate_bounds: Dict[str, float]
    attribute_statistics: Dict[str, Any]
    impact_metrics: Optional[Dict[str, Any]] = None


class ZoneBounds(BaseModel):
    """Bounding box for a specific zone."""
    
    event_id: str
    zone_index: int
    bounds: Optional[Dict[str, float]]


class ZoneStatistics(BaseModel):
    """Statistics for a specific zone."""
    
    event_id: str
    zone_index: int
    statistics: Optional[Dict[str, Any]]


class EventList(BaseModel):
    """Paginated list of sp_events."""
    
    sp_events: List[EventInfo]
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

