"""
Event management, computation, and visualization endpoints.
"""

import time
from typing import Optional
import numpy as np
from fastapi import APIRouter, HTTPException, Query, Depends, Response
from fastapi.responses import JSONResponse

from src.api.models import (
    CreateEventRequest,
    CreateEventResponse,
    ComputeZonesRequest,
    ComputeZonesResponse,
    ComputeImpactRequest,
    ComputeImpactResponse,
    EventInfo,
    EventSummary,
    ZoneBounds,
    ZoneStatistics,
    EventList,
    EventLinks,
    EventStatus,
    ErrorResponse,
)
from src.api.storage import EventStorage
from src.core.sp_events.flood_event import FloodEvent
from src.core.visualization.geojson import (
    sp_event_to_geojson,
    sp_event_to_geojson_with_impacts,
    get_zone_bounds,
    get_zone_statistics,
)


router = APIRouter()


# Dependency to get storage
def get_storage() -> EventStorage:
    """Get sp_event storage (will be injected from main app)."""
    from src.api.main import get_storage as _get_storage
    return _get_storage()


def get_sp_event_or_404(event_id: str, storage: EventStorage):
    """Get sp_event or raise 404."""
    phenom_data = storage.get(event_id)
    if not phenom_data:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "PHENOMENON_NOT_FOUND",
                "message": f"Event with ID '{event_id}' not found",
                "details": {"event_id": event_id},
            }
        )
    return phenom_data


def build_sp_event_links(event_id: str) -> EventLinks:
    """Build HATEOAS links for sp_event."""
    return EventLinks(
        self=f"/api/v1/sp_events/{event_id}",
        compute_zones=f"/api/v1/sp_events/{event_id}/zones",
        compute_impact=f"/api/v1/sp_events/{event_id}/impact",
        geojson=f"/api/v1/sp_events/{event_id}/geojson",
        summary=f"/api/v1/sp_events/{event_id}/summary",
    )


# ============================================================================
# Event CRUD Endpoints
# ============================================================================


@router.post(
    "/sp_events",
    response_model=CreateEventResponse,
    status_code=201,
    summary="Create a new spatial sp_event",
    description="Create a new spatial sp_event from data (flood, contagion, supply-chain, etc.)"
)
async def create_sp_event(
    request: CreateEventRequest,
    storage: EventStorage = Depends(get_storage),
):
    """Create a new spatial sp_event."""
    try:
        # Extract data
        data = request.data
        options = request.options or {}
        
        # Create sp_event based on type
        if request.event_type == "flood":
            # Convert lists to numpy arrays
            entity_ids = data["entity_ids"]
            coordinates = np.array(data["coordinates"])
            adjacency_matrix = np.array(data["adjacency_matrix"])
            attributes = data["attributes"]
            
            # Create FloodEvent
            sp_event = FloodEvent(
                parcel_ids=entity_ids,
                coordinates=coordinates,
                adjacency_matrix=adjacency_matrix,
                elevations=np.array(attributes["elevations"]),
                land_values=np.array(attributes["land_values"]),
                building_values=np.array(attributes["building_values"]),
            )
        else:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "UNSUPPORTED_PHENOMENON_TYPE",
                    "message": f"Event type '{request.event_type}' is not yet supported",
                    "details": {
                        "event_type": request.event_type,
                        "supported_types": ["flood"],
                    },
                }
            )
        
        # Store sp_event
        event_id = storage.create(sp_event, request.event_type.value)
        
        # Return response
        return CreateEventResponse(
            id=event_id,
            event_type=request.event_type,
            n_entities=len(sp_event.entity_ids),
            created_at=storage.get(event_id)["created_at"],
            status=EventStatus.READY,
            links=build_sp_event_links(event_id),
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "code": "CREATION_FAILED",
                "message": f"Failed to create sp_event: {str(e)}",
                "details": {"error": str(e)},
            }
        )


@router.get(
    "/sp_events/{event_id}",
    response_model=EventInfo,
    summary="Get sp_event information",
    description="Retrieve detailed information about a specific sp_event"
)
async def get_sp_event(
    event_id: str,
    storage: EventStorage = Depends(get_storage),
):
    """Get sp_event information."""
    phenom_data = get_sp_event_or_404(event_id, storage)
    sp_event = phenom_data["sp_event"]
    
    return EventInfo(
        id=event_id,
        event_type=phenom_data["event_type"],
        n_entities=len(sp_event.entity_ids),
        created_at=phenom_data["created_at"],
        status=phenom_data["status"],
        has_zones=sp_event.zones is not None,
        has_impacts=sp_event.impact_metrics is not None,
        n_scenarios=len(sp_event.zones) if sp_event.zones else None,
        links=build_sp_event_links(event_id),
    )


@router.delete(
    "/sp_events/{event_id}",
    status_code=204,
    summary="Delete a sp_event",
    description="Remove a sp_event from storage"
)
async def delete_sp_event(
    event_id: str,
    storage: EventStorage = Depends(get_storage),
):
    """Delete a sp_event."""
    if not storage.delete(event_id):
        raise HTTPException(
            status_code=404,
            detail={
                "code": "PHENOMENON_NOT_FOUND",
                "message": f"Event with ID '{event_id}' not found",
            }
        )
    
    return Response(status_code=204)


@router.get(
    "/sp_events",
    response_model=EventList,
    summary="List all sp_events",
    description="Get paginated list of all sp_events"
)
async def list_sp_events(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=100, description="Items per page"),
    storage: EventStorage = Depends(get_storage),
):
    """List all sp_events with pagination."""
    skip = (page - 1) * page_size
    sp_events_data = storage.list_all(skip=skip, limit=page_size)
    total = storage.count()
    total_pages = (total + page_size - 1) // page_size
    
    sp_events_list = []
    for item in sp_events_data:
        sp_event = item["sp_event"]
        sp_events_list.append(
            EventInfo(
                id=item["id"],
                event_type=item["event_type"],
                n_entities=len(sp_event.entity_ids),
                created_at=item["created_at"],
                status=item["status"],
                has_zones=sp_event.zones is not None,
                has_impacts=sp_event.impact_metrics is not None,
                n_scenarios=len(sp_event.zones) if sp_event.zones else None,
                links=build_sp_event_links(item["id"]),
            )
        )
    
    return EventList(
        sp_events=sp_events_list,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


# ============================================================================
# Computation Endpoints
# ============================================================================


@router.post(
    "/sp_events/{event_id}/zones",
    response_model=ComputeZonesResponse,
    summary="Compute zones/scenarios",
    description="Compute impact zones for different scenarios"
)
async def compute_zones(
    event_id: str,
    request: ComputeZonesRequest,
    storage: EventStorage = Depends(get_storage),
):
    """Compute zones/scenarios for sp_event."""
    phenom_data = get_sp_event_or_404(event_id, storage)
    sp_event = phenom_data["sp_event"]
    
    try:
        # Update status
        storage.update_status(event_id, "computing_zones")
        
        # Measure computation time
        start_time = time.time()
        
        # Compute zones
        zones = sp_event.compute_zones(request.scenario_params)
        
        computation_time_ms = (time.time() - start_time) * 1000
        
        # Update status
        storage.update_status(event_id, "zones_computed")
        
        return ComputeZonesResponse(
            event_id=event_id,
            n_scenarios=len(zones),
            scenarios_computed=True,
            computation_time_ms=computation_time_ms,
            links={
                "geojson": f"/api/v1/sp_events/{event_id}/geojson",
                "compute_impact": f"/api/v1/sp_events/{event_id}/impact",
                "summary": f"/api/v1/sp_events/{event_id}/summary",
            },
        )
    
    except Exception as e:
        storage.update_status(event_id, "error")
        raise HTTPException(
            status_code=500,
            detail={
                "code": "COMPUTATION_FAILED",
                "message": f"Failed to compute zones: {str(e)}",
                "details": {"error": str(e)},
            }
        )


@router.post(
    "/sp_events/{event_id}/impact",
    response_model=ComputeImpactResponse,
    summary="Compute impact metrics",
    description="Calculate impact metrics for computed zones"
)
async def compute_impact(
    event_id: str,
    request: ComputeImpactRequest,
    storage: EventStorage = Depends(get_storage),
):
    """Compute impact metrics for sp_event."""
    phenom_data = get_sp_event_or_404(event_id, storage)
    sp_event = phenom_data["sp_event"]
    
    # Check if zones have been computed
    if sp_event.zones is None:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "ZONES_NOT_COMPUTED",
                "message": "Zones must be computed before calculating impact",
                "details": {
                    "event_id": event_id,
                    "compute_zones_url": f"/api/v1/sp_events/{event_id}/zones",
                },
            }
        )
    
    try:
        # Update status
        storage.update_status(event_id, "computing_impact")
        
        # Measure computation time
        start_time = time.time()
        
        # Compute impact
        impact_metrics = sp_event.compute_impact(
            sp_event.zones,
            request.scenario_params
        )
        
        computation_time_ms = (time.time() - start_time) * 1000
        
        # Update status
        storage.update_status(event_id, "complete")
        
        return ComputeImpactResponse(
            event_id=event_id,
            impact_metrics=impact_metrics,
            computation_time_ms=computation_time_ms,
            links={
                "geojson": f"/api/v1/sp_events/{event_id}/geojson",
                "summary": f"/api/v1/sp_events/{event_id}/summary",
            },
        )
    
    except Exception as e:
        storage.update_status(event_id, "error")
        raise HTTPException(
            status_code=500,
            detail={
                "code": "COMPUTATION_FAILED",
                "message": f"Failed to compute impact: {str(e)}",
                "details": {"error": str(e)},
            }
        )


# ============================================================================
# Visualization Endpoints
# ============================================================================


@router.get(
    "/sp_events/{event_id}/geojson",
    summary="Get GeoJSON representation",
    description="Get sp_event data in GeoJSON format for visualization"
)
async def get_geojson(
    event_id: str,
    scenario: Optional[int] = Query(None, description="Specific scenario index"),
    include_zones: bool = Query(True, description="Include zone data"),
    include_impacts: bool = Query(True, description="Include impact data"),
    include_attributes: bool = Query(True, description="Include sp_event attributes"),
    storage: EventStorage = Depends(get_storage),
):
    """Get sp_event as GeoJSON FeatureCollection."""
    phenom_data = get_sp_event_or_404(event_id, storage)
    sp_event = phenom_data["sp_event"]
    
    try:
        if include_impacts and sp_event.impact_metrics:
            geojson = sp_event_to_geojson_with_impacts(
                sp_event,
                zone_index=scenario,
            )
        else:
            geojson = sp_event_to_geojson(
                sp_event,
                include_zones=include_zones,
                zone_index=scenario,
                include_attributes=include_attributes,
            )
        
        # Add sp_event ID to metadata
        geojson["metadata"]["event_id"] = event_id
        
        return JSONResponse(content=geojson)
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "code": "GEOJSON_GENERATION_FAILED",
                "message": f"Failed to generate GeoJSON: {str(e)}",
                "details": {"error": str(e)},
            }
        )


@router.get(
    "/sp_events/{event_id}/summary",
    response_model=EventSummary,
    summary="Get sp_event summary",
    description="Get summary statistics for sp_event"
)
async def get_summary(
    event_id: str,
    storage: EventStorage = Depends(get_storage),
):
    """Get sp_event summary statistics."""
    phenom_data = get_sp_event_or_404(event_id, storage)
    sp_event = phenom_data["sp_event"]
    
    try:
        summary = sp_event.get_summary()
        
        return EventSummary(
            event_id=event_id,
            event_type=phenom_data["event_type"],
            n_entities=summary["n_entities"],
            n_scenarios=summary.get("n_zones", 0),
            coordinate_bounds=summary["coordinate_bounds"],
            attribute_statistics={
                k: v for k, v in summary.items()
                if k.endswith("_range") or k.endswith("_mean")
            },
            impact_metrics=summary.get("impact_metrics"),
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "code": "SUMMARY_GENERATION_FAILED",
                "message": f"Failed to generate summary: {str(e)}",
                "details": {"error": str(e)},
            }
        )


@router.get(
    "/sp_events/{event_id}/zones/{zone_index}/bounds",
    response_model=ZoneBounds,
    summary="Get zone bounding box",
    description="Get geographic bounds for a specific zone"
)
async def get_zone_bounds_endpoint(
    event_id: str,
    zone_index: int,
    storage: EventStorage = Depends(get_storage),
):
    """Get bounding box for specific zone."""
    phenom_data = get_sp_event_or_404(event_id, storage)
    sp_event = phenom_data["sp_event"]
    
    bounds = get_zone_bounds(sp_event, zone_index)
    
    return ZoneBounds(
        event_id=event_id,
        zone_index=zone_index,
        bounds=bounds,
    )


@router.get(
    "/sp_events/{event_id}/zones/{zone_index}/stats",
    response_model=ZoneStatistics,
    summary="Get zone statistics",
    description="Get statistical summary for a specific zone"
)
async def get_zone_stats_endpoint(
    event_id: str,
    zone_index: int,
    storage: EventStorage = Depends(get_storage),
):
    """Get statistics for specific zone."""
    phenom_data = get_sp_event_or_404(event_id, storage)
    sp_event = phenom_data["sp_event"]
    
    stats = get_zone_statistics(sp_event, zone_index)
    
    return ZoneStatistics(
        event_id=event_id,
        zone_index=zone_index,
        statistics=stats,
    )

