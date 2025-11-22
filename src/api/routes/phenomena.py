"""
Phenomenon management, computation, and visualization endpoints.
"""

import time
from typing import Optional
import numpy as np
from fastapi import APIRouter, HTTPException, Query, Depends, Response
from fastapi.responses import JSONResponse

from src.api.models import (
    CreatePhenomenonRequest,
    CreatePhenomenonResponse,
    ComputeZonesRequest,
    ComputeZonesResponse,
    ComputeImpactRequest,
    ComputeImpactResponse,
    PhenomenonInfo,
    PhenomenonSummary,
    ZoneBounds,
    ZoneStatistics,
    PhenomenonList,
    PhenomenonLinks,
    PhenomenonStatus,
    ErrorResponse,
)
from src.api.storage import PhenomenonStorage
from src.core.phenomena.flood import FloodPhenomenon
from src.core.visualization.geojson import (
    phenomenon_to_geojson,
    phenomenon_to_geojson_with_impacts,
    get_zone_bounds,
    get_zone_statistics,
)


router = APIRouter()


# Dependency to get storage
def get_storage() -> PhenomenonStorage:
    """Get phenomenon storage (will be injected from main app)."""
    from src.api.main import get_storage as _get_storage
    return _get_storage()


def get_phenomenon_or_404(phenomenon_id: str, storage: PhenomenonStorage):
    """Get phenomenon or raise 404."""
    phenom_data = storage.get(phenomenon_id)
    if not phenom_data:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "PHENOMENON_NOT_FOUND",
                "message": f"Phenomenon with ID '{phenomenon_id}' not found",
                "details": {"phenomenon_id": phenomenon_id},
            }
        )
    return phenom_data


def build_phenomenon_links(phenomenon_id: str) -> PhenomenonLinks:
    """Build HATEOAS links for phenomenon."""
    return PhenomenonLinks(
        self=f"/api/v1/phenomena/{phenomenon_id}",
        compute_zones=f"/api/v1/phenomena/{phenomenon_id}/zones",
        compute_impact=f"/api/v1/phenomena/{phenomenon_id}/impact",
        geojson=f"/api/v1/phenomena/{phenomenon_id}/geojson",
        summary=f"/api/v1/phenomena/{phenomenon_id}/summary",
    )


# ============================================================================
# Phenomenon CRUD Endpoints
# ============================================================================


@router.post(
    "/phenomena",
    response_model=CreatePhenomenonResponse,
    status_code=201,
    summary="Create a new spatial phenomenon",
    description="Create a new spatial phenomenon from data (flood, contagion, supply-chain, etc.)"
)
async def create_phenomenon(
    request: CreatePhenomenonRequest,
    storage: PhenomenonStorage = Depends(get_storage),
):
    """Create a new spatial phenomenon."""
    try:
        # Extract data
        data = request.data
        options = request.options or {}
        
        # Create phenomenon based on type
        if request.phenomenon_type == "flood":
            # Convert lists to numpy arrays
            entity_ids = data["entity_ids"]
            coordinates = np.array(data["coordinates"])
            adjacency_matrix = np.array(data["adjacency_matrix"])
            attributes = data["attributes"]
            
            # Create FloodPhenomenon
            phenomenon = FloodPhenomenon(
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
                    "message": f"Phenomenon type '{request.phenomenon_type}' is not yet supported",
                    "details": {
                        "phenomenon_type": request.phenomenon_type,
                        "supported_types": ["flood"],
                    },
                }
            )
        
        # Store phenomenon
        phenomenon_id = storage.create(phenomenon, request.phenomenon_type.value)
        
        # Return response
        return CreatePhenomenonResponse(
            id=phenomenon_id,
            phenomenon_type=request.phenomenon_type,
            n_entities=len(phenomenon.entity_ids),
            created_at=storage.get(phenomenon_id)["created_at"],
            status=PhenomenonStatus.READY,
            links=build_phenomenon_links(phenomenon_id),
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "code": "CREATION_FAILED",
                "message": f"Failed to create phenomenon: {str(e)}",
                "details": {"error": str(e)},
            }
        )


@router.get(
    "/phenomena/{phenomenon_id}",
    response_model=PhenomenonInfo,
    summary="Get phenomenon information",
    description="Retrieve detailed information about a specific phenomenon"
)
async def get_phenomenon(
    phenomenon_id: str,
    storage: PhenomenonStorage = Depends(get_storage),
):
    """Get phenomenon information."""
    phenom_data = get_phenomenon_or_404(phenomenon_id, storage)
    phenomenon = phenom_data["phenomenon"]
    
    return PhenomenonInfo(
        id=phenomenon_id,
        phenomenon_type=phenom_data["phenomenon_type"],
        n_entities=len(phenomenon.entity_ids),
        created_at=phenom_data["created_at"],
        status=phenom_data["status"],
        has_zones=phenomenon.zones is not None,
        has_impacts=phenomenon.impact_metrics is not None,
        n_scenarios=len(phenomenon.zones) if phenomenon.zones else None,
        links=build_phenomenon_links(phenomenon_id),
    )


@router.delete(
    "/phenomena/{phenomenon_id}",
    status_code=204,
    summary="Delete a phenomenon",
    description="Remove a phenomenon from storage"
)
async def delete_phenomenon(
    phenomenon_id: str,
    storage: PhenomenonStorage = Depends(get_storage),
):
    """Delete a phenomenon."""
    if not storage.delete(phenomenon_id):
        raise HTTPException(
            status_code=404,
            detail={
                "code": "PHENOMENON_NOT_FOUND",
                "message": f"Phenomenon with ID '{phenomenon_id}' not found",
            }
        )
    
    return Response(status_code=204)


@router.get(
    "/phenomena",
    response_model=PhenomenonList,
    summary="List all phenomena",
    description="Get paginated list of all phenomena"
)
async def list_phenomena(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=100, description="Items per page"),
    storage: PhenomenonStorage = Depends(get_storage),
):
    """List all phenomena with pagination."""
    skip = (page - 1) * page_size
    phenomena_data = storage.list_all(skip=skip, limit=page_size)
    total = storage.count()
    total_pages = (total + page_size - 1) // page_size
    
    phenomena_list = []
    for item in phenomena_data:
        phenomenon = item["phenomenon"]
        phenomena_list.append(
            PhenomenonInfo(
                id=item["id"],
                phenomenon_type=item["phenomenon_type"],
                n_entities=len(phenomenon.entity_ids),
                created_at=item["created_at"],
                status=item["status"],
                has_zones=phenomenon.zones is not None,
                has_impacts=phenomenon.impact_metrics is not None,
                n_scenarios=len(phenomenon.zones) if phenomenon.zones else None,
                links=build_phenomenon_links(item["id"]),
            )
        )
    
    return PhenomenonList(
        phenomena=phenomena_list,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


# ============================================================================
# Computation Endpoints
# ============================================================================


@router.post(
    "/phenomena/{phenomenon_id}/zones",
    response_model=ComputeZonesResponse,
    summary="Compute zones/scenarios",
    description="Compute impact zones for different scenarios"
)
async def compute_zones(
    phenomenon_id: str,
    request: ComputeZonesRequest,
    storage: PhenomenonStorage = Depends(get_storage),
):
    """Compute zones/scenarios for phenomenon."""
    phenom_data = get_phenomenon_or_404(phenomenon_id, storage)
    phenomenon = phenom_data["phenomenon"]
    
    try:
        # Update status
        storage.update_status(phenomenon_id, "computing_zones")
        
        # Measure computation time
        start_time = time.time()
        
        # Compute zones
        zones = phenomenon.compute_zones(request.scenario_params)
        
        computation_time_ms = (time.time() - start_time) * 1000
        
        # Update status
        storage.update_status(phenomenon_id, "zones_computed")
        
        return ComputeZonesResponse(
            phenomenon_id=phenomenon_id,
            n_scenarios=len(zones),
            scenarios_computed=True,
            computation_time_ms=computation_time_ms,
            links={
                "geojson": f"/api/v1/phenomena/{phenomenon_id}/geojson",
                "compute_impact": f"/api/v1/phenomena/{phenomenon_id}/impact",
                "summary": f"/api/v1/phenomena/{phenomenon_id}/summary",
            },
        )
    
    except Exception as e:
        storage.update_status(phenomenon_id, "error")
        raise HTTPException(
            status_code=500,
            detail={
                "code": "COMPUTATION_FAILED",
                "message": f"Failed to compute zones: {str(e)}",
                "details": {"error": str(e)},
            }
        )


@router.post(
    "/phenomena/{phenomenon_id}/impact",
    response_model=ComputeImpactResponse,
    summary="Compute impact metrics",
    description="Calculate impact metrics for computed zones"
)
async def compute_impact(
    phenomenon_id: str,
    request: ComputeImpactRequest,
    storage: PhenomenonStorage = Depends(get_storage),
):
    """Compute impact metrics for phenomenon."""
    phenom_data = get_phenomenon_or_404(phenomenon_id, storage)
    phenomenon = phenom_data["phenomenon"]
    
    # Check if zones have been computed
    if phenomenon.zones is None:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "ZONES_NOT_COMPUTED",
                "message": "Zones must be computed before calculating impact",
                "details": {
                    "phenomenon_id": phenomenon_id,
                    "compute_zones_url": f"/api/v1/phenomena/{phenomenon_id}/zones",
                },
            }
        )
    
    try:
        # Update status
        storage.update_status(phenomenon_id, "computing_impact")
        
        # Measure computation time
        start_time = time.time()
        
        # Compute impact
        impact_metrics = phenomenon.compute_impact(
            phenomenon.zones,
            request.scenario_params
        )
        
        computation_time_ms = (time.time() - start_time) * 1000
        
        # Update status
        storage.update_status(phenomenon_id, "complete")
        
        return ComputeImpactResponse(
            phenomenon_id=phenomenon_id,
            impact_metrics=impact_metrics,
            computation_time_ms=computation_time_ms,
            links={
                "geojson": f"/api/v1/phenomena/{phenomenon_id}/geojson",
                "summary": f"/api/v1/phenomena/{phenomenon_id}/summary",
            },
        )
    
    except Exception as e:
        storage.update_status(phenomenon_id, "error")
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
    "/phenomena/{phenomenon_id}/geojson",
    summary="Get GeoJSON representation",
    description="Get phenomenon data in GeoJSON format for visualization"
)
async def get_geojson(
    phenomenon_id: str,
    scenario: Optional[int] = Query(None, description="Specific scenario index"),
    include_zones: bool = Query(True, description="Include zone data"),
    include_impacts: bool = Query(True, description="Include impact data"),
    include_attributes: bool = Query(True, description="Include phenomenon attributes"),
    storage: PhenomenonStorage = Depends(get_storage),
):
    """Get phenomenon as GeoJSON FeatureCollection."""
    phenom_data = get_phenomenon_or_404(phenomenon_id, storage)
    phenomenon = phenom_data["phenomenon"]
    
    try:
        if include_impacts and phenomenon.impact_metrics:
            geojson = phenomenon_to_geojson_with_impacts(
                phenomenon,
                zone_index=scenario,
            )
        else:
            geojson = phenomenon_to_geojson(
                phenomenon,
                include_zones=include_zones,
                zone_index=scenario,
                include_attributes=include_attributes,
            )
        
        # Add phenomenon ID to metadata
        geojson["metadata"]["phenomenon_id"] = phenomenon_id
        
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
    "/phenomena/{phenomenon_id}/summary",
    response_model=PhenomenonSummary,
    summary="Get phenomenon summary",
    description="Get summary statistics for phenomenon"
)
async def get_summary(
    phenomenon_id: str,
    storage: PhenomenonStorage = Depends(get_storage),
):
    """Get phenomenon summary statistics."""
    phenom_data = get_phenomenon_or_404(phenomenon_id, storage)
    phenomenon = phenom_data["phenomenon"]
    
    try:
        summary = phenomenon.get_summary()
        
        return PhenomenonSummary(
            phenomenon_id=phenomenon_id,
            phenomenon_type=phenom_data["phenomenon_type"],
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
    "/phenomena/{phenomenon_id}/zones/{zone_index}/bounds",
    response_model=ZoneBounds,
    summary="Get zone bounding box",
    description="Get geographic bounds for a specific zone"
)
async def get_zone_bounds_endpoint(
    phenomenon_id: str,
    zone_index: int,
    storage: PhenomenonStorage = Depends(get_storage),
):
    """Get bounding box for specific zone."""
    phenom_data = get_phenomenon_or_404(phenomenon_id, storage)
    phenomenon = phenom_data["phenomenon"]
    
    bounds = get_zone_bounds(phenomenon, zone_index)
    
    return ZoneBounds(
        phenomenon_id=phenomenon_id,
        zone_index=zone_index,
        bounds=bounds,
    )


@router.get(
    "/phenomena/{phenomenon_id}/zones/{zone_index}/stats",
    response_model=ZoneStatistics,
    summary="Get zone statistics",
    description="Get statistical summary for a specific zone"
)
async def get_zone_stats_endpoint(
    phenomenon_id: str,
    zone_index: int,
    storage: PhenomenonStorage = Depends(get_storage),
):
    """Get statistics for specific zone."""
    phenom_data = get_phenomenon_or_404(phenomenon_id, storage)
    phenomenon = phenom_data["phenomenon"]
    
    stats = get_zone_statistics(phenomenon, zone_index)
    
    return ZoneStatistics(
        phenomenon_id=phenomenon_id,
        zone_index=zone_index,
        statistics=stats,
    )

