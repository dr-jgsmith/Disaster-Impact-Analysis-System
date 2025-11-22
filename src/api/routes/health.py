"""
Health check and system information endpoints.
"""

from datetime import datetime
from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns system health status and basic information.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "DIAS API",
        "version": "2.0.0",
    }


@router.get("/info")
async def api_info():
    """
    API information endpoint.
    
    Returns API version, capabilities, and supported phenomena types.
    """
    return {
        "name": "DIAS - Disaster Impact Analysis System",
        "version": "2.0.0",
        "description": "Multi-phenomenon spatial analysis and visualization API",
        "capabilities": {
            "phenomena_types": ["flood", "contagion", "supply_chain"],
            "visualization": ["geojson", "summary_statistics"],
            "computation": ["zones", "impacts"],
            "export": ["geojson", "csv", "dataframe"],
        },
        "api_version": "v1",
        "documentation": {
            "swagger": "/docs",
            "redoc": "/redoc",
            "openapi": "/api/openapi.json",
        },
    }

