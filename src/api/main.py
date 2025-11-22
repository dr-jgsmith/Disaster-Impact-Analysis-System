"""
FastAPI main application for DIAS.

This module provides the REST API for the Disaster Impact Analysis System,
supporting multi-phenomenon spatial analysis and visualization.
"""

from contextlib import asynccontextmanager
from typing import Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import health, phenomena
from src.api.storage import PhenomenonStorage


# Application state
app_state: Dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Initializes storage and other resources on startup,
    cleans up on shutdown.
    """
    # Startup
    app_state["storage"] = PhenomenonStorage()
    print("DIAS API started - Phenomenon storage initialized")
    
    yield
    
    # Shutdown
    app_state.clear()
    print("DIAS API shutdown - Resources cleaned up")


# Create FastAPI application
app = FastAPI(
    title="DIAS - Disaster Impact Analysis System",
    description=(
        "Multi-phenomenon spatial analysis and visualization API. "
        "Supports floods, contagion, supply-chain disruptions, and more."
    ),
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/api/openapi.json",
)


# Configure CORS
origins = [
    "http://localhost:3000",  # React dev server
    "http://localhost:8080",  # Alternative frontend
    "http://localhost:5173",  # Vite dev server
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8080",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Include routers
app.include_router(health.router, tags=["health"])
app.include_router(
    phenomena.router,
    prefix="/api/v1",
    tags=["phenomena"],
)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "DIAS API",
        "version": "2.0.0",
        "description": "Disaster Impact Analysis System",
        "documentation": {
            "swagger_ui": "/docs",
            "redoc": "/redoc",
            "openapi_spec": "/api/openapi.json",
        },
        "endpoints": {
            "health": "/health",
            "api_base": "/api/v1",
        },
    }


def get_storage() -> PhenomenonStorage:
    """Get the global phenomenon storage instance."""
    return app_state["storage"]

