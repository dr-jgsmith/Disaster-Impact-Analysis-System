"""
DIAS Core Module - Multi-event spatial analysis.

This module provides the core functionality for analyzing spatial events
including abstract base classes, concrete implementations, and utilities.
"""

# Base abstractions
from src.core.base.sp_event import SpatialEvent

# Concrete event implementations
from src.core.sp_events.flood_event import FloodEvent, build_flood_event_from_data

# Visualization utilities
from src.core.visualization.geojson import (
    sp_event_to_geojson,
    sp_event_to_geojson_with_impacts,
)

# JAX operations and legacy model (for backward compatibility)
from src.core import jax_ops
from src.core import model

__all__ = [
    # Base abstractions
    "SpatialEvent",
    # Flood event
    "FloodEvent",
    "build_flood_event_from_data",
    # Visualization
    "sp_event_to_geojson",
    "sp_event_to_geojson_with_impacts",
    # Utilities
    "jax_ops",
    "model",
]
