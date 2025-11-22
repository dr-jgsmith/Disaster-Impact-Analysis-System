"""
DIAS Core Module - Multi-phenomenon spatial analysis.

This module provides the core functionality for analyzing spatial phenomena
including abstract base classes, concrete implementations, and utilities.
"""

# Base abstractions
from src.core.base.phenomenon import SpatialPhenomenon

# Concrete phenomena implementations
from src.core.phenomena.flood import FloodPhenomenon, build_flood_model_from_data

# Visualization utilities
from src.core.visualization.geojson import (
    phenomenon_to_geojson,
    phenomenon_to_geojson_with_impacts,
)

# JAX operations and legacy model (for backward compatibility)
from src.core import jax_ops
from src.core import model

__all__ = [
    # Base abstractions
    "SpatialPhenomenon",
    # Flood phenomenon
    "FloodPhenomenon",
    "build_flood_model_from_data",
    # Visualization
    "phenomenon_to_geojson",
    "phenomenon_to_geojson_with_impacts",
    # Utilities
    "jax_ops",
    "model",
]
