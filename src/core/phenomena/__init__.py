"""
Spatial phenomenon implementations for DIAS.

This package contains concrete implementations of different spatial
phenomena that can be analyzed and visualized.
"""

from src.core.phenomena.flood import FloodPhenomenon, build_flood_model_from_data

__all__ = ["FloodPhenomenon", "build_flood_model_from_data"]

