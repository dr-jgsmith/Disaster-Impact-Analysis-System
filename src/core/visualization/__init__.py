"""
Visualization utilities for DIAS.

This package provides generic visualization tools that work with any
spatial phenomenon, enabling GeoJSON export, mapping, and analysis.
"""

from src.core.visualization.geojson import phenomenon_to_geojson

__all__ = ["phenomenon_to_geojson"]

