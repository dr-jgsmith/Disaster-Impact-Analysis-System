"""
Visualization utilities for DIAS.

This package provides generic visualization tools that work with any
spatial sp_event, enabling GeoJSON export, mapping, and analysis.
"""

from src.core.visualization.geojson import sp_event_to_geojson

__all__ = ["sp_event_to_geojson"]

