"""
Spatial event implementations for DIAS.

This package contains concrete implementations of different spatial
events that can be analyzed and visualized.
"""

from src.core.sp_events.flood_event import FloodEvent, build_flood_event_from_data

__all__ = ["FloodEvent", "build_flood_event_from_data"]
