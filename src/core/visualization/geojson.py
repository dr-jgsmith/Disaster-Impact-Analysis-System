"""
Generic GeoJSON conversion for spatial phenomena.

This module provides phenomenon-agnostic GeoJSON conversion, enabling
visualization of any spatial phenomenon (floods, contagion, supply-chain, etc.)
using standard GIS tools and web mapping libraries like Leaflet.js.
"""

from typing import Dict, Any, Optional, List
import numpy as np

from src.core.base.phenomenon import SpatialPhenomenon


def phenomenon_to_geojson(
    phenomenon: SpatialPhenomenon,
    include_zones: bool = True,
    zone_index: Optional[int] = None,
    include_attributes: bool = True,
) -> Dict[str, Any]:
    """
    Convert any spatial phenomenon to GeoJSON FeatureCollection.
    
    This function works with ANY phenomenon type (flood, contagion, supply-chain)
    and produces standard GeoJSON that can be visualized in Leaflet, QGIS, etc.
    
    Args:
        phenomenon: Any SpatialPhenomenon instance
        include_zones: Include zone data in properties
        zone_index: Specific zone index to include (None = all zones)
        include_attributes: Include phenomenon attributes in properties
    
    Returns:
        GeoJSON FeatureCollection with points for each entity
    
    Example:
        >>> from src.core.phenomena.flood import build_flood_model_from_data
        >>> flood = build_flood_model_from_data(data)
        >>> flood.compute_zones({"min_water_level": 3, "max_water_level": 14})
        >>> geojson = phenomenon_to_geojson(flood)
        >>> # Save to file
        >>> import json
        >>> with open("flood_map.geojson", "w") as f:
        ...     json.dump(geojson, f)
    """
    features = []
    
    for i, entity_id in enumerate(phenomenon.entity_ids):
        # Create point geometry
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [
                    float(phenomenon.coordinates[i, 1]),  # lon (x)
                    float(phenomenon.coordinates[i, 0]),  # lat (y)
                ]
            },
            "properties": {
                "id": entity_id,
                "phenomenon_type": phenomenon.get_phenomenon_type(),
                "entity_index": i,
            }
        }
        
        # Add phenomenon attributes
        if include_attributes:
            for attr_name, attr_values in phenomenon.attributes.items():
                feature["properties"][attr_name] = float(attr_values[i])
        
        # Add zone data if requested
        if include_zones and phenomenon.zones:
            if zone_index is not None:
                # Single zone
                if zone_index < len(phenomenon.zones):
                    feature["properties"]["zone"] = int(phenomenon.zones[zone_index][i])
            else:
                # All zones
                for z_idx, zone in enumerate(phenomenon.zones):
                    feature["properties"][f"zone_{z_idx}"] = int(zone[i])
        
        features.append(feature)
    
    # Create FeatureCollection
    geojson = {
        "type": "FeatureCollection",
        "features": features,
        "metadata": phenomenon.to_dict(),
    }
    
    return geojson


def phenomenon_to_geojson_with_impacts(
    phenomenon: SpatialPhenomenon,
    zone_index: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Convert phenomenon to GeoJSON including impact metrics.
    
    Similar to phenomenon_to_geojson but includes impact data if available.
    
    Args:
        phenomenon: Any SpatialPhenomenon instance with computed impacts
        zone_index: Specific zone/scenario index to include
    
    Returns:
        GeoJSON FeatureCollection with impact data
    
    Example:
        >>> flood.compute_impact(flood.zones, {"loss_percent": 0.8})
        >>> geojson = phenomenon_to_geojson_with_impacts(flood, zone_index=0)
    """
    # Get base GeoJSON
    geojson = phenomenon_to_geojson(
        phenomenon,
        include_zones=True,
        zone_index=zone_index,
        include_attributes=True,
    )
    
    # Add impact metrics if available
    if phenomenon.impact_metrics:
        geojson["metadata"]["impact_metrics"] = phenomenon.impact_metrics
        
        # Add per-entity impact data if available
        if "impact_intensities" in phenomenon.impact_metrics:
            impact_intensities = phenomenon.impact_metrics["impact_intensities"]
            
            for i, feature in enumerate(geojson["features"]):
                if zone_index is not None and zone_index < len(impact_intensities):
                    # Single scenario impact
                    feature["properties"]["impact_intensity"] = float(
                        impact_intensities[zone_index][i]
                    )
                else:
                    # All scenario impacts
                    for j, impact_array in enumerate(impact_intensities):
                        feature["properties"][f"impact_{j}"] = float(impact_array[i])
    
    return geojson


def get_zone_bounds(
    phenomenon: SpatialPhenomenon,
    zone_index: int,
) -> Optional[Dict[str, float]]:
    """
    Get bounding box for a specific zone.
    
    Args:
        phenomenon: Phenomenon with computed zones
        zone_index: Zone index to get bounds for
    
    Returns:
        Bounding box dict with min_lat, max_lat, min_lon, max_lon,
        or None if zone not found
    
    Example:
        >>> bounds = get_zone_bounds(flood, zone_index=0)
        >>> bounds["min_lat"]
        29.76
    """
    if not phenomenon.zones or zone_index >= len(phenomenon.zones):
        return None
    
    zone = phenomenon.zones[zone_index]
    affected_mask = zone > 0
    
    if not np.any(affected_mask):
        return None
    
    affected_coords = phenomenon.coordinates[affected_mask]
    
    return {
        "min_lat": float(affected_coords[:, 0].min()),
        "max_lat": float(affected_coords[:, 0].max()),
        "min_lon": float(affected_coords[:, 1].min()),
        "max_lon": float(affected_coords[:, 1].max()),
    }


def get_zone_statistics(
    phenomenon: SpatialPhenomenon,
    zone_index: int,
) -> Optional[Dict[str, Any]]:
    """
    Get statistics for a specific zone.
    
    Args:
        phenomenon: Phenomenon with computed zones
        zone_index: Zone index to get statistics for
    
    Returns:
        Dictionary with zone statistics or None if zone not found
    
    Example:
        >>> stats = get_zone_statistics(flood, zone_index=0)
        >>> stats["n_entities"]
        25
    """
    if not phenomenon.zones or zone_index >= len(phenomenon.zones):
        return None
    
    zone = phenomenon.zones[zone_index]
    affected_mask = zone > 0
    n_affected = int(np.sum(affected_mask))
    
    if n_affected == 0:
        return None
    
    stats = {
        "zone_index": zone_index,
        "n_entities": n_affected,
        "percent_affected": float(n_affected / len(zone) * 100),
        "unique_zones": int(np.unique(zone[affected_mask]).size),
    }
    
    # Add attribute statistics for affected entities
    for attr_name, attr_values in phenomenon.attributes.items():
        affected_values = attr_values[affected_mask]
        stats[f"{attr_name}_affected"] = {
            "min": float(affected_values.min()),
            "max": float(affected_values.max()),
            "mean": float(affected_values.mean()),
            "sum": float(affected_values.sum()),
        }
    
    return stats


def export_all_scenarios(
    phenomenon: SpatialPhenomenon,
    output_dir: str,
    prefix: str = "scenario",
) -> List[str]:
    """
    Export all scenario zones as separate GeoJSON files.
    
    Args:
        phenomenon: Phenomenon with computed zones
        output_dir: Directory to save GeoJSON files
        prefix: Filename prefix for scenarios
    
    Returns:
        List of created file paths
    
    Example:
        >>> files = export_all_scenarios(flood, "./output", "flood")
        >>> print(files)
        ['./output/flood_0.geojson', './output/flood_1.geojson', ...]
    """
    import json
    import os
    
    if not phenomenon.zones:
        return []
    
    os.makedirs(output_dir, exist_ok=True)
    created_files = []
    
    for i in range(len(phenomenon.zones)):
        # Generate GeoJSON for this scenario
        geojson = phenomenon_to_geojson_with_impacts(phenomenon, zone_index=i)
        
        # Create filename
        filename = os.path.join(output_dir, f"{prefix}_{i}.geojson")
        
        # Write to file
        with open(filename, "w") as f:
            json.dump(geojson, f, indent=2)
        
        created_files.append(filename)
    
    return created_files

