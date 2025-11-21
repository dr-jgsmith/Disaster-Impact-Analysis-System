"""
Core model building and simulation functions for DIAS.

This module provides JAX-based implementations of the disaster impact
analysis model, including connectivity, impact zones, and value simulation.
"""

from typing import Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components

from src.core import jax_ops


# ============================================================================
# Model Building Functions
# ============================================================================


def build_connectivity_matrix(
    coords: np.ndarray,
    use_geodesic: bool = False,
    proximity_threshold: Optional[float] = None,
) -> jnp.ndarray:
    """
    Build connectivity/adjacency matrix from parcel coordinates.
    
    Args:
        coords: Coordinate array of shape (n, 2) with [lat, lon] or [x, y]
        use_geodesic: If True, use Haversine distance; else Euclidean
        proximity_threshold: Distance threshold for connectivity (meters or units).
                           If None, computed automatically.
    
    Returns:
        Binary adjacency matrix of shape (n, n)
    
    Example:
        >>> coords = np.array([[29.76, -95.37], [29.77, -95.38]])
        >>> adj_matrix = build_connectivity_matrix(coords, use_geodesic=True)
    """
    coords_jax = jnp.array(coords)
    
    # Compute pairwise distances
    if use_geodesic:
        distances = jax_ops.haversine_distance_matrix(coords_jax, coords_jax)
    else:
        distances = jax_ops.euclidean_distance_pairwise(coords_jax, coords_jax)
    
    # Compute proximity threshold if not provided
    if proximity_threshold is None:
        # Use a heuristic: max distance divided by 7 (from legacy code)
        max_dist = float(jnp.max(distances))
        proximity_threshold = max_dist / 7.0
    
    # Build adjacency matrix
    adjacency = jax_ops.build_adjacency_matrix(distances, proximity_threshold)
    
    return adjacency


def compute_impact_zones(
    elevations: np.ndarray,
    adjacency_matrix: np.ndarray,
    min_water_level: float,
    max_water_level: float,
) -> List[np.ndarray]:
    """
    Compute impact zones for different water levels.
    
    For each water level from min to max, identifies connected components
    of flooded parcels (those below the water level).
    
    Args:
        elevations: Elevation array of shape (n,)
        adjacency_matrix: Adjacency matrix of shape (n, n)
        min_water_level: Minimum water level to consider
        max_water_level: Maximum water level to consider
    
    Returns:
        List of zone arrays, one per water level. Each zone array
        contains zone IDs (0 = not flooded, 1+ = zone ID).
    
    Example:
        >>> elevations = np.array([5.0, 10.0, 8.0, 6.0])
        >>> adjacency = np.ones((4, 4))
        >>> zones = compute_impact_zones(elevations, adjacency, 3, 12)
        >>> len(zones)  # One zone array per water level
        10
    """
    zones = []
    elevations_jax = jnp.array(elevations)
    adjacency_jax = jnp.array(adjacency_matrix)
    
    for water_level in range(int(min_water_level), int(max_water_level) + 1):
        # Compute connectivity at this water level
        connectivity = jax_ops.compute_connectivity(
            elevations_jax, adjacency_jax, float(water_level)
        )
        
        # Convert to numpy for connected components analysis
        connectivity_np = np.array(connectivity)
        
        # Find connected components (flood zones)
        n_components, labels = connected_components(
            csgraph=connectivity_np, directed=False, return_labels=True
        )
        
        # Create zone data: 0 for not flooded, zone ID for flooded
        zone_data = np.zeros(len(elevations))
        flooded_mask = elevations < water_level
        
        if flooded_mask.any():
            # Assign zone IDs to flooded parcels
            zone_data[flooded_mask] = labels[flooded_mask] + 1
        
        zones.append(zone_data)
    
    return zones


def binarize_zones(zones: List[np.ndarray]) -> List[jnp.ndarray]:
    """
    Convert zone arrays to binary (flooded/not flooded).
    
    Args:
        zones: List of zone arrays with zone IDs
    
    Returns:
        List of binary arrays (1 = flooded, 0 = not flooded)
    
    Example:
        >>> zones = [np.array([0, 1, 1, 0]), np.array([0, 2, 2, 1])]
        >>> binary_zones = binarize_zones(zones)
        >>> print(binary_zones[0])
        [0. 1. 1. 0.]
    """
    return [(jnp.array(zone) > 0.0).astype(jnp.float32) for zone in zones]


def compute_loss_multiplier(
    elevations: np.ndarray,
    binary_zone: jnp.ndarray,
    water_level: float,
    loss_percent: float,
) -> jnp.ndarray:
    """
    Compute loss multiplier based on elevation and flood zone.
    
    Loss is proportional to how far below the water level each parcel is.
    
    Args:
        elevations: Elevation array of shape (n,)
        binary_zone: Binary zone mask (1 = flooded, 0 = not)
        water_level: Water level for this scenario
        loss_percent: Base loss percentage (0-1)
    
    Returns:
        Loss multiplier array of shape (n,)
    
    Example:
        >>> elevations = np.array([5.0, 10.0, 8.0])
        >>> zone = jnp.array([1.0, 0.0, 1.0])
        >>> multiplier = compute_loss_multiplier(elevations, zone, 9.0, 0.8)
    """
    elevations_jax = jnp.array(elevations)
    
    # Mask elevations by flood zone
    masked_elevations = elevations_jax * binary_zone
    
    # Compute depth below water level
    depth_below_water = jnp.where(
        binary_zone > 0,
        water_level - elevations_jax,
        0.0
    )
    
    # Loss multiplier proportional to depth
    impact_multiplier = loss_percent * jnp.abs(depth_below_water)
    
    return impact_multiplier


def compute_impact_intensities(
    elevations: np.ndarray,
    binary_zones: List[jnp.ndarray],
    loss_percent: float,
    min_water_level: float,
) -> Tuple[List[jnp.ndarray], List[float]]:
    """
    Compute impact intensities for all water levels.
    
    Args:
        elevations: Elevation array
        binary_zones: List of binary zone masks
        loss_percent: Base loss percentage
        min_water_level: Starting water level
    
    Returns:
        Tuple of (impact_arrays, mean_impacts)
    
    Example:
        >>> elevations = np.array([5.0, 10.0, 8.0])
        >>> zones = [jnp.array([1.0, 0.0, 1.0])]
        >>> impacts, means = compute_impact_intensities(elevations, zones, 0.8, 3)
    """
    impact_arrays = []
    mean_impacts = []
    
    for i, binary_zone in enumerate(binary_zones):
        water_level = min_water_level + i
        
        impact_multiplier = compute_loss_multiplier(
            elevations, binary_zone, water_level, loss_percent
        )
        
        impact_arrays.append(impact_multiplier)
        mean_impacts.append(float(jnp.mean(impact_multiplier)))
    
    return impact_arrays, mean_impacts


# ============================================================================
# Complete Model Builder
# ============================================================================


class DisasterImpactModel:
    """
    Complete disaster impact analysis model.
    
    This class encapsulates the full model including connectivity,
    elevation data, impact zones, and simulation state.
    """
    
    def __init__(
        self,
        parcel_ids: List[str],
        coordinates: np.ndarray,
        elevations: np.ndarray,
        land_values: np.ndarray,
        building_values: np.ndarray,
        use_geodesic: bool = True,
        proximity_threshold: Optional[float] = None,
    ):
        """
        Initialize disaster impact model.
        
        Args:
            parcel_ids: List of parcel identifiers
            coordinates: Coordinate array (n, 2) with [lat, lon]
            elevations: Elevation array (n,)
            land_values: Land value array (n,)
            building_values: Building value array (n,)
            use_geodesic: Use geographic distances
            proximity_threshold: Distance threshold for adjacency
        """
        self.parcel_ids = parcel_ids
        self.coordinates = coordinates
        self.elevations = elevations
        self.land_values = land_values
        self.building_values = building_values
        self.use_geodesic = use_geodesic
        
        # Build connectivity matrix
        self.adjacency_matrix = build_connectivity_matrix(
            coordinates, use_geodesic, proximity_threshold
        )
        
        # Storage for computed zones and impacts
        self.zones: Optional[List[np.ndarray]] = None
        self.binary_zones: Optional[List[jnp.ndarray]] = None
        self.impact_intensities: Optional[List[jnp.ndarray]] = None
        self.mean_impacts: Optional[List[float]] = None
    
    def compute_zones(
        self, min_water_level: float, max_water_level: float
    ) -> None:
        """
        Compute impact zones for water level range.
        
        Args:
            min_water_level: Minimum water level
            max_water_level: Maximum water level
        """
        self.zones = compute_impact_zones(
            self.elevations,
            self.adjacency_matrix,
            min_water_level,
            max_water_level,
        )
        self.binary_zones = binarize_zones(self.zones)
    
    def compute_impacts(
        self, min_water_level: float, loss_percent: float
    ) -> None:
        """
        Compute impact intensities.
        
        Args:
            min_water_level: Starting water level
            loss_percent: Base loss percentage (0-1)
        """
        if self.binary_zones is None:
            raise ValueError("Must call compute_zones() first")
        
        self.impact_intensities, self.mean_impacts = compute_impact_intensities(
            self.elevations, self.binary_zones, loss_percent, min_water_level
        )
    
    def get_summary(self) -> Dict:
        """
        Get model summary statistics.
        
        Returns:
            Dictionary with model summary
        """
        return {
            "n_parcels": len(self.parcel_ids),
            "elevation_range": (
                float(self.elevations.min()),
                float(self.elevations.max()),
            ),
            "total_land_value": float(self.land_values.sum()),
            "total_building_value": float(self.building_values.sum()),
            "n_zones": len(self.zones) if self.zones else 0,
            "mean_connectivity": float(np.mean(self.adjacency_matrix)),
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Export model data to DataFrame.
        
        Returns:
            DataFrame with all model data
        """
        data = {
            "PARCELID": self.parcel_ids,
            "LAT": self.coordinates[:, 0],
            "LON": self.coordinates[:, 1],
            "ELEVATION": self.elevations,
            "LANDVALUE": self.land_values,
            "BLDGVALUE": self.building_values,
        }
        
        # Add zone data if computed
        if self.zones:
            for i, zone in enumerate(self.zones):
                data[f"Impact_Zone_{i}"] = zone
        
        # Add impact intensities if computed
        if self.impact_intensities:
            for i, impact in enumerate(self.impact_intensities):
                data[f"Impact_Intensity_{i}"] = np.array(impact)
        
        return pd.DataFrame(data)


def build_model_from_data(
    data: pd.DataFrame,
    lat_field: str = "LAT",
    lon_field: str = "LON",
    parcel_field: str = "PARCELID",
    elevation_field: str = "ELEVATION",
    land_value_field: str = "LANDVALUE",
    building_value_field: str = "BLDGVALUE",
    use_geodesic: bool = True,
) -> DisasterImpactModel:
    """
    Build model from DataFrame.
    
    Args:
        data: DataFrame with parcel data
        lat_field: Latitude column name
        lon_field: Longitude column name
        parcel_field: Parcel ID column name
        elevation_field: Elevation column name
        land_value_field: Land value column name
        building_value_field: Building value column name
        use_geodesic: Use geographic distances
    
    Returns:
        Initialized DisasterImpactModel
    
    Example:
        >>> import pandas as pd
        >>> data = pd.read_csv("parcels.csv")
        >>> model = build_model_from_data(data)
        >>> model.compute_zones(3, 14)
        >>> model.compute_impacts(3, 0.8)
        >>> summary = model.get_summary()
    """
    # Extract data
    parcel_ids = data[parcel_field].tolist()
    coordinates = data[[lat_field, lon_field]].values
    elevations = data[elevation_field].values
    land_values = data[land_value_field].values
    building_values = data[building_value_field].values
    
    # Build model
    model = DisasterImpactModel(
        parcel_ids=parcel_ids,
        coordinates=coordinates,
        elevations=elevations,
        land_values=land_values,
        building_values=building_values,
        use_geodesic=use_geodesic,
    )
    
    return model

