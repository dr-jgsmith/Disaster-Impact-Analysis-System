"""
Flood disaster phenomenon implementation.

This module provides a flood-specific implementation of the SpatialPhenomenon
base class, enabling flood impact analysis and visualization.
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
import jax.numpy as jnp
from scipy.sparse.csgraph import connected_components

from src.core.base.phenomenon import SpatialPhenomenon
from src.core import jax_ops


class FloodPhenomenon(SpatialPhenomenon):
    """
    Flood disaster analysis phenomenon.
    
    Models flood impacts based on elevation, connectivity, and water levels.
    Computes flooded zones, property value loss, and impact metrics.
    
    Example:
        >>> flood = FloodPhenomenon(
        ...     parcel_ids=["P001", "P002", "P003"],
        ...     coordinates=np.array([[29.76, -95.37], ...]),
        ...     adjacency_matrix=adjacency,
        ...     elevations=np.array([5.0, 10.0, 8.0]),
        ...     land_values=np.array([100000, 150000, 120000]),
        ...     building_values=np.array([200000, 250000, 220000]),
        ... )
        >>> zones = flood.compute_zones({
        ...     "min_water_level": 3.0,
        ...     "max_water_level": 14.0
        ... })
        >>> impact = flood.compute_impact(zones, {"loss_percent": 0.8})
    """
    
    def __init__(
        self,
        parcel_ids: List[str],
        coordinates: np.ndarray,
        adjacency_matrix: np.ndarray,
        elevations: np.ndarray,
        land_values: np.ndarray,
        building_values: np.ndarray,
    ):
        """
        Initialize flood phenomenon.
        
        Args:
            parcel_ids: List of parcel identifiers
            coordinates: Coordinate array (n, 2) with [lat, lon]
            adjacency_matrix: Spatial connectivity matrix (n, n)
            elevations: Elevation array (n,)
            land_values: Land value array (n,)
            building_values: Building value array (n,)
        """
        # Package flood-specific attributes
        attributes = {
            "elevations": elevations,
            "land_values": land_values,
            "building_values": building_values,
        }
        
        # Initialize base class
        super().__init__(parcel_ids, coordinates, adjacency_matrix, attributes)
        
        # Convenient access to flood attributes
        self.elevations = elevations
        self.land_values = land_values
        self.building_values = building_values
        
        # Storage for computed binary zones and impacts
        self.binary_zones: Optional[List[jnp.ndarray]] = None
        self.impact_intensities: Optional[List[jnp.ndarray]] = None
        self.mean_impacts: Optional[List[float]] = None
        self.scenario_params: Optional[Dict[str, Any]] = None
    
    def compute_zones(self, scenario_params: Dict[str, Any]) -> List[np.ndarray]:
        """
        Compute flood impact zones for water level range.
        
        For each water level from min to max, identifies connected components
        of flooded parcels (those below the water level).
        
        Args:
            scenario_params: Must contain:
                - min_water_level: Minimum water level (float)
                - max_water_level: Maximum water level (float)
        
        Returns:
            List of zone arrays, one per water level. Each zone array
            contains zone IDs (0 = not flooded, 1+ = zone ID).
        
        Example:
            >>> zones = flood.compute_zones({
            ...     "min_water_level": 3.0,
            ...     "max_water_level": 14.0
            ... })
            >>> len(zones)  # One zone array per water level
            12
        """
        # Extract parameters
        min_water_level = scenario_params["min_water_level"]
        max_water_level = scenario_params["max_water_level"]
        
        # Store for later use in compute_impact
        self.scenario_params = scenario_params
        
        zones = []
        elevations_jax = jnp.array(self.elevations)
        adjacency_jax = jnp.array(self.adjacency_matrix)
        
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
            zone_data = np.zeros(len(self.elevations))
            flooded_mask = self.elevations < water_level
            
            if flooded_mask.any():
                # Assign zone IDs to flooded parcels
                zone_data[flooded_mask] = labels[flooded_mask] + 1
            
            zones.append(zone_data)
        
        # Store zones
        self.zones = zones
        
        # Compute binary zones
        self.binary_zones = self._binarize_zones(zones)
        
        return zones
    
    def compute_impact(
        self,
        zones: List[np.ndarray],
        scenario_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate property value loss from flooding.
        
        Computes loss multipliers based on how far below water level
        each parcel is, then calculates total property value loss.
        
        Args:
            zones: Zone arrays from compute_zones()
            scenario_params: Must contain:
                - loss_percent: Base loss percentage, 0-1 (float)
                - min_water_level: Starting water level (float)
        
        Returns:
            Dictionary with:
                - impact_intensities: List of impact multiplier arrays
                - mean_impacts: List of mean impacts per scenario
                - total_property_loss: List of total $ loss per scenario
                - affected_parcels: List of # affected parcels per scenario
        
        Example:
            >>> impact = flood.compute_impact(zones, {
            ...     "loss_percent": 0.8,
            ...     "min_water_level": 3.0
            ... })
            >>> impact["total_property_loss"][0]
            1250000.0
        """
        # Extract parameters
        loss_percent = scenario_params["loss_percent"]
        min_water_level = scenario_params.get(
            "min_water_level",
            self.scenario_params.get("min_water_level") if self.scenario_params else 3.0
        )
        
        # Compute binary zones if not already done
        if self.binary_zones is None:
            self.binary_zones = self._binarize_zones(zones)
        
        impact_arrays = []
        mean_impacts = []
        total_losses = []
        affected_counts = []
        
        for i, binary_zone in enumerate(self.binary_zones):
            water_level = min_water_level + i
            
            # Compute loss multiplier
            impact_multiplier = self._compute_loss_multiplier(
                binary_zone, water_level, loss_percent
            )
            
            impact_arrays.append(impact_multiplier)
            mean_impacts.append(float(jnp.mean(impact_multiplier)))
            
            # Calculate total property value loss
            total_property_values = self.land_values + self.building_values
            property_losses = np.array(impact_multiplier) * total_property_values
            total_loss = float(property_losses.sum())
            total_losses.append(total_loss)
            
            # Count affected parcels
            affected = int(np.sum(binary_zone > 0))
            affected_counts.append(affected)
        
        # Store impact data
        self.impact_intensities = impact_arrays
        self.mean_impacts = mean_impacts
        
        # Package impact metrics
        self.impact_metrics = {
            "impact_intensities": impact_arrays,
            "mean_impacts": mean_impacts,
            "total_property_loss": total_losses,
            "affected_parcels": affected_counts,
            "total_land_value": float(self.land_values.sum()),
            "total_building_value": float(self.building_values.sum()),
            "n_scenarios": len(zones),
        }
        
        return self.impact_metrics
    
    def get_phenomenon_type(self) -> str:
        """Return phenomenon type identifier."""
        return "flood"
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Export flood data to DataFrame.
        
        Returns:
            DataFrame with parcel data, elevations, values, zones, and impacts
        """
        data = {
            "PARCELID": self.entity_ids,
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
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get flood model summary statistics.
        
        Returns:
            Dictionary with flood-specific summary
        """
        summary = super().get_summary()
        
        # Add flood-specific metrics
        summary.update({
            "total_land_value": float(self.land_values.sum()),
            "total_building_value": float(self.building_values.sum()),
            "total_property_value": float(
                self.land_values.sum() + self.building_values.sum()
            ),
            "mean_connectivity": float(np.mean(self.adjacency_matrix)),
        })
        
        return summary
    
    # ========================================================================
    # Private Helper Methods
    # ========================================================================
    
    def _binarize_zones(self, zones: List[np.ndarray]) -> List[jnp.ndarray]:
        """
        Convert zone arrays to binary (flooded/not flooded).
        
        Args:
            zones: List of zone arrays with zone IDs
        
        Returns:
            List of binary arrays (1 = flooded, 0 = not flooded)
        """
        return [(jnp.array(zone) > 0.0).astype(jnp.float32) for zone in zones]
    
    def _compute_loss_multiplier(
        self,
        binary_zone: jnp.ndarray,
        water_level: float,
        loss_percent: float,
    ) -> jnp.ndarray:
        """
        Compute loss multiplier based on elevation and flood zone.
        
        Loss is proportional to how far below the water level each parcel is.
        
        Args:
            binary_zone: Binary zone mask (1 = flooded, 0 = not)
            water_level: Water level for this scenario
            loss_percent: Base loss percentage (0-1)
        
        Returns:
            Loss multiplier array
        """
        elevations_jax = jnp.array(self.elevations)
        
        # Compute depth below water level
        depth_below_water = jnp.where(
            binary_zone > 0,
            water_level - elevations_jax,
            0.0
        )
        
        # Loss multiplier proportional to depth
        impact_multiplier = loss_percent * jnp.abs(depth_below_water)
        
        return impact_multiplier


# ============================================================================
# Factory Functions
# ============================================================================


def build_flood_model_from_data(
    data: pd.DataFrame,
    lat_field: str = "LAT",
    lon_field: str = "LON",
    parcel_field: str = "PARCELID",
    elevation_field: str = "ELEVATION",
    land_value_field: str = "LANDVALUE",
    building_value_field: str = "BLDGVALUE",
    use_geodesic: bool = True,
    proximity_threshold: Optional[float] = None,
) -> FloodPhenomenon:
    """
    Build flood model from DataFrame.
    
    Convenience function to create a FloodPhenomenon from tabular data.
    
    Args:
        data: DataFrame with parcel data
        lat_field: Latitude column name
        lon_field: Longitude column name
        parcel_field: Parcel ID column name
        elevation_field: Elevation column name
        land_value_field: Land value column name
        building_value_field: Building value column name
        use_geodesic: Use geographic distances (True) or Euclidean (False)
        proximity_threshold: Distance threshold for adjacency (auto if None)
    
    Returns:
        Initialized FloodPhenomenon
    
    Example:
        >>> import pandas as pd
        >>> data = pd.read_csv("parcels.csv")
        >>> flood = build_flood_model_from_data(data)
        >>> zones = flood.compute_zones({
        ...     "min_water_level": 3.0,
        ...     "max_water_level": 14.0
        ... })
        >>> impact = flood.compute_impact(zones, {"loss_percent": 0.8})
    """
    # Import here to avoid circular dependency
    from src.core.model import build_connectivity_matrix
    
    # Extract data
    parcel_ids = data[parcel_field].tolist()
    coordinates = data[[lat_field, lon_field]].values
    elevations = data[elevation_field].values
    land_values = data[land_value_field].values
    building_values = data[building_value_field].values
    
    # Build connectivity matrix
    adjacency_matrix = build_connectivity_matrix(
        coordinates, use_geodesic, proximity_threshold
    )
    
    # Create flood phenomenon
    flood = FloodPhenomenon(
        parcel_ids=parcel_ids,
        coordinates=coordinates,
        adjacency_matrix=np.array(adjacency_matrix),
        elevations=elevations,
        land_values=land_values,
        building_values=building_values,
    )
    
    return flood

