"""
Abstract base class for spatial phenomena.

This module defines the interface that all spatial phenomena must implement,
enabling generic visualization, analysis, and API endpoints.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd


class SpatialPhenomenon(ABC):
    """
    Abstract base class for spatial phenomena.
    
    This class defines the interface for any spatial phenomenon that
    propagates through a network (floods, contagion, supply-chain
    disruptions, social movements, etc.).
    
    Subclasses must implement:
    - compute_zones(): Identify affected zones/clusters
    - compute_impact(): Calculate phenomenon-specific impacts
    - get_phenomenon_type(): Return type identifier
    
    Example:
        >>> class FloodPhenomenon(SpatialPhenomenon):
        ...     def compute_zones(self, params):
        ...         # Flood-specific zone computation
        ...         pass
        ...     def compute_impact(self, zones, params):
        ...         # Property value loss calculation
        ...         pass
        ...     def get_phenomenon_type(self):
        ...         return "flood"
    """
    
    def __init__(
        self,
        entity_ids: List[str],
        coordinates: np.ndarray,
        adjacency_matrix: np.ndarray,
        entity_attributes: Dict[str, np.ndarray],
    ):
        """
        Initialize spatial phenomenon.
        
        Args:
            entity_ids: Unique identifiers for spatial entities
                       (e.g., parcel IDs, person IDs, facility IDs)
            coordinates: Coordinate array of shape (n, 2) with [lat, lon]
            adjacency_matrix: Spatial connectivity matrix of shape (n, n)
            entity_attributes: Phenomenon-specific attributes as dict of arrays
                              (e.g., {"elevations": [...], "values": [...]})
        
        Example:
            >>> entity_ids = ["P001", "P002", "P003"]
            >>> coords = np.array([[29.76, -95.37], [29.77, -95.38], ...])
            >>> adjacency = np.array([[1, 1, 0], [1, 1, 1], [0, 1, 1]])
            >>> attributes = {"elevations": np.array([5.0, 10.0, 8.0])}
            >>> phenomenon = ConcretePhenomenon(
            ...     entity_ids, coords, adjacency, attributes
            ... )
        """
        self.entity_ids = entity_ids
        self.coordinates = coordinates
        self.adjacency_matrix = adjacency_matrix
        self.attributes = entity_attributes
        
        # Storage for computed state (set by subclass methods)
        self.zones: Optional[List[np.ndarray]] = None
        self.impact_metrics: Optional[Dict[str, Any]] = None
    
    @abstractmethod
    def compute_zones(self, scenario_params: Dict[str, Any]) -> List[np.ndarray]:
        """
        Compute affected zones based on phenomenon-specific rules.
        
        For floods: Connected areas below water level
        For contagion: Connected infected populations over time
        For supply-chain: Connected disrupted facilities
        
        Args:
            scenario_params: Phenomenon-specific parameters
                           For flood: {"min_water_level": 3, "max_water_level": 14}
                           For contagion: {"transmission_rate": 0.3, "time_steps": 30}
        
        Returns:
            List of zone arrays, one per scenario/time step.
            Each array contains zone IDs (0 = unaffected, 1+ = zone ID).
        
        Example:
            >>> zones = phenomenon.compute_zones({"param1": value1})
            >>> len(zones)  # Number of scenarios
            10
            >>> zones[0].shape  # Shape matches number of entities
            (100,)
        """
        pass
    
    @abstractmethod
    def compute_impact(
        self, 
        zones: List[np.ndarray],
        scenario_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate impact based on phenomenon-specific metrics.
        
        For floods: Property value loss
        For contagion: Infection rates, health outcomes, economic cost
        For supply-chain: Economic loss, delay costs
        
        Args:
            zones: Zone arrays from compute_zones()
            scenario_params: Phenomenon-specific impact parameters
                           For flood: {"loss_percent": 0.8}
                           For contagion: {"cost_per_case": 5000}
        
        Returns:
            Dictionary with impact metrics (phenomenon-specific keys)
        
        Example:
            >>> impact = phenomenon.compute_impact(zones, {"param": value})
            >>> impact["total_loss"]
            1250000.0
        """
        pass
    
    @abstractmethod
    def get_phenomenon_type(self) -> str:
        """
        Return phenomenon type identifier.
        
        Returns:
            String identifier (e.g., "flood", "contagion", "supply_chain")
        
        Example:
            >>> phenomenon.get_phenomenon_type()
            'flood'
        """
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Export phenomenon state to dictionary.
        
        Returns:
            Dictionary with phenomenon metadata
        
        Example:
            >>> data = phenomenon.to_dict()
            >>> data["phenomenon_type"]
            'flood'
            >>> data["n_entities"]
            100
        """
        return {
            "phenomenon_type": self.get_phenomenon_type(),
            "n_entities": len(self.entity_ids),
            "has_zones": self.zones is not None,
            "has_impacts": self.impact_metrics is not None,
            "coordinate_bounds": self._get_coordinate_bounds(),
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Export phenomenon data to DataFrame.
        
        Returns:
            DataFrame with entity data, attributes, zones, and impacts
        
        Example:
            >>> df = phenomenon.to_dataframe()
            >>> df.columns
            ['entity_id', 'lat', 'lon', 'attr1', 'zone_0', ...]
        """
        data = {
            "entity_id": self.entity_ids,
            "lat": self.coordinates[:, 0],
            "lon": self.coordinates[:, 1],
        }
        
        # Add all phenomenon attributes
        for attr_name, attr_values in self.attributes.items():
            data[attr_name] = attr_values
        
        # Add zone data if computed
        if self.zones:
            for i, zone in enumerate(self.zones):
                data[f"zone_{i}"] = zone
        
        return pd.DataFrame(data)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics for phenomenon.
        
        Returns:
            Dictionary with summary statistics
        
        Example:
            >>> summary = phenomenon.get_summary()
            >>> summary["n_entities"]
            100
        """
        summary = {
            "phenomenon_type": self.get_phenomenon_type(),
            "n_entities": len(self.entity_ids),
            "n_zones": len(self.zones) if self.zones else 0,
            "coordinate_bounds": self._get_coordinate_bounds(),
        }
        
        # Add attribute statistics
        for attr_name, attr_values in self.attributes.items():
            summary[f"{attr_name}_range"] = (
                float(attr_values.min()),
                float(attr_values.max())
            )
            summary[f"{attr_name}_mean"] = float(attr_values.mean())
        
        # Add impact metrics if available
        if self.impact_metrics:
            summary["impact_metrics"] = self.impact_metrics
        
        return summary
    
    def _get_coordinate_bounds(self) -> Dict[str, float]:
        """Get bounding box of coordinates."""
        return {
            "min_lat": float(self.coordinates[:, 0].min()),
            "max_lat": float(self.coordinates[:, 0].max()),
            "min_lon": float(self.coordinates[:, 1].min()),
            "max_lon": float(self.coordinates[:, 1].max()),
        }
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"{self.__class__.__name__}("
            f"type={self.get_phenomenon_type()}, "
            f"entities={len(self.entity_ids)}, "
            f"zones={len(self.zones) if self.zones else 0})"
        )

