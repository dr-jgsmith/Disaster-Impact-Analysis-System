# Multi-Phenomenon Architecture Refactoring Plan

## Current Status ✅

**Good News:** Foundation is solid!
- ✅ `src/core/jax_ops.py` EXISTS (15.5KB, fully generic)
- ✅ `src/core/model.py` EXISTS (13.2KB, flood-specific)
- ✅ All JAX utilities are phenomenon-agnostic
- ✅ Connectivity operations are reusable

**What Needs Work:** Abstraction layer for multiple phenomena

---

## Simplified Refactoring Plan

### PHASE 1: Create Abstract Base (~3 hours)
**Branch:** `feature/refactor-multi-phenomenon`

#### Task 1.1: Create Abstract Base Class (1 hour)
**File:** `src/core/base/__init__.py`
**File:** `src/core/base/phenomenon.py`

```python
"""Abstract base for spatial phenomena."""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import numpy as np

class SpatialPhenomenon(ABC):
    """Base class for any spatial phenomenon (floods, contagion, supply-chain, etc.)."""
    
    def __init__(
        self,
        entity_ids: List[str],
        coordinates: np.ndarray,
        adjacency_matrix: np.ndarray,
        entity_attributes: Dict[str, np.ndarray],
    ):
        self.entity_ids = entity_ids
        self.coordinates = coordinates
        self.adjacency_matrix = adjacency_matrix
        self.attributes = entity_attributes
        self.zones: Optional[List[np.ndarray]] = None
        self.impact_metrics: Optional[Dict[str, Any]] = None
    
    @abstractmethod
    def compute_zones(self, scenario_params: Dict[str, Any]) -> List[np.ndarray]:
        """Compute zones based on phenomenon-specific rules."""
        pass
    
    @abstractmethod
    def compute_impact(self, zones: List[np.ndarray], 
                      scenario_params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate impact with phenomenon-specific metrics."""
        pass
    
    @abstractmethod
    def get_phenomenon_type(self) -> str:
        """Return phenomenon type identifier."""
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """Export phenomenon state."""
        return {
            "phenomenon_type": self.get_phenomenon_type(),
            "n_entities": len(self.entity_ids),
            "has_zones": self.zones is not None,
            "has_impacts": self.impact_metrics is not None,
        }
```

#### Task 1.2: Add Base Tests (30 min)
**File:** `tests/unit/test_base_phenomenon.py`

```python
"""Tests for abstract phenomenon base class."""
import pytest
import numpy as np
from src.core.base.phenomenon import SpatialPhenomenon

def test_cannot_instantiate_abstract_class():
    """Test that abstract class cannot be instantiated."""
    with pytest.raises(TypeError):
        SpatialPhenomenon([], np.array([]), np.array([]), {})

def test_subclass_must_implement_methods():
    """Test that subclass must implement abstract methods."""
    class IncompletePhenomenon(SpatialPhenomenon):
        pass
    
    with pytest.raises(TypeError):
        IncompletePhenomenon([], np.array([]), np.array([]), {})
```

#### Task 1.3: Update Core Init (30 min)
**File:** `src/core/__init__.py`

```python
"""DIAS Core Module - Multi-phenomenon spatial analysis."""

from src.core.base.phenomenon import SpatialPhenomenon

__all__ = ["SpatialPhenomenon", "jax_ops"]
```

---

### PHASE 2: Refactor Flood as Phenomenon (~2.5 hours)

#### Task 2.1: Create Phenomena Package (15 min)
```bash
mkdir -p src/core/phenomena
touch src/core/phenomena/__init__.py
```

#### Task 2.2: Implement FloodPhenomenon (1.5 hours)
**File:** `src/core/phenomena/flood.py`

Migrate existing `DisasterImpactModel` to:

```python
"""Flood disaster phenomenon implementation."""

from typing import Dict, List, Any
import numpy as np
import jax.numpy as jnp
from scipy.sparse.csgraph import connected_components

from src.core.base.phenomenon import SpatialPhenomenon
from src.core import jax_ops

class FloodPhenomenon(SpatialPhenomenon):
    """Flood disaster analysis."""
    
    def __init__(
        self,
        parcel_ids: List[str],
        coordinates: np.ndarray,
        adjacency_matrix: np.ndarray,
        elevations: np.ndarray,
        land_values: np.ndarray,
        building_values: np.ndarray,
    ):
        attributes = {
            "elevations": elevations,
            "land_values": land_values,
            "building_values": building_values,
        }
        super().__init__(parcel_ids, coordinates, adjacency_matrix, attributes)
    
    def compute_zones(self, scenario_params: Dict[str, Any]) -> List[np.ndarray]:
        """
        Compute flood zones.
        
        scenario_params = {
            "min_water_level": 3.0,
            "max_water_level": 14.0,
        }
        """
        # Move logic from model.py compute_impact_zones()
        pass
    
    def compute_impact(self, zones: List[np.ndarray], 
                      scenario_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute property value loss.
        
        scenario_params = {
            "loss_percent": 0.8,
        }
        """
        # Move logic from model.py compute_impact_intensities()
        pass
    
    def get_phenomenon_type(self) -> str:
        return "flood"
```

#### Task 2.3: Add Helper Factory Function (30 min)
**File:** `src/core/phenomena/flood.py`

```python
def build_flood_model_from_data(
    data: pd.DataFrame,
    lat_field: str = "LAT",
    lon_field: str = "LON",
    parcel_field: str = "PARCELID",
    elevation_field: str = "ELEVATION",
    land_value_field: str = "LANDVALUE",
    building_value_field: str = "BLDGVALUE",
    use_geodesic: bool = True,
) -> FloodPhenomenon:
    """Build flood model from DataFrame (convenience function)."""
    # Extract data
    parcel_ids = data[parcel_field].tolist()
    coordinates = data[[lat_field, lon_field]].values
    elevations = data[elevation_field].values
    land_values = data[land_value_field].values
    building_values = data[building_value_field].values
    
    # Build connectivity
    from src.core.model import build_connectivity_matrix
    adjacency_matrix = build_connectivity_matrix(coordinates, use_geodesic)
    
    # Create phenomenon
    return FloodPhenomenon(
        parcel_ids, coordinates, adjacency_matrix,
        elevations, land_values, building_values
    )
```

#### Task 2.4: Update Tests (1 hour)
**File:** `tests/unit/test_flood_phenomenon.py`

Migrate tests from `test_model.py` to work with new `FloodPhenomenon` class.

---

### PHASE 3: Generic Visualization Prep (~30 min)

#### Task 3.1: Create Visualization Package
```bash
mkdir -p src/core/visualization
touch src/core/visualization/__init__.py
```

#### Task 3.2: Generic GeoJSON Converter (30 min)
**File:** `src/core/visualization/geojson.py`

```python
"""Generic GeoJSON conversion for any spatial phenomenon."""

from typing import Dict, Any
from src.core.base.phenomenon import SpatialPhenomenon

def phenomenon_to_geojson(
    phenomenon: SpatialPhenomenon,
    include_zones: bool = True,
    zone_index: int = None,
) -> Dict[str, Any]:
    """
    Convert ANY spatial phenomenon to GeoJSON.
    
    Works for floods, contagion, supply-chain, etc.
    """
    features = []
    
    for i, entity_id in enumerate(phenomenon.entity_ids):
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [
                    float(phenomenon.coordinates[i, 1]),  # lon
                    float(phenomenon.coordinates[i, 0]),  # lat
                ]
            },
            "properties": {
                "id": entity_id,
                "phenomenon_type": phenomenon.get_phenomenon_type(),
            }
        }
        
        # Add all phenomenon attributes
        for attr_name, attr_values in phenomenon.attributes.items():
            feature["properties"][attr_name] = float(attr_values[i])
        
        # Add zone data if requested
        if include_zones and phenomenon.zones:
            if zone_index is not None:
                feature["properties"]["zone"] = int(phenomenon.zones[zone_index][i])
            else:
                # Add all zones
                for z_idx, zone in enumerate(phenomenon.zones):
                    feature["properties"][f"zone_{z_idx}"] = int(zone[i])
        
        features.append(feature)
    
    return {
        "type": "FeatureCollection",
        "features": features,
        "metadata": phenomenon.to_dict(),
    }
```

---

## Summary of Changes

### New Files (6 total)
1. `src/core/base/__init__.py`
2. `src/core/base/phenomenon.py` - Abstract base class
3. `src/core/phenomena/__init__.py`
4. `src/core/phenomena/flood.py` - Refactored flood model
5. `src/core/visualization/__init__.py`
6. `src/core/visualization/geojson.py` - Generic converter
7. `tests/unit/test_base_phenomenon.py`
8. `tests/unit/test_flood_phenomenon.py` (migrated)

### Modified Files (3 total)
1. `src/core/__init__.py` - Export new abstractions
2. `src/core/model.py` - Deprecate or keep helper functions only
3. `tests/unit/test_model.py` - Update imports

### Deprecated (but keep for compatibility)
1. `src/core/model.py` - Keep `build_connectivity_matrix()` as utility
2. Add deprecation warnings for `DisasterImpactModel`

---

## Updated Directory Structure

```
src/core/
├── __init__.py                    # Updated exports
├── jax_ops.py                     # ✅ EXISTS - Generic operations
├── model.py                       # DEPRECATED - Keep utilities only
│
├── base/                          # NEW - Abstractions
│   ├── __init__.py
│   └── phenomenon.py              # SpatialPhenomenon ABC
│
├── phenomena/                     # NEW - Implementations
│   ├── __init__.py
│   └── flood.py                   # FloodPhenomenon
│
└── visualization/                 # NEW - Generic viz
    ├── __init__.py
    └── geojson.py                 # phenomenon_to_geojson()
```

---

## Testing Strategy

### Phase 1 Tests
```bash
# Test abstract base class
pytest tests/unit/test_base_phenomenon.py -v

# Should pass: Cannot instantiate abstract class
# Should pass: Subclass must implement methods
```

### Phase 2 Tests
```bash
# Test flood phenomenon
pytest tests/unit/test_flood_phenomenon.py -v

# Should pass: All existing flood tests
# Should pass: New abstraction compatibility
```

### Phase 3 Tests
```bash
# Test generic GeoJSON
pytest tests/unit/test_visualization_geojson.py -v

# Should pass: Converts flood phenomenon
# Should pass: Includes all attributes
# Should pass: Handles zones correctly
```

---

## Migration Path for Existing Code

### Before (Old Way)
```python
from src.core.model import build_model_from_data

model = build_model_from_data(df)
model.compute_zones(3, 14)
model.compute_impacts(3, 0.8)
```

### After (New Way)
```python
from src.core.phenomena.flood import build_flood_model_from_data

flood = build_flood_model_from_data(df)
flood.compute_zones({"min_water_level": 3, "max_water_level": 14})
flood.compute_impact(flood.zones, {"loss_percent": 0.8})
```

### Backward Compatibility (Transition Period)
Keep wrapper in `model.py`:
```python
# src/core/model.py
import warnings
from src.core.phenomena.flood import build_flood_model_from_data

def build_model_from_data(*args, **kwargs):
    warnings.warn(
        "build_model_from_data is deprecated. "
        "Use flood.build_flood_model_from_data instead.",
        DeprecationWarning
    )
    return build_flood_model_from_data(*args, **kwargs)
```

---

## Time Estimate

| Phase | Task | Time |
|-------|------|------|
| 1.1 | Create abstract base class | 1.0h |
| 1.2 | Add base tests | 0.5h |
| 1.3 | Update core init | 0.5h |
| 2.1 | Create phenomena package | 0.25h |
| 2.2 | Implement FloodPhenomenon | 1.5h |
| 2.3 | Add factory function | 0.5h |
| 2.4 | Update/migrate tests | 1.0h |
| 3.1 | Create visualization package | 0.1h |
| 3.2 | Generic GeoJSON converter | 0.5h |
| **TOTAL** | | **~6 hours** |

Reduced from 8 hours since jax_ops.py already exists!

---

## Validation Checklist

After refactoring, verify:

- [ ] Abstract base class defines clear interface
- [ ] FloodPhenomenon implements all abstract methods
- [ ] All existing tests pass (with updated imports)
- [ ] Generic GeoJSON works with FloodPhenomenon
- [ ] Backward compatibility maintained
- [ ] No flood-specific logic in base classes
- [ ] Documentation updated
- [ ] Ready for TICKET 10-VIZ

---

## Next: TICKET 10-VIZ Can Use Generic Approach

Once refactoring is complete, visualization will:

✅ Work with FloodPhenomenon  
✅ Work with future ContagionPhenomenon  
✅ Work with future SupplyChainPhenomenon  
✅ Use same `phenomenon_to_geojson()` for all  
✅ API endpoints phenomenon-agnostic  

---

**Ready to proceed?** 

With ~6 hours of refactoring, we establish proper multi-phenomenon foundations, then continue with visualization tickets.

**Next Command:** Create branch and start Phase 1?

