# TICKET 0: Create Test Dataset Generation Script

**Type:** Task  
**Priority:** Critical  
**Estimated Effort:** 4 hours

## Description
Create a Python script that generates realistic synthetic test data for DIAS development and testing. This includes parcel data with spatial attributes and corresponding elevation data in GIS-compatible formats.

## Acceptance Criteria
- [ ] Script generates DBF file with parcel data
- [ ] Script generates CSV file with elevation data
- [ ] Data includes realistic coordinate ranges
- [ ] All required fields present (parcel ID, lat/lon, land value, building value)
- [ ] Sample data covers elevation range suitable for flood testing (0-20 feet)
- [ ] At least 100 sample parcels generated
- [ ] Documentation includes instructions for regenerating data

## Required Output Files

### 1. Parcel Data (DBF format)
**Filename:** `test_data/parcels_test.dbf`

**Required Fields:**
- `PARCELID` (String, 10 chars) - Unique parcel identifier (e.g., "P00001", "P00002")
- `LAT` (Float) - Latitude in decimal degrees
- `LON` (Float) - Longitude in decimal degrees
- `LANDVALUE` (Float) - Land value in USD (e.g., 50000-150000)
- `BLDGVALUE` (Float) - Building value in USD (e.g., 100000-500000)
- `LANDUSE` (String, 20 chars) - Land use type (e.g., "Residential", "Commercial", "Industrial")
- `STRUCTURE` (String, 1 char) - Structure present: "Y" or "N"
- `OWNER` (String, 50 chars) - Owner name (optional, can be generic)
- `ZONING` (String, 10 chars) - Zoning classification (optional)

### 2. Elevation Data (CSV format)
**Filename:** `test_data/elevations_test.csv`

**Required Fields:**
- `PARCELID` - Matches parcel ID from DBF file
- `ELEVATION` - Elevation in feet (range: 0-20 feet for flood testing)
- `LAT` - Latitude (matches parcel data)
- `LON` - Longitude (matches parcel data)

## Technical Requirements

### Geographic Area
Use a realistic coastal area for testing (e.g., Houston, TX or Miami, FL):
- **Houston Example:** 
  - Latitude range: 29.65 to 29.85
  - Longitude range: -95.50 to -95.30
- **Miami Example:**
  - Latitude range: 25.70 to 25.85
  - Longitude range: -80.30 to -80.15

### Elevation Distribution
Generate elevations that create interesting flood scenarios:
- 20% of parcels: 0-5 feet (high flood risk)
- 30% of parcels: 5-10 feet (moderate risk)
- 30% of parcels: 10-15 feet (lower risk)
- 20% of parcels: 15-20 feet (minimal risk)

### Value Distribution
Realistic property values:
- Land value: $30,000 - $200,000 (normal distribution, mean ~$80,000)
- Building value: $100,000 - $600,000 (normal distribution, mean ~$250,000)
- Commercial properties: 1.5x - 2x higher values
- Vacant land (no structure): $0 building value

## Implementation Steps

1. **Create the generation script** at `scripts/generate_test_data.py`:
   ```python
   """
   Generate synthetic test data for DIAS testing.
   
   Usage:
       python scripts/generate_test_data.py --parcels 100 --output test_data/
   """
   import argparse
   import numpy as np
   import pandas as pd
   from dbfread import DBF
   from dbfpy import dbf  # or use simpledbf/pyshp
   ```

2. **Install required dependencies** (add to requirements):
   - `dbfpy` or `simpledbf` or `pyshp` for DBF writing
   - `numpy` and `pandas` (already required)

3. **Generate parcel locations** with realistic spatial distribution:
   - Use random distribution within bounding box
   - Optionally cluster some parcels (residential neighborhoods)
   - Ensure minimum distance between parcels (~50-100 feet)

4. **Generate elevation data** with spatial correlation:
   - Parcels closer together should have similar elevations
   - Create gradual elevation changes (not random)
   - Can use simple distance-weighted interpolation

5. **Generate property values** correlated with elevation:
   - Higher elevation properties may have higher values
   - Add random variation
   - Commercial properties get higher values

6. **Write DBF file** using appropriate library

7. **Write CSV file** for elevations

8. **Create data validation function** to verify output

## Example Script Structure

```python
import argparse
import numpy as np
import pandas as pd
from dbfpy3 import dbf
import os

def generate_coordinates(n_parcels, bbox, min_distance=0.001):
    """Generate spatially distributed coordinates."""
    # Implementation
    pass

def generate_elevations(coords, elevation_range=(0, 20)):
    """Generate spatially correlated elevations."""
    # Use distance-based interpolation from random seed points
    pass

def generate_property_values(elevations, has_structure):
    """Generate realistic property values."""
    # Higher elevation -> potentially higher value
    # Add realistic variation
    pass

def write_dbf(data, filename):
    """Write data to DBF file."""
    pass

def write_csv(data, filename):
    """Write elevation data to CSV."""
    pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parcels', type=int, default=100)
    parser.add_argument('--output', default='test_data/')
    parser.add_argument('--region', default='houston', 
                       choices=['houston', 'miami'])
    args = parser.parse_args()
    
    # Generate data
    # Write files
    # Validate output
    
if __name__ == '__main__':
    main()
```

## Testing the Generated Data

Create a validation script `scripts/validate_test_data.py`:

```python
"""Validate generated test data."""
import pandas as pd
from dbfread import DBF

def validate_parcel_data(dbf_file):
    """Validate DBF file structure and content."""
    # Check all required fields present
    # Check data types
    # Check value ranges
    # Check for duplicates
    pass

def validate_elevation_data(csv_file, dbf_file):
    """Validate elevation data matches parcels."""
    # Check all parcel IDs match
    # Check elevation ranges
    # Check coordinates match
    pass
```

## Documentation

Create `test_data/README.md`:

```markdown
# Test Data for DIAS

## Files
- `parcels_test.dbf` - Parcel attribute data (100 parcels)
- `elevations_test.csv` - Elevation data for each parcel

## Regenerating Data
To regenerate test data:
```bash
python scripts/generate_test_data.py --parcels 100 --region houston
```

## Data Description
- **Region:** Houston, TX area
- **Parcels:** 100 synthetic parcels
- **Elevation Range:** 0-20 feet
- **Property Values:** Realistic residential/commercial ranges

## Using Test Data
```python
from dias.scripts.base_model import build_base_model

file = "test_data/parcels_test.dbf"
elevations = "test_data/elevations_test.csv"
lat = 'LAT'
lon = 'LON'
parcel_field = 'PARCELID'
building_value_field = 'BLDGVALUE'
land_value_field = 'LANDVALUE'

model = build_base_model(file, elevations, lat, lon, 
                        max_impact=14, impact_multiplier=0.8)
```
```

## Dependencies
None - this can be done in parallel with other tickets

## Notes for Implementation
- Use `dbfpy3` (Python 3 compatible) or `pyshp` for DBF writing
- Consider using `scipy.spatial` for spatial correlation in elevation generation
- Generate reproducible data (use fixed random seed)
- Include example showing how to load and use the test data

## Deliverables
- [ ] `scripts/generate_test_data.py` - Main generation script
- [ ] `scripts/validate_test_data.py` - Validation script
- [ ] `test_data/parcels_test.dbf` - Generated parcel data
- [ ] `test_data/elevations_test.csv` - Generated elevation data
- [ ] `test_data/README.md` - Documentation
- [ ] Requirements updated with DBF writing library

## Success Criteria
Test data can be successfully loaded by existing DIAS code:
```python
from dias.scripts.base_model import build_base_model
model = build_base_model('test_data/parcels_test.dbf', 
                        'test_data/elevations_test.csv', 
                        'LAT', 'LON', 14, 0.8)
# Should complete without errors
```

