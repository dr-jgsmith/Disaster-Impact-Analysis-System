# DIAS Test Data

This directory contains synthetic test data for developing and testing the Disaster Impact Analysis System (DIAS).

## Files

### Generated Files

After running the generation script, this directory will contain:

- **`parcels_test.dbf`** - Parcel attribute data in dBase format (GIS-compatible)
- **`parcels_test.csv`** - Parcel attribute data in CSV format (fallback if DBF fails)
- **`elevations_test.csv`** - Elevation data for each parcel
- **`README.txt`** - Summary of generated data with statistics

## Generating Test Data

### Quick Start

```bash
# Generate 100 parcels for Houston area
python scripts/generate_test_data.py --parcels 100 --region houston

# Generate 200 parcels for Miami area
python scripts/generate_test_data.py --parcels 200 --region miami
```

### Full Options

```bash
python scripts/generate_test_data.py \
    --parcels 100 \
    --region houston \
    --output test_data/ \
    --seed 42
```

**Parameters:**
- `--parcels`: Number of parcels to generate (default: 100)
- `--region`: Geographic region - `houston` or `miami` (default: houston)
- `--output`: Output directory (default: test_data/)
- `--seed`: Random seed for reproducibility (default: 42)

### Available Regions

**Houston, TX**
- Latitude: 29.65° to 29.85°
- Longitude: -95.50° to -95.30°
- Characteristics: Gulf Coast flood risk area

**Miami, FL**
- Latitude: 25.70° to 25.85°
- Longitude: -80.30° to -80.15°
- Characteristics: Atlantic Coast flood risk area

## Data Description

### Parcel Data Fields

| Field | Type | Description |
|-------|------|-------------|
| `PARCELID` | String | Unique parcel identifier (e.g., "P00001") |
| `LAT` | Float | Latitude in decimal degrees |
| `LON` | Float | Longitude in decimal degrees |
| `LANDVALUE` | Float | Land value in USD ($30,000 - $500,000) |
| `BLDGVALUE` | Float | Building value in USD ($0 - $1,500,000) |
| `LANDUSE` | String | Land use type (Residential, Commercial, Industrial) |
| `STRUCTURE` | String | Structure present: "Y" or "N" |
| `OWNER` | String | Property owner name (generic) |
| `ZONING` | String | Zoning classification (Z1-Z5) |

### Elevation Data Fields

| Field | Type | Description |
|-------|------|-------------|
| `PARCELID` | String | Matches parcel ID from parcel data |
| `ELEVATION` | Float | Elevation in feet (0-20 feet typical) |
| `LAT` | Float | Latitude (matches parcel data) |
| `LON` | Float | Longitude (matches parcel data) |

### Data Characteristics

**Elevation Distribution:**
- 20% of parcels: 0-5 feet (high flood risk)
- 30% of parcels: 5-10 feet (moderate risk)
- 30% of parcels: 10-15 feet (lower risk)
- 20% of parcels: 15-20 feet (minimal risk)

**Land Use Distribution:**
- 70% Residential
- 20% Commercial (higher property values)
- 10% Industrial (moderate property values)

**Structures:**
- 85% of parcels have buildings
- 15% are vacant land (BLDGVALUE = 0)

**Property Values:**
- Land: Correlated with elevation (higher = more valuable)
- Buildings: Correlated with land use and elevation
- Commercial properties have 2x multiplier
- Industrial properties have 1.5x multiplier

## Validating Test Data

After generating data, validate it:

```bash
python scripts/validate_test_data.py --data test_data/
```

This checks:
- ✅ All required fields present
- ✅ No duplicate parcel IDs
- ✅ Coordinate values in valid ranges
- ✅ Property values are non-negative
- ✅ Elevation data matches parcel data
- ✅ Consistent structure flags and building values

## Using Test Data with DIAS

### Legacy DIAS (v1.x)

```python
from dias.scripts.base_model import build_base_model

# Define field mappings
file = "test_data/parcels_test.dbf"
elevations = "test_data/elevations_test.csv"
lat = 'LAT'
lon = 'LON'
parcel_field = 'PARCELID'
building_value_field = 'BLDGVALUE'
land_value_field = 'LANDVALUE'

# Build model
model = build_base_model(
    file, 
    elevations, 
    lat, 
    lon, 
    max_impact=14, 
    impact_multiplier=0.8
)
```

### DIAS v2.0 Service

```bash
# Build a model via API
curl -X POST http://localhost:8000/api/v1/models/build \
  -F "file=@test_data/parcels_test.dbf" \
  -F "elevations=@test_data/elevations_test.csv" \
  -d '{
    "lat_field": "LAT",
    "lon_field": "LON",
    "parcel_field": "PARCELID",
    "building_value_field": "BLDGVALUE",
    "land_value_field": "LANDVALUE",
    "max_impact": 14.0,
    "impact_multiplier": 0.8
  }'
```

## Data Format Notes

### DBF Format
- Industry-standard format for GIS attribute data
- Compatible with ESRI ArcGIS, QGIS, and other GIS software
- Can be read with `dbfread` library in Python

### CSV Format
- Universal text format
- Easy to inspect and manipulate
- Can be imported into any GIS or analysis tool

## Regenerating with Different Parameters

### More Parcels
```bash
python scripts/generate_test_data.py --parcels 500
```

### Different Random Seed
```bash
python scripts/generate_test_data.py --seed 12345
```

### Different Region
```bash
python scripts/generate_test_data.py --region miami
```

## Customizing Test Data

To create custom test data:

1. Modify `scripts/generate_test_data.py`
2. Adjust parameters in the generation functions:
   - `generate_coordinates()` - spatial distribution
   - `generate_elevations()` - elevation patterns
   - `generate_property_values()` - value distributions
3. Add new regions to `REGIONS` dictionary
4. Regenerate data

## Troubleshooting

**Problem:** DBF file not generated
- **Solution:** Install dbfpy3: `pip install dbfpy3`
- **Fallback:** Use CSV file instead (functionality is the same)

**Problem:** Validation fails
- **Solution:** Regenerate data with same seed for consistency
- **Check:** Ensure both parcels and elevations files exist

**Problem:** Not enough parcels generated
- **Solution:** Reduce `min_distance` parameter or increase lat/lon range

## Real Data Integration

While this test data is synthetic, DIAS can work with real parcel data from:

- County assessor offices (tax parcel data)
- GIS departments (spatial data layers)
- FEMA flood maps (elevation and flood zone data)
- USGS elevation data (DEM files)

### Real Data Requirements
- Parcel IDs must be unique
- Coordinates in decimal degrees (WGS84)
- Property values in USD
- Elevation in feet (or convert from meters)

## License

Test data generated by this script is for development and testing purposes only. Do not use for actual flood risk assessment or financial decisions.

---

*Generated by DIAS Test Data Generator*  
*See `scripts/generate_test_data.py` for implementation details*

