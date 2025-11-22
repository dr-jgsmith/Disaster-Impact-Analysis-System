# DIAS Leaflet.js Visualization Frontend

Interactive web-based visualization for the Disaster Impact Analysis System using Leaflet.js.

## Quick Start

### 1. Start the API

```bash
# From project root
docker-compose up -d

# Verify API is running
curl http://localhost:8000/health
```

### 2. Open the Visualization

```bash
# From project root
open frontend/index.html

# Or use Python's HTTP server
cd frontend
python3 -m http.server 8080
# Then open: http://localhost:8080
```

### 3. Run Through Test Case

1. **Load Test Data** - Click "Load Sample Flood Data" to generate 20 sample parcels
2. **Create Phenomenon** - Click "Create Flood Model" to send data to API
3. **Compute Zones** - Adjust water levels and click "Compute Flood Zones"
4. **Compute Impact** - Set loss percentage and click "Calculate Property Loss"
5. **Visualize** - Use the scenario slider to see different water levels

## Features

### Interactive Map
- **Leaflet.js** map with OpenStreetMap tiles
- **Circle markers** for each parcel
- **Color-coded** by flood impact:
  - 🟢 Green = Not flooded
  - 🟠 Orange = Low impact
  - 🔴 Red = High impact

### Sidebar Controls
- **Test data generation** (20 sample parcels in Houston area)
- **Water level configuration** (min/max range)
- **Loss percentage** adjustment
- **Scenario slider** to navigate through different water levels
- **Real-time statistics** (water level, affected parcels, total loss)

### API Integration
- Connects to DIAS API at `http://localhost:8000`
- Uses all API endpoints:
  - `POST /api/v1/phenomena` - Create phenomenon
  - `POST /api/v1/phenomena/{id}/zones` - Compute zones
  - `POST /api/v1/phenomena/{id}/impact` - Compute impact
  - `GET /api/v1/phenomena/{id}/geojson` - Get visualization data

## Architecture

```
┌─────────────────────────────────────┐
│  Frontend (index.html)              │
│  - Leaflet.js map                   │
│  - Scenario controls                │
│  - Statistics display               │
└────────────┬────────────────────────┘
             │ HTTP/REST
             ↓
┌─────────────────────────────────────┐
│  DIAS API (Docker)                  │
│  http://localhost:8000              │
└────────────┬────────────────────────┘
             │
             ↓
┌─────────────────────────────────────┐
│  FloodPhenomenon                    │
│  - JAX computations                 │
│  - GeoJSON conversion               │
└─────────────────────────────────────┘
```

## Usage Guide

### Step-by-Step Workflow

#### 1. Load Test Data
Generates 20 random parcels in Houston area with:
- Random elevations (3-13 feet)
- Random property values ($50k-$450k total)
- Simple adjacency matrix

#### 2. Create Phenomenon
Sends data to API to create a `FloodPhenomenon` instance.
Returns a phenomenon ID (e.g., `flood_abc12345`).

#### 3. Compute Zones
Configure water level range (e.g., 3-14 feet).
API computes connected flood zones for each water level.

#### 4. Compute Impact
Set loss percentage (e.g., 80%).
API calculates property value loss based on:
- How far below water level
- Loss percentage
- Property values

#### 5. Visualize & Explore
- Use scenario slider to see different water levels
- Click parcels for detailed information
- Watch statistics update in real-time

### Sample Test Case

**Scenario:** Hurricane storm surge in Houston

```
Settings:
- Min Water Level: 3 feet
- Max Water Level: 14 feet
- Loss Percentage: 80%

Results (varies by random data):
- Scenario 0 (3ft): 2-5 parcels flooded, ~$200k loss
- Scenario 5 (8ft): 10-15 parcels flooded, ~$1.5M loss
- Scenario 11 (14ft): 18-20 parcels flooded, ~$3M loss
```

## Data Format

### Generated Test Data

```javascript
{
  "phenomenon_type": "flood",
  "data": {
    "entity_ids": ["P001", "P002", ...],
    "coordinates": [[29.76, -95.37], [29.77, -95.38], ...],
    "adjacency_matrix": [[1,1,0,...], [1,1,1,...], ...],
    "attributes": {
      "elevations": [5.2, 10.8, 8.3, ...],
      "land_values": [100000, 150000, ...],
      "building_values": [200000, 250000, ...]
    }
  }
}
```

### GeoJSON Response

```json
{
  "type": "FeatureCollection",
  "features": [{
    "type": "Feature",
    "geometry": {
      "type": "Point",
      "coordinates": [-95.37, 29.76]
    },
    "properties": {
      "id": "P001",
      "elevations": 5.2,
      "land_values": 100000,
      "building_values": 200000,
      "zone": 1,
      "impact_0": 2.4
    }
  }]
}
```

## Customization

### Change Map Center
```javascript
// In initMap()
map = L.map('map').setView([YOUR_LAT, YOUR_LON], ZOOM);
```

### Adjust Color Scheme
```javascript
// In visualizeGeoJSON(), pointToLayer
let color = '#2ecc71';  // Not flooded (green)
if (isFlooded) {
    color = impact > 5 ? '#e74c3c' : '#f39c12';  // High/Low (red/orange)
}
```

### Add More Statistics
```javascript
// In updateStats()
document.getElementById('newStat').textContent = value;
```

## Troubleshooting

### Map Not Loading
- Check browser console for errors
- Verify Leaflet CDN is accessible
- Ensure you're not blocking external resources

### API Connection Failed
```bash
# Verify API is running
curl http://localhost:8000/health

# Check CORS headers
curl -H "Origin: http://localhost:8080" \
  -H "Access-Control-Request-Method: POST" \
  -H "Access-Control-Request-Headers: Content-Type" \
  -X OPTIONS http://localhost:8000/api/v1/phenomena
```

### No Data on Map
- Check browser console for fetch errors
- Verify phenomenon was created (check sidebar status)
- Ensure zones and impact were computed before visualizing

## Browser Compatibility

- ✅ Chrome 90+
- ✅ Firefox 88+
- ✅ Safari 14+
- ✅ Edge 90+

## Dependencies

All loaded via CDN (no npm required):
- **Leaflet 1.9.4** - https://unpkg.com/leaflet@1.9.4/

## Future Enhancements

Potential improvements:
- [ ] File upload for custom parcel data
- [ ] Export visualization as PNG/PDF
- [ ] Timeline animation of flooding progression
- [ ] Heatmap overlay for impact intensity
- [ ] Multi-phenomenon support (contagion, supply-chain)
- [ ] Save/load scenarios
- [ ] Comparison view (before/after)
- [ ] Statistics charts (Chart.js)

## Performance

- Handles 100+ parcels smoothly
- Real-time scenario switching
- Lazy loading of GeoJSON per scenario
- Efficient Leaflet rendering

## License

Same as DIAS project.

## Support

For issues:
1. Check browser console
2. Verify API is running
3. Review API logs: `docker-compose logs`
4. Open GitHub issue

