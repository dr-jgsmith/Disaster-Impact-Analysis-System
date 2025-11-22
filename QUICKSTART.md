# DIAS Quick Start Guide 🚀

Complete guide to running the DIAS spatial visualization system!

---

## Prerequisites

- ✅ Docker Desktop running
- ✅ Port 8000 available
- ✅ Modern web browser (Chrome, Firefox, Safari)

---

## Step 1: Start the Docker Container

Open your terminal and run:

```bash
cd /Users/justinsmith/Projects/Disaster-Impact-Analysis-System

# Stop any existing containers
docker-compose down

# Start fresh
docker-compose up -d
```

**Expected output:**
```
✔ Container dias-api  Started
```

---

## Step 2: Verify API is Running

Check the API health:

```bash
# Health check
curl http://localhost:8000/health

# Should return:
# {"status":"healthy","service":"DIAS API","version":"1.0.0"}
```

Or check the API info:

```bash
curl http://localhost:8000/info

# Should return system information
```

---

## Step 3: View Container Logs (Optional)

To see what's happening:

```bash
# View logs
docker-compose logs -f dias-api

# Press Ctrl+C to exit
```

---

## Step 4: Open the Leaflet.js Visualization

**Option A: Direct File (Quick)**
```bash
open frontend/index.html
```

**Option B: HTTP Server (Recommended)**
```bash
# Using Python 3
cd /Users/justinsmith/Projects/Disaster-Impact-Analysis-System
python3 -m http.server 8080

# Then open: http://localhost:8080/frontend/index.html
```

**Option C: Node.js HTTP Server**
```bash
npx http-server -p 8080

# Then open: http://localhost:8080/frontend/index.html
```

---

## Step 5: Run a Test Simulation

Once the frontend is open, follow the 5-step workflow in the sidebar:

### 📋 Step-by-Step Workflow

1. **📊 Generate Test Data**
   - Click "Generate Sample Data"
   - Creates 20 sample parcels in Houston, TX
   - You'll see: "✓ Generated 20 sample parcels"

2. **🏗️ Create Spatial Event**
   - Click "Create Flood Event"
   - Sends data to API at `/api/v1/sp_events`
   - You'll see: "✓ Created flood event: flood_xxxxx"

3. **🌊 Compute Flood Zones**
   - Set water levels (default: min=3.0m, max=14.0m)
   - Click "Compute Zones"
   - Calculates 10 scenarios
   - You'll see: "✓ Computed 10 flood scenarios"

4. **💰 Compute Economic Impact**
   - Set loss percentage (default: 80%)
   - Click "Compute Impact"
   - Calculates property losses
   - You'll see: "✓ Computed economic impact"

5. **🗺️ Visualize on Map**
   - Use slider to select scenario (0-9)
   - Click "Show on Map"
   - Map updates with color-coded parcels:
     - 🟢 **Green**: No impact
     - 🟠 **Orange**: Low impact
     - 🔴 **Red**: High impact

### 📍 Map Interaction

- **Pan**: Click and drag
- **Zoom**: Scroll wheel or +/- buttons
- **Parcel Info**: Click any parcel for details:
  - Parcel ID
  - Location (lat/lon)
  - Elevation
  - Land & Building Value
  - Flood Zone
  - Impact Intensity
  - Economic Loss

### 📊 Statistics Panel

Watch the stats panel update:
- **Current Water Level**: Changes with scenario slider
- **Affected Parcels**: Number of flooded parcels
- **Total Economic Loss**: Sum of all losses

---

## Example Session

Here's what a complete test run looks like:

```
1. Generate Sample Data
   → "✓ Generated 20 sample parcels"
   
2. Create Flood Event
   → "✓ Created flood event: flood_a1b2c3d4"
   
3. Compute Zones (min: 3.0m, max: 14.0m)
   → "✓ Computed 10 flood scenarios"
   
4. Compute Impact (80% loss)
   → "✓ Computed economic impact"
   
5. Visualize (Scenario 5: Water Level = 9.2m)
   → Map shows 12 affected parcels
   → Total loss: $1,245,678
   → 8 high-impact (red), 4 low-impact (orange)
```

---

## API Endpoints Reference

The frontend uses these API endpoints:

```bash
# Create spatial event
POST http://localhost:8000/api/v1/sp_events

# Compute flood zones
POST http://localhost:8000/api/v1/sp_events/{id}/zones

# Compute impact
POST http://localhost:8000/api/v1/sp_events/{id}/impact

# Get GeoJSON for visualization
GET http://localhost:8000/api/v1/sp_events/{id}/geojson?scenario=5

# Get summary statistics
GET http://localhost:8000/api/v1/sp_events/{id}/summary

# List all events
GET http://localhost:8000/api/v1/sp_events

# Delete event
DELETE http://localhost:8000/api/v1/sp_events/{id}
```

---

## Manual API Testing (cURL)

Want to test the API directly? Here are some examples:

### 1. Create Event

```bash
curl -X POST http://localhost:8000/api/v1/sp_events \
  -H "Content-Type: application/json" \
  -d '{
    "event_type": "flood",
    "parcel_ids": ["P001", "P002"],
    "coordinates": [[29.76, -95.37], [29.77, -95.38]],
    "elevations": [5.0, 10.0],
    "land_values": [100000, 150000],
    "building_values": [200000, 300000],
    "use_geodesic": true,
    "proximity_threshold": 1000.0
  }'
```

### 2. Compute Zones

```bash
curl -X POST http://localhost:8000/api/v1/sp_events/{EVENT_ID}/zones \
  -H "Content-Type: application/json" \
  -d '{
    "min_water_level": 3.0,
    "max_water_level": 14.0,
    "num_scenarios": 10
  }'
```

### 3. Get GeoJSON

```bash
curl http://localhost:8000/api/v1/sp_events/{EVENT_ID}/geojson?scenario=5
```

---

## Troubleshooting

### Port 8000 Already in Use

```bash
# Find process using port 8000
lsof -ti:8000

# Kill it
lsof -ti:8000 | xargs kill -9

# Or use a different port in docker-compose.yml
```

### Container Won't Start

```bash
# Check logs
docker-compose logs dias-api

# Rebuild from scratch
docker-compose down -v
docker-compose build --no-cache
docker-compose up -d
```

### Import Errors

```bash
# Check Python packages in container
docker exec dias-api pip list

# Verify JAX installed
docker exec dias-api python -c "import jax; print(jax.__version__)"
```

### Frontend Can't Connect to API

1. Check API is running: `curl http://localhost:8000/health`
2. Check browser console for CORS errors
3. Ensure frontend is using `http://localhost:8000` not `https`
4. Try opening frontend via HTTP server instead of file://

---

## Stopping the System

When you're done:

```bash
# Stop container (keeps data)
docker-compose stop

# Stop and remove container (keeps data volumes)
docker-compose down

# Stop and remove everything including data
docker-compose down -v
```

---

## Next Steps

Once you have the basic visualization working:

1. **Try Different Scenarios**
   - Adjust water level ranges
   - Change loss percentages
   - Generate different parcel data

2. **Explore the Code**
   - `src/core/sp_events/flood_event.py` - Flood modeling logic
   - `src/api/routes/sp_events.py` - API endpoints
   - `frontend/index.html` - Visualization code

3. **Add New Event Types**
   - Create new class extending `SpatialEvent`
   - Implement `compute_zones()` and `compute_impact()`
   - Add to API routes

4. **Extend Visualization**
   - Add heatmaps
   - Time-series animation
   - Comparison views

---

## Quick Reference Card

| Action | Command |
|--------|---------|
| **Start System** | `docker-compose up -d` |
| **Check Health** | `curl http://localhost:8000/health` |
| **View Logs** | `docker-compose logs -f` |
| **Open Frontend** | `open frontend/index.html` |
| **Stop System** | `docker-compose down` |
| **Rebuild** | `docker-compose build --no-cache` |

---

## Support

- **Documentation**: See `SESSION-COMPLETE.md`, `DEPLOYMENT_GUIDE.md`
- **API Docs**: http://localhost:8000/docs (Swagger UI)
- **Tests**: `docker exec dias-api pytest -v`

---

**🎉 You're Ready! Start with Step 1 and enjoy visualizing spatial flood events!**

