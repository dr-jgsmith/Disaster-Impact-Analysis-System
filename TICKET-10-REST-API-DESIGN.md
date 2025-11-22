# TICKET 10: Design REST API Endpoints

**Status:** In Progress  
**Estimated Time:** 4-6 hours  
**Priority:** High  
**Dependencies:** TICKET 9 (Core Model Migration) ✅, Multi-Phenomenon Refactoring ✅

---

## Objective

Design and document RESTful API endpoints that leverage the multi-phenomenon architecture. The API should be phenomenon-agnostic, supporting floods, contagion, supply-chain disruptions, and future phenomena types.

---

## Background

With the multi-phenomenon refactoring complete, we now have:
- ✅ `SpatialPhenomenon` abstract base class
- ✅ `FloodPhenomenon` implementation
- ✅ Generic `phenomenon_to_geojson()` conversion
- ✅ Extensible architecture

The API should provide endpoints for:
1. Creating/loading phenomena
2. Computing zones/scenarios
3. Computing impacts
4. Retrieving results (GeoJSON, summary stats, etc.)
5. Visualization support (Leaflet.js integration)

---

## API Design Principles

### 1. RESTful Design
- Use standard HTTP methods (GET, POST, PUT, DELETE)
- Resource-based URLs
- Stateless operations where possible

### 2. Phenomenon-Agnostic
- API should work with ANY phenomenon type
- Phenomenon type specified in request body
- Same endpoints for flood, contagion, supply-chain, etc.

### 3. GeoJSON-First
- Primary data format is GeoJSON (Leaflet.js compatible)
- Support for JSON summary statistics
- Optional CSV/DataFrame export

### 4. Async Support
- FastAPI async endpoints
- Support for long-running computations
- Optional task queue for heavy processing

---

## Proposed Endpoint Structure

### Base URL
```
http://localhost:8000/api/v1
```

### Health & Info
```
GET  /health                    # Health check
GET  /info                      # API version, capabilities
```

### Phenomena Management
```
POST   /phenomena               # Create phenomenon from data
GET    /phenomena/{id}          # Get phenomenon info
DELETE /phenomena/{id}          # Delete phenomenon
GET    /phenomena               # List all phenomena (paginated)
```

### Computation
```
POST   /phenomena/{id}/zones    # Compute zones/scenarios
POST   /phenomena/{id}/impact   # Compute impact metrics
GET    /phenomena/{id}/status   # Get computation status
```

### Results & Visualization
```
GET    /phenomena/{id}/geojson               # Get GeoJSON (all scenarios)
GET    /phenomena/{id}/geojson/{scenario}    # Get specific scenario
GET    /phenomena/{id}/summary               # Get summary statistics
GET    /phenomena/{id}/zones/{zone}/bounds   # Get zone bounding box
GET    /phenomena/{id}/zones/{zone}/stats    # Get zone statistics
```

### Export
```
GET    /phenomena/{id}/export/geojson        # Download all scenarios as ZIP
GET    /phenomena/{id}/export/csv            # Export as CSV
GET    /phenomena/{id}/export/dataframe      # Export as JSON (DataFrame format)
```

---

## Detailed Endpoint Specifications

### 1. Create Phenomenon

**Endpoint:** `POST /api/v1/phenomena`

**Purpose:** Create a new spatial phenomenon from uploaded data

**Request Body:**
```json
{
  "phenomenon_type": "flood",
  "data": {
    "entity_ids": ["P001", "P002", "P003"],
    "coordinates": [[29.76, -95.37], [29.77, -95.38], [29.78, -95.39]],
    "adjacency_matrix": [[1, 1, 0], [1, 1, 1], [0, 1, 1]],
    "attributes": {
      "elevations": [5.0, 10.0, 8.0],
      "land_values": [100000, 150000, 120000],
      "building_values": [200000, 250000, 220000]
    }
  },
  "options": {
    "use_geodesic": true,
    "proximity_threshold": null
  }
}
```

**Response (201 Created):**
```json
{
  "id": "flood_abc123",
  "phenomenon_type": "flood",
  "n_entities": 3,
  "created_at": "2024-11-21T17:30:00Z",
  "status": "ready",
  "links": {
    "self": "/api/v1/phenomena/flood_abc123",
    "compute_zones": "/api/v1/phenomena/flood_abc123/zones",
    "compute_impact": "/api/v1/phenomena/flood_abc123/impact",
    "geojson": "/api/v1/phenomena/flood_abc123/geojson"
  }
}
```

---

### 2. Compute Zones

**Endpoint:** `POST /api/v1/phenomena/{id}/zones`

**Purpose:** Compute zones/scenarios for phenomenon

**Request Body (Flood):**
```json
{
  "scenario_params": {
    "min_water_level": 3.0,
    "max_water_level": 14.0
  }
}
```

**Request Body (Contagion - future):**
```json
{
  "scenario_params": {
    "transmission_rate": 0.3,
    "time_steps": 30,
    "initial_infected": ["P001", "P005"]
  }
}
```

**Response (200 OK):**
```json
{
  "phenomenon_id": "flood_abc123",
  "n_scenarios": 12,
  "scenarios_computed": true,
  "computation_time_ms": 150,
  "links": {
    "geojson": "/api/v1/phenomena/flood_abc123/geojson",
    "compute_impact": "/api/v1/phenomena/flood_abc123/impact"
  }
}
```

---

### 3. Compute Impact

**Endpoint:** `POST /api/v1/phenomena/{id}/impact`

**Purpose:** Compute impact metrics for computed zones

**Request Body (Flood):**
```json
{
  "scenario_params": {
    "loss_percent": 0.8,
    "min_water_level": 3.0
  }
}
```

**Response (200 OK):**
```json
{
  "phenomenon_id": "flood_abc123",
  "impact_metrics": {
    "n_scenarios": 12,
    "total_property_loss": [0, 50000, 125000, ...],
    "affected_parcels": [0, 2, 5, ...],
    "mean_impacts": [0.0, 0.5, 1.2, ...],
    "total_land_value": 370000,
    "total_building_value": 670000
  },
  "computation_time_ms": 75,
  "links": {
    "geojson": "/api/v1/phenomena/flood_abc123/geojson",
    "summary": "/api/v1/phenomena/flood_abc123/summary"
  }
}
```

---

### 4. Get GeoJSON

**Endpoint:** `GET /api/v1/phenomena/{id}/geojson`

**Query Parameters:**
- `scenario` (int, optional): Specific scenario index
- `include_zones` (bool, default=true): Include zone data
- `include_impacts` (bool, default=true): Include impact data
- `include_attributes` (bool, default=true): Include phenomenon attributes

**Response (200 OK):**
```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Point",
        "coordinates": [-95.37, 29.76]
      },
      "properties": {
        "id": "P001",
        "phenomenon_type": "flood",
        "elevations": 5.0,
        "land_values": 100000.0,
        "building_values": 200000.0,
        "zone_0": 1,
        "zone_1": 1,
        "impact_0": 2.4,
        "impact_1": 3.2
      }
    }
  ],
  "metadata": {
    "phenomenon_id": "flood_abc123",
    "phenomenon_type": "flood",
    "n_entities": 3,
    "n_scenarios": 12,
    "has_zones": true,
    "has_impacts": true,
    "coordinate_bounds": {
      "min_lat": 29.76,
      "max_lat": 29.78,
      "min_lon": -95.39,
      "max_lon": -95.37
    }
  }
}
```

---

### 5. Get Summary

**Endpoint:** `GET /api/v1/phenomena/{id}/summary`

**Response (200 OK):**
```json
{
  "phenomenon_id": "flood_abc123",
  "phenomenon_type": "flood",
  "n_entities": 100,
  "n_scenarios": 12,
  "coordinate_bounds": {
    "min_lat": 29.70,
    "max_lat": 29.85,
    "min_lon": -95.45,
    "max_lon": -95.30
  },
  "elevations_range": [2.5, 15.8],
  "elevations_mean": 8.3,
  "total_land_value": 15000000,
  "total_building_value": 25000000,
  "total_property_value": 40000000,
  "mean_connectivity": 0.15,
  "impact_metrics": {
    "max_property_loss": 5000000,
    "max_affected_parcels": 45,
    "scenarios": 12
  }
}
```

---

### 6. Get Zone Bounds

**Endpoint:** `GET /api/v1/phenomena/{id}/zones/{zone_index}/bounds`

**Response (200 OK):**
```json
{
  "phenomenon_id": "flood_abc123",
  "zone_index": 5,
  "bounds": {
    "min_lat": 29.72,
    "max_lat": 29.80,
    "min_lon": -95.42,
    "max_lon": -95.35
  }
}
```

---

### 7. Get Zone Statistics

**Endpoint:** `GET /api/v1/phenomena/{id}/zones/{zone_index}/stats`

**Response (200 OK):**
```json
{
  "phenomenon_id": "flood_abc123",
  "zone_index": 5,
  "n_entities": 25,
  "percent_affected": 25.0,
  "unique_zones": 3,
  "elevations_affected": {
    "min": 3.2,
    "max": 7.5,
    "mean": 5.8,
    "sum": 145.0
  },
  "land_values_affected": {
    "min": 75000,
    "max": 200000,
    "mean": 125000,
    "sum": 3125000
  }
}
```

---

### 8. Export GeoJSON (Batch)

**Endpoint:** `GET /api/v1/phenomena/{id}/export/geojson`

**Response (200 OK):**
- Content-Type: `application/zip`
- Downloads ZIP file containing:
  - `scenario_0.geojson`
  - `scenario_1.geojson`
  - ...
  - `scenario_11.geojson`
  - `metadata.json`

---

## Error Responses

### Standard Error Format
```json
{
  "error": {
    "code": "PHENOMENON_NOT_FOUND",
    "message": "Phenomenon with ID 'flood_xyz' not found",
    "details": {
      "phenomenon_id": "flood_xyz"
    }
  }
}
```

### Error Codes
- `400 Bad Request`: Invalid request body or parameters
- `404 Not Found`: Phenomenon or resource not found
- `409 Conflict`: Operation not allowed in current state
- `422 Unprocessable Entity`: Valid syntax but semantic errors
- `500 Internal Server Error`: Server-side error

**Example Error Codes:**
- `PHENOMENON_NOT_FOUND` - Phenomenon ID doesn't exist
- `ZONES_NOT_COMPUTED` - Zones must be computed before impact
- `INVALID_PHENOMENON_TYPE` - Unknown phenomenon type
- `INVALID_SCENARIO_PARAMS` - Invalid parameters for phenomenon type
- `COMPUTATION_FAILED` - Error during zone/impact computation

---

## API Versioning

### URL Versioning
- Current: `/api/v1/`
- Future: `/api/v2/` (when breaking changes needed)

### Response Versioning
```json
{
  "api_version": "1.0.0",
  "data": { ... }
}
```

---

## CORS Configuration

For Leaflet.js web frontend:
```python
origins = [
    "http://localhost:3000",  # React dev server
    "http://localhost:8080",  # Alternative frontend
    "https://dias.example.com",  # Production
]
```

---

## Rate Limiting

- **Development:** No limits
- **Production:** 
  - 100 requests/minute per IP
  - 10 phenomenon creations/hour per IP
  - Exemptions for authenticated users

---

## Implementation Plan

### Phase 1: Basic CRUD (2 hours)
1. Create FastAPI app structure
2. Implement health and info endpoints
3. Implement phenomenon creation (POST /phenomena)
4. Implement phenomenon retrieval (GET /phenomena/{id})
5. In-memory storage for development

### Phase 2: Computation Endpoints (1.5 hours)
1. Implement compute zones endpoint
2. Implement compute impact endpoint
3. Error handling for computation failures
4. Status tracking

### Phase 3: Visualization Endpoints (1 hour)
1. Implement GeoJSON endpoint
2. Implement summary endpoint
3. Implement zone bounds/stats endpoints
4. Query parameter support

### Phase 4: Export & Polish (1.5 hours)
1. Implement batch export endpoints
2. Add comprehensive error handling
3. Add request validation (Pydantic models)
4. Add API documentation (OpenAPI/Swagger)
5. Add CORS configuration

---

## Testing Strategy

### Unit Tests
- Test each endpoint with valid inputs
- Test error conditions
- Test query parameters
- Test different phenomenon types

### Integration Tests
- Full workflow tests (create → zones → impact → geojson)
- Test with real flood data
- Test GeoJSON validity
- Performance testing

### API Documentation Tests
- Ensure OpenAPI spec is correct
- Test example requests/responses
- Validate schema compliance

---

## OpenAPI/Swagger Documentation

FastAPI will auto-generate interactive documentation at:
- `/docs` - Swagger UI
- `/redoc` - ReDoc
- `/openapi.json` - OpenAPI specification

**Example tags:**
- `phenomena` - Phenomenon management
- `computation` - Zone and impact computation
- `visualization` - GeoJSON and visualization endpoints
- `export` - Data export endpoints

---

## Success Criteria

- [ ] All endpoints implemented and functional
- [ ] API works with FloodPhenomenon
- [ ] GeoJSON output is valid and Leaflet.js compatible
- [ ] Comprehensive error handling
- [ ] Request/response validation with Pydantic
- [ ] OpenAPI documentation complete
- [ ] Unit tests for all endpoints
- [ ] Integration test for full workflow
- [ ] Ready for frontend integration

---

## Next Steps

After TICKET 10:
1. **TICKET 11:** Implement FastAPI service
2. **TICKET 12:** Create Docker configuration
3. **Frontend:** Build Leaflet.js visualization
4. **Testing:** Comprehensive API testing

---

## Notes

- API design is phenomenon-agnostic by design
- Same endpoints will work for future ContagionPhenomenon, SupplyChainPhenomenon
- GeoJSON format ensures compatibility with any mapping library
- Architecture supports both synchronous and async operations
- Can add task queue (Celery) later for heavy computations

---

**Prepared By:** AI Assistant  
**Date:** November 21, 2024  
**Status:** Ready for implementation

