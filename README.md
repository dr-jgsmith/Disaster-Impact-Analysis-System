# Disaster Impact Analysis System (DIAS) v2.0

## Overview

The Disaster Impact Analysis System (DIAS) is a modern, containerized service for simulating the impacts of recurring flood events on real estate prices and assessing flood mitigation strategies using multi-criteria decision methods. DIAS implements methods for representing the connectivity of urban spaces to model hydrologic events such as floods and storm surges.

**Version 2.0** represents a complete modernization with:
- ✅ Python 3.9+ support
- ✅ JAX optimization engine (replacing Numba)
- ✅ RESTful API service architecture
- ✅ Docker containerization
- ✅ Comprehensive test coverage
- ✅ Modern development practices

## Quick Start with Docker

```bash
# 1. Clone the repository
git clone https://github.com/dr-jgsmith/Disaster-Impact-Analysis-System
cd Disaster-Impact-Analysis-System

# 2. Set up environment variables
cp .env.example .env
# Edit .env with your configuration

# 3. Start the service
cd docker
docker-compose up -d

# 4. Verify service is running
curl http://localhost:8000/api/v1/health
```

## Architecture

DIAS v2.0 is organized as a service with clear separation of concerns:

```
dias-service/
├── src/
│   ├── api/          # REST API endpoints (FastAPI)
│   ├── core/         # Core business logic (models, simulation)
│   ├── utils/        # Utility functions
│   └── config/       # Configuration management
├── tests/
│   ├── unit/         # Unit tests
│   ├── integration/  # Integration tests
│   └── fixtures/     # Test fixtures and sample data
├── docker/           # Docker configuration
├── test_data/        # Sample test data
├── scripts/          # Utility scripts
└── docs/             # Comprehensive documentation
```

## Features

### Core Capabilities

- **Flood Impact Modeling**: Simulate recurring flood events on real estate values
- **Connectivity Analysis**: Model water flow and inundation zones based on elevation
- **Multi-Criteria Decision Analysis**: Evaluate mitigation strategies using MCQA
- **Spatial Analysis**: Process GIS data (DBF, CSV formats)
- **REST API**: Programmatic access to all functionality

### Technical Stack

- **Python 3.9+**: Modern Python with type hints
- **JAX**: High-performance numerical computations
- **FastAPI**: Modern, fast web framework
- **Docker**: Containerized deployment
- **pytest**: Comprehensive testing framework

## API Usage

### Build a Model

```bash
curl -X POST http://localhost:8000/api/v1/models/build \
  -F "file=@parcels.dbf" \
  -F "elevations=@elevations.csv" \
  -H "Content-Type: multipart/form-data" \
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

### Run Simulation

```bash
curl -X POST http://localhost:8000/api/v1/models/{model_id}/simulate \
  -H "Content-Type: application/json" \
  -d '{
    "impact_range": [3, 14],
    "iterations": 500,
    "time_step": 25
  }'
```

See [API Documentation](docs/API_SPECIFICATION.md) for complete endpoint reference.

## Development Setup

### Prerequisites

- Python 3.9 or higher
- Docker 20.10+ and Docker Compose 2.0+ (for containerized development)
- Git

### Local Development

```bash
# 1. Create virtual environment
python3.9 -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# 2. Install dependencies
pip install -r requirements/dev.txt

# 3. Run tests
pytest

# 4. Run linters
bash scripts/lint.sh

# 5. Format code
bash scripts/format.sh

# 6. Start development server
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

See [Development Guide](docs/DEVELOPMENT.md) for detailed instructions.

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test suite
pytest tests/unit/
pytest tests/integration/
```

See [Testing Guide](docs/TESTING.md) for more information.

## Data Format

DIAS works with standard GIS data formats:

### Parcel Data (DBF or CSV)

Required fields:
- `PARCELID` - Unique parcel identifier
- `LAT` - Latitude (decimal degrees)
- `LON` - Longitude (decimal degrees)
- `LANDVALUE` - Land value (USD)
- `BLDGVALUE` - Building value (USD)

### Elevation Data (CSV)

Required fields:
- `PARCELID` - Matches parcel data
- `ELEVATION` - Elevation (feet)

Sample test data is provided in `test_data/` directory.

## Migration from V1

If you're upgrading from the original DIAS package:

1. **API-Based**: DIAS is now a service, not a Python package
2. **JAX Instead of Numba**: New optimization engine for better performance
3. **Docker-First**: Run in containers for consistency
4. **Type-Safe**: Full type hints for better IDE support

See [Migration Guide](docs/MIGRATION_FROM_V1.md) for detailed migration instructions.

## Documentation

- **[API Specification](docs/API_SPECIFICATION.md)** - Complete API reference
- **[Deployment Guide](docs/DEPLOYMENT.md)** - Production deployment
- **[Development Guide](docs/DEVELOPMENT.md)** - Development setup and workflow
- **[Testing Guide](docs/TESTING.md)** - Testing strategy and examples
- **[Architecture](docs/)** - System architecture and design decisions

## Contributing

We welcome contributions! Please see [Development Guide](docs/DEVELOPMENT.md) for:

- Code style guidelines
- Testing requirements
- Pull request process
- Development workflow

## Performance

DIAS v2.0 with JAX provides:
- Sub-second response times for typical analyses
- Efficient memory usage with JAX's JIT compilation
- Scalable architecture for large datasets
- Near-C performance for numerical computations

## License

MIT License - see [LICENSE](LICENSE) for details

## Authors

- **Justin G. Smith** - Original author and maintainer
- Email: justingriffis@wsu.edu
- GitHub: [@dr-jgsmith](https://github.com/dr-jgsmith)

## Citation

If you use DIAS in your research, please cite:

```bibtex
@software{smith2024dias,
  author = {Smith, Justin G.},
  title = {Disaster Impact Analysis System (DIAS)},
  year = {2024},
  version = {2.0.0},
  url = {https://github.com/dr-jgsmith/Disaster-Impact-Analysis-System}
}
```

## Support

- **Issues**: [GitHub Issues](https://github.com/dr-jgsmith/Disaster-Impact-Analysis-System/issues)
- **Discussions**: [GitHub Discussions](https://github.com/dr-jgsmith/Disaster-Impact-Analysis-System/discussions)
- **Email**: justingriffis@wsu.edu

## Changelog

### Version 2.0.0 (2024)

- Complete rewrite as containerized service
- Migration from Python 3.6 to Python 3.9+
- Replaced Numba with JAX for optimization
- Added REST API with FastAPI
- Comprehensive test coverage (80%+)
- Docker containerization
- Modern development practices

### Version 1.0 (Original)

- Python 3.6 package
- Numba JIT compilation
- Jupyter Notebook interface
- Basic flood modeling capabilities

---

**Status**: ✅ Production Ready | **Version**: 2.0.0 | **Python**: 3.9+
