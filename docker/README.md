# Docker Configuration for DIAS

This directory contains Docker configuration files for the Disaster Impact Analysis System.

## Files

- `Dockerfile` - Multi-stage Docker image build (located in project root)
- `docker-compose.yml` - Docker Compose orchestration (located in project root)
- `start.sh` - Container startup script
- `.dockerignore` - Files to exclude from Docker context
- `env.example` - Example environment variables

## Quick Start

### 1. Build and Run with Docker Compose

```bash
# From project root
docker-compose up --build
```

The API will be available at: http://localhost:8000

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
- **Health Check:** http://localhost:8000/health

### 2. Stop Services

```bash
docker-compose down
```

### 3. View Logs

```bash
docker-compose logs -f dias-api
```

## Build Options

### Development Mode

For development with hot reload:

```bash
# Set environment variable
export APP_ENV=development

# Or create .env file with:
# APP_ENV=development

docker-compose up
```

### Production Mode

```bash
export APP_ENV=production
docker-compose up -d
```

## Environment Variables

Copy `docker/env.example` to `.env` in the project root and customize:

```bash
cp docker/env.example .env
```

### Key Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_ENV` | production | Environment (development/production) |
| `LOG_LEVEL` | INFO | Logging level |
| `API_HOST` | 0.0.0.0 | API host |
| `API_PORT` | 8000 | API port |
| `API_WORKERS` | 4 | Number of uvicorn workers |
| `JAX_PLATFORM_NAME` | cpu | JAX platform (cpu/gpu) |

## Docker Commands

### Build Image

```bash
docker build -t dias:2.0.0 .
```

### Run Container

```bash
docker run -p 8000:8000 --name dias-api dias:2.0.0
```

### Execute Commands in Container

```bash
# Run bash
docker exec -it dias-api /bin/bash

# Run tests
docker exec dias-api pytest tests/

# Check logs
docker logs dias-api
```

### Clean Up

```bash
# Stop and remove containers
docker-compose down

# Remove volumes
docker-compose down -v

# Remove images
docker rmi dias:2.0.0
```

## Health Checks

The container includes health checks that run every 30 seconds:

```bash
# Check container health
docker ps
# Look for "(healthy)" status

# Manual health check
curl http://localhost:8000/health
```

## Volumes

### Data Persistence

Data and logs are stored in Docker volumes:

- `dias-data` - Application data
- `dias-logs` - Application logs

### Access Volume Data

```bash
# Inspect volumes
docker volume ls
docker volume inspect dias-data

# Backup data
docker run --rm -v dias-data:/data -v $(pwd):/backup \
  ubuntu tar czf /backup/dias-data-backup.tar.gz /data
```

## Resource Limits

Default resource limits (configurable in docker-compose.yml):

- **CPU:** 2 cores (limit), 1 core (reservation)
- **Memory:** 4GB (limit), 2GB (reservation)

## Troubleshooting

### Container Won't Start

```bash
# Check logs
docker-compose logs dias-api

# Check if port is in use
lsof -i :8000
```

### Permission Issues

```bash
# Verify user
docker exec dias-api whoami
# Should output: dias

# Check file permissions
docker exec dias-api ls -la /app
```

### Health Check Failing

```bash
# Test health endpoint
curl http://localhost:8000/health

# Check inside container
docker exec dias-api curl http://localhost:8000/health
```

### Memory Issues

```bash
# Check resource usage
docker stats dias-api

# Increase memory limit in docker-compose.yml
```

## Multi-Stage Build

The Dockerfile uses a multi-stage build:

1. **Builder Stage:** Compiles dependencies
2. **Runtime Stage:** Minimal image with only runtime dependencies

This results in a smaller, more secure production image.

## Security

- Runs as non-root user (`dias`)
- Minimal base image (python:3.9-slim)
- No unnecessary packages
- Health checks enabled
- Resource limits enforced

## Development

### Hot Reload

For development with code hot reload:

```yaml
# Uncomment in docker-compose.yml
volumes:
  - ./src:/app/src
```

Then:

```bash
APP_ENV=development docker-compose up
```

### Running Tests

```bash
# Run all tests
docker exec dias-api pytest

# Run specific test file
docker exec dias-api pytest tests/unit/test_jax_ops.py

# Run with coverage
docker exec dias-api pytest --cov=src tests/
```

## Production Deployment

### Build for Production

```bash
docker build -t dias:2.0.0 -t dias:latest .
```

### Push to Registry

```bash
# Tag for registry
docker tag dias:2.0.0 your-registry/dias:2.0.0

# Push
docker push your-registry/dias:2.0.0
```

### Deploy

```bash
# Using docker-compose
docker-compose -f docker-compose.prod.yml up -d

# Or using Docker Swarm/Kubernetes
# (deployment configs would go here)
```

## Monitoring

### Container Stats

```bash
docker stats dias-api
```

### Logs

```bash
# Follow logs
docker-compose logs -f

# Last 100 lines
docker-compose logs --tail=100

# Specific service
docker-compose logs -f dias-api
```

## Next Steps

After Docker deployment:
1. Test all API endpoints
2. Run integration tests
3. Set up monitoring (Prometheus/Grafana)
4. Configure reverse proxy (Nginx)
5. Set up SSL/TLS
6. Implement backup strategy

