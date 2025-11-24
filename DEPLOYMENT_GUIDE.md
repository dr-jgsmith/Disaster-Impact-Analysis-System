# DIAS Deployment Guide

Complete guide for deploying the Disaster Impact Analysis System in various environments.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Development Deployment](#development-deployment)
3. [Production Deployment](#production-deployment)
4. [Verification & Testing](#verification--testing)
5. [Monitoring & Maintenance](#monitoring--maintenance)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Software

- **Docker:** 20.10+ ([Install Docker](https://docs.docker.com/get-docker/))
- **Docker Compose:** 1.29+ (included with Docker Desktop)
- **Git:** For cloning the repository

### System Requirements

**Minimum:**
- CPU: 2 cores
- RAM: 4GB
- Disk: 10GB free space

**Recommended:**
- CPU: 4+ cores  
- RAM: 8GB+
- Disk: 20GB+ free space
- SSD for better performance

---

## Development Deployment

### 1. Clone Repository

```bash
git clone https://github.com/dr-jgsmith/Disaster-Impact-Analysis-System.git
cd Disaster-Impact-Analysis-System
```

### 2. Configure Environment

```bash
# Copy example environment file
cp docker/env.example .env

# Edit configuration (optional)
# nano .env
```

**Development Settings:**
```env
APP_ENV=development
LOG_LEVEL=DEBUG
API_WORKERS=1
```

### 3. Build and Run

```bash
# Build and start in foreground
docker-compose up --build

# Or run in background
docker-compose up --build -d
```

### 4. Verify Deployment

```bash
# Check health
curl http://localhost:8000/health

# Expected response:
# {
#   "status": "healthy",
#   "timestamp": "2024-11-21T...",
#   "service": "DIAS API",
#   "version": "2.0.0"
# }
```

### 5. Access Documentation

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
- **API Root:** http://localhost:8000/

### 6. Hot Reload (Development)

For live code reloading during development:

**docker-compose.yml (uncomment):**
```yaml
volumes:
  - ./src:/app/src  # Enable hot reload
```

Then restart:
```bash
docker-compose restart dias-api
```

---

## Production Deployment

### 1. Prepare Environment

```bash
# Production environment file
cp docker/env.example .env
```

**Production Settings:**
```env
APP_ENV=production
LOG_LEVEL=INFO
API_WORKERS=4

# Resource limits (adjust based on your needs)
# Configured in docker-compose.yml
```

### 2. Build Production Image

```bash
docker-compose build --no-cache
```

### 3. Start Services

```bash
# Start in detached mode
docker-compose up -d

# Check status
docker-compose ps
```

### 4. Verify Health

```bash
# Container health
docker ps
# Look for "(healthy)" status

# API health
curl http://localhost:8000/health
curl http://localhost:8000/info
```

### 5. View Logs

```bash
# Follow logs
docker-compose logs -f

# Specific service
docker-compose logs -f dias-api

# Last 100 lines
docker-compose logs --tail=100 dias-api
```

---

## Verification & Testing

### API Endpoint Tests

#### 1. Health Check

```bash
curl http://localhost:8000/health
```

#### 2. API Info

```bash
curl http://localhost:8000/info
```

#### 3. Create Flood Phenomenon

```bash
curl -X POST http://localhost:8000/api/v1/events \
  -H "Content-Type: application/json" \
  -d '{
    "event_type": "flood",
    "data": {
      "entity_ids": ["P001", "P002", "P003"],
      "coordinates": [[29.76, -95.37], [29.77, -95.38], [29.78, -95.39]],
      "adjacency_matrix": [[1,1,0], [1,1,1], [0,1,1]],
      "attributes": {
        "elevations": [5.0, 10.0, 8.0],
        "land_values": [100000, 150000, 120000],
        "building_values": [200000, 250000, 220000]
      }
    }
  }'
```

**Save the returned `id` for next steps.**

#### 4. Compute Zones

```bash
PHENOM_ID="flood_abc12345"  # Use ID from step 3

curl -X POST http://localhost:8000/api/v1/events/$PHENOM_ID/zones \
  -H "Content-Type: application/json" \
  -d '{
    "scenario_params": {
      "min_water_level": 3.0,
      "max_water_level": 14.0
    }
  }'
```

#### 5. Get GeoJSON

```bash
curl http://localhost:8000/api/v1/events/$PHENOM_ID/geojson
```

### Integration Tests

```bash
# Run all tests in container
docker exec dias-api pytest

# Run with coverage
docker exec dias-api pytest --cov=src tests/

# Specific test file
docker exec dias-api pytest tests/integration/test_api.py -v
```

### Performance Test

```bash
# Using Apache Bench
ab -n 100 -c 10 http://localhost:8000/health

# Using wrk
wrk -t4 -c100 -d30s http://localhost:8000/health
```

---

## Monitoring & Maintenance

### Resource Monitoring

```bash
# Container stats
docker stats dias-api

# System resources
docker system df

# Volume usage
docker volume ls
```

### Log Management

```bash
# Current logs
docker-compose logs -f dias-api

# Rotate logs (if needed)
docker-compose logs --tail=1000 dias-api > logs/api.log
```

### Backup & Restore

#### Backup Data

```bash
# Backup data volume
docker run --rm \
  -v dias-data:/data \
  -v $(pwd)/backups:/backup \
  ubuntu tar czf /backup/dias-data-$(date +%Y%m%d).tar.gz /data

# Backup logs volume
docker run --rm \
  -v dias-logs:/logs \
  -v $(pwd)/backups:/backup \
  ubuntu tar czf /backup/dias-logs-$(date +%Y%m%d).tar.gz /logs
```

#### Restore Data

```bash
# Restore data volume
docker run --rm \
  -v dias-data:/data \
  -v $(pwd)/backups:/backup \
  ubuntu tar xzf /backup/dias-data-20241121.tar.gz -C /

# Restart services
docker-compose restart
```

### Updates & Upgrades

#### Update Code

```bash
# Pull latest code
git pull origin main

# Rebuild and restart
docker-compose up --build -d

# Verify
curl http://localhost:8000/health
```

#### Update Dependencies

```bash
# Update requirements files
# Edit requirements/base.txt or requirements/prod.txt

# Rebuild image
docker-compose build --no-cache

# Restart
docker-compose up -d
```

---

## Troubleshooting

### Container Won't Start

**Check logs:**
```bash
docker-compose logs dias-api
```

**Common issues:**
- Port 8000 already in use
- Insufficient memory
- Permission issues

**Solutions:**
```bash
# Check port usage
lsof -i :8000

# Increase memory in docker-compose.yml
# Check Docker Desktop resources
```

### Health Check Failing

**Test manually:**
```bash
# From host
curl -v http://localhost:8000/health

# From container
docker exec dias-api curl http://localhost:8000/health
```

**Check:**
- Application started correctly
- Port mapping correct
- No errors in logs

### API Errors

**500 Internal Server Error:**
```bash
# Check logs for stack trace
docker-compose logs dias-api | grep ERROR

# Check dependencies
docker exec dias-api pip list
```

**404 Not Found:**
- Verify endpoint URL
- Check API documentation: http://localhost:8000/docs

### Performance Issues

**Slow response:**
```bash
# Check resource usage
docker stats dias-api

# Check JAX configuration
docker exec dias-api env | grep JAX

# Increase workers
# Edit .env: API_WORKERS=8
docker-compose restart
```

### Out of Memory

```bash
# Check memory usage
docker stats

# Increase limit in docker-compose.yml:
# memory: 8G

docker-compose up -d
```

### Permission Errors

```bash
# Check user in container
docker exec dias-api whoami
# Should be: dias

# Check file permissions
docker exec dias-api ls -la /app

# Fix if needed (rebuild)
docker-compose build --no-cache
```

---

## Production Best Practices

### 1. Reverse Proxy

Use Nginx or Traefik in front of DIAS:

**nginx.conf example:**
```nginx
server {
    listen 80;
    server_name dias.example.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 2. SSL/TLS

```bash
# Using Let's Encrypt
certbot --nginx -d dias.example.com
```

### 3. Monitoring

**Prometheus + Grafana:**
- Add metrics endpoint to API
- Configure Prometheus scraping
- Create Grafana dashboards

### 4. Logging

**Centralized logging:**
- ELK Stack (Elasticsearch, Logstash, Kibana)
- Loki + Grafana
- Splunk

**docker-compose.yml logging:**
```yaml
logging:
  driver: "json-file"
  options:
    max-size: "10m"
    max-file: "3"
```

### 5. Backup Strategy

**Automated backups:**
```bash
# Cron job (daily at 2 AM)
0 2 * * * /path/to/backup-script.sh
```

**Retention policy:**
- Daily: Keep 7 days
- Weekly: Keep 4 weeks
- Monthly: Keep 12 months

### 6. Security

- Run as non-root user ✅ (already configured)
- Limit container resources ✅ (already configured)
- Network segmentation
- Regular security updates
- API authentication (future)
- Rate limiting (future)

---

## Environment-Specific Configurations

### Development

```yaml
# docker-compose.override.yml
version: '3.8'
services:
  dias-api:
    environment:
      - APP_ENV=development
      - LOG_LEVEL=DEBUG
    volumes:
      - ./src:/app/src  # Hot reload
```

### Staging

```yaml
# docker-compose.staging.yml
version: '3.8'
services:
  dias-api:
    environment:
      - APP_ENV=staging
      - LOG_LEVEL=INFO
    deploy:
      resources:
        limits:
          cpus: '1.5'
          memory: 3G
```

### Production

```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  dias-api:
    environment:
      - APP_ENV=production
      - LOG_LEVEL=WARNING
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
```

---

## Quick Reference

### Common Commands

```bash
# Start
docker-compose up -d

# Stop
docker-compose down

# Restart
docker-compose restart

# Logs
docker-compose logs -f

# Status
docker-compose ps

# Execute command
docker exec dias-api <command>

# Shell access
docker exec -it dias-api /bin/bash
```

### Useful Endpoints

- Health: `GET /health`
- Info: `GET /info`
- Docs: `GET /docs`
- Create event: `POST /api/v1/events`
- List events: `GET /api/v1/events`
- GeoJSON: `GET /api/v1/events/{id}/geojson`

---

## Support

For issues or questions:
1. Check logs: `docker-compose logs`
2. Review documentation: `/docs`
3. Open GitHub issue
4. Contact: dias-support@example.com

---

**Last Updated:** November 21, 2024  
**Version:** 2.0.0  
**Docker Image:** dias:2.0.0

