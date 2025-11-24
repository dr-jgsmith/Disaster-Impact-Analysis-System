#!/bin/bash
# Startup script for DIAS API service

set -e

echo "========================================"
echo "DIAS - Disaster Impact Analysis System"
echo "Version: 2.0.0"
echo "========================================"

# Environment info
echo "Environment: ${APP_ENV:-production}"
echo "Log Level: ${LOG_LEVEL:-INFO}"
echo "API Host: ${API_HOST:-0.0.0.0}"
echo "API Port: ${API_PORT:-8000}"
echo "Workers: ${API_WORKERS:-4}"

# Create directories if they don't exist
mkdir -p /app/data
mkdir -p /app/logs

# Run database migrations (if needed in future)
# python -m alembic upgrade head

# Start the application
echo "Starting DIAS API..."

# Check if running in development mode
if [ "${APP_ENV}" = "development" ]; then
    echo "Running in DEVELOPMENT mode with hot reload"
    exec uvicorn src.api.main:app \
        --host "${API_HOST:-0.0.0.0}" \
        --port "${API_PORT:-8000}" \
        --reload \
        --log-level "${LOG_LEVEL:-info}"
else
    echo "Running in PRODUCTION mode"
    exec uvicorn src.api.main:app \
        --host "${API_HOST:-0.0.0.0}" \
        --port "${API_PORT:-8000}" \
        --workers "${API_WORKERS:-4}" \
        --log-level "${LOG_LEVEL:-info}"
fi

