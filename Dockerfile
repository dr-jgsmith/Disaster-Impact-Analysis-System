# Dockerfile for DIAS - Disaster Impact Analysis System
# Multi-stage build for optimized production image

# ============================================================================
# Stage 1: Builder
# ============================================================================
FROM python:3.9-slim as builder

LABEL maintainer="DIAS Team"
LABEL description="Disaster Impact Analysis System - Builder Stage"

# Set working directory
WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files
COPY requirements/base.txt requirements/base.txt
COPY requirements/prod.txt requirements/prod.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements/prod.txt


# ============================================================================
# Stage 2: Runtime
# ============================================================================
FROM python:3.9-slim

LABEL maintainer="DIAS Team"
LABEL description="Disaster Impact Analysis System - Production"
LABEL version="2.0.0"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Create non-root user
RUN groupadd -r dias && useradd -r -g dias dias

# Copy application code
COPY --chown=dias:dias src/ /app/src/
COPY --chown=dias:dias tests/ /app/tests/
COPY --chown=dias:dias scripts/ /app/scripts/
COPY --chown=dias:dias pyproject.toml /app/

# Create necessary directories
RUN mkdir -p /app/data /app/logs && \
    chown -R dias:dias /app/data /app/logs

# Switch to non-root user
USER dias

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

