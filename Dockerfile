# Use Python 3.12 slim image as base
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libxml2-dev \
    libxslt-dev \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY setup.py pyproject.toml ./
COPY dias/ ./dias/

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -e . && \
    pip install -e ".[dev]"

# Copy the rest of the application
COPY . .

# Create a non-root user
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Expose port for Jupyter notebook (if needed)
EXPOSE 8888

# Default command
CMD ["python", "-c", "import dias; print('DIAS container is ready!')"]