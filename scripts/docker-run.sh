#!/bin/bash
# Build and run DIAS service in Docker

set -e

echo "====================================="
echo "  DIAS Docker Deployment"
echo "====================================="
echo

# Get the script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT/docker"

# Parse command
COMMAND="${1:-up}"

case "$COMMAND" in
    up)
        echo "🐳 Starting DIAS service..."
        docker-compose up --build
        ;;
    down)
        echo "🛑 Stopping DIAS service..."
        docker-compose down
        ;;
    restart)
        echo "🔄 Restarting DIAS service..."
        docker-compose down
        docker-compose up --build -d
        echo "✅ Service restarted"
        ;;
    logs)
        echo "📋 Showing logs..."
        docker-compose logs -f
        ;;
    build)
        echo "🔨 Building Docker image..."
        docker-compose build --no-cache
        ;;
    *)
        echo "❌ Unknown command: $COMMAND"
        echo "Usage: $0 [up|down|restart|logs|build]"
        exit 1
        ;;
esac

