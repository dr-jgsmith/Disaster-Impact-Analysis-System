#!/bin/bash
# Run DIAS test suite

set -e

echo "====================================="
echo "  DIAS Test Suite"
echo "====================================="
echo

# Get the script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Parse arguments
TEST_PATH="${1:-tests}"
COVERAGE="${2:-true}"

echo "📋 Test Configuration:"
echo "  Path: $TEST_PATH"
echo "  Coverage: $COVERAGE"
echo

if [ "$COVERAGE" = "true" ]; then
    echo "🧪 Running tests with coverage..."
    pytest "$TEST_PATH" \
        --cov=src \
        --cov-report=html \
        --cov-report=term-missing \
        --cov-report=xml \
        -v
    
    echo
    echo "📊 Coverage report generated at: htmlcov/index.html"
else
    echo "🧪 Running tests without coverage..."
    pytest "$TEST_PATH" -v
fi

echo
echo "====================================="
echo "✅ Test suite complete!"

