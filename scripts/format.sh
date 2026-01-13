#!/bin/bash
# Format Python code using Black and isort

set -e

echo "====================================="
echo "  DIAS Code Formatting"
echo "====================================="
echo

# Get the script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "📝 Running Black formatter..."
black src/ tests/ --line-length 88 --target-version py39

echo
echo "📝 Running isort import sorter..."
isort src/ tests/ --profile black

echo
echo "✅ Code formatting complete!"
echo "====================================="

