#!/bin/bash
# Run linters and type checkers on DIAS codebase

set -e

echo "====================================="
echo "  DIAS Code Quality Checks"
echo "====================================="
echo

# Get the script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Track if any checks fail
FAILED=0

echo "🔍 Running Flake8 linter..."
if flake8 src/ tests/; then
    echo "✅ Flake8 passed"
else
    echo "❌ Flake8 failed"
    FAILED=1
fi

echo
echo "🔍 Running Black format checker..."
if black src/ tests/ --check --diff; then
    echo "✅ Black format check passed"
else
    echo "❌ Black format check failed - run ./scripts/format.sh to fix"
    FAILED=1
fi

echo
echo "🔍 Running isort import checker..."
if isort src/ tests/ --check-only --profile black; then
    echo "✅ isort check passed"
else
    echo "❌ isort check failed - run ./scripts/format.sh to fix"
    FAILED=1
fi

echo
echo "🔍 Running MyPy type checker..."
if mypy src/; then
    echo "✅ MyPy passed"
else
    echo "❌ MyPy failed"
    FAILED=1
fi

echo
echo "====================================="
if [ $FAILED -eq 0 ]; then
    echo "✅ All code quality checks passed!"
    exit 0
else
    echo "❌ Some checks failed. Please fix the issues above."
    exit 1
fi

