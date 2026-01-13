# Contributing to DIAS

Thank you for considering contributing to the Disaster Impact Analysis System!

## Development Workflow

### Branching Strategy

We use a feature branch workflow:

1. **Main Branch** (`master`): Production-ready code
2. **Feature Branches**: `feature/ticket-{number}-{description}`
3. **Bug Fix Branches**: `bugfix/{issue-number}-{description}`
4. **Hotfix Branches**: `hotfix/{issue-number}-{description}`

### Creating a Feature Branch

```bash
# Update master
git checkout master
git pull origin master

# Create feature branch
git checkout -b feature/ticket-123-add-new-feature

# Work on your changes
# ... make changes ...

# Commit with conventional commit format
git add .
git commit -m "feat(TICKET-123): Add new feature

- Detailed description of change
- Additional context
- Related information"

# Push to remote
git push -u origin feature/ticket-123-add-new-feature
```

### Commit Message Format

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(TICKET-ID): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Example:**
```
feat(TICKET-5): Add JAX optimization engine

- Replaced Numba JIT with JAX JIT compilation
- Improved performance by 25%
- Updated all numeric operations to use jax.numpy

Closes #123
```

### Pull Request Process

1. **Create PR**: From your feature branch to `master`
2. **Title Format**: `feat(TICKET-ID): Brief description`
3. **Description**: 
   - What changed
   - Why it changed
   - How to test
   - Related tickets/issues
4. **Review**: Wait for code review and approval
5. **Tests**: Ensure all tests pass
6. **Merge**: Squash and merge after approval

### Code Review Checklist

Before submitting a PR, ensure:

- [ ] All tests pass (`pytest`)
- [ ] Code follows style guidelines (`black`, `flake8`, `mypy`)
- [ ] New code has tests (80%+ coverage)
- [ ] Documentation updated if needed
- [ ] Commit messages follow conventional format
- [ ] No unnecessary files committed
- [ ] `.gitignore` is up to date

## Development Setup

See [Development Guide](docs/DEVELOPMENT.md) for detailed setup instructions.

### Quick Start

```bash
# 1. Clone and branch
git clone https://github.com/dr-jgsmith/Disaster-Impact-Analysis-System
cd Disaster-Impact-Analysis-System
git checkout -b feature/my-new-feature

# 2. Set up environment
python3.9 -m venv venv
source venv/bin/activate
pip install -r requirements/dev.txt

# 3. Run tests
pytest

# 4. Make changes and test
# ... your changes ...
pytest
bash scripts/lint.sh

# 5. Commit and push
git add .
git commit -m "feat(TICKET-X): Your change"
git push -u origin feature/my-new-feature
```

## Code Style

### Python Style Guidelines

- **Python Version**: 3.9+
- **Formatter**: Black (line length 88)
- **Import Sorter**: isort (Black-compatible)
- **Linter**: flake8
- **Type Checker**: mypy

### Running Code Quality Tools

```bash
# Format code
bash scripts/format.sh

# Check linting
bash scripts/lint.sh

# Check types
mypy src/
```

### File Organization

- All files under 500 lines
- Clear separation of concerns
- Descriptive function and variable names
- Comprehensive docstrings

### Docstring Format

Use Google-style docstrings:

```python
def calculate_distance(lat1: float, lon1: float, 
                      lat2: float, lon2: float) -> float:
    """Calculate geodesic distance between two coordinates.
    
    Args:
        lat1: Latitude of first point in decimal degrees
        lon1: Longitude of first point in decimal degrees
        lat2: Latitude of second point in decimal degrees
        lon2: Longitude of second point in decimal degrees
    
    Returns:
        Distance in meters
    
    Raises:
        ValueError: If coordinates are invalid
    
    Example:
        >>> dist = calculate_distance(29.7604, -95.3698, 29.7605, -95.3699)
        >>> print(f"Distance: {dist:.2f} meters")
    """
    # Implementation
    pass
```

## Testing

### Writing Tests

- **Unit Tests**: Test individual functions/classes
- **Integration Tests**: Test API endpoints and workflows
- **Fixtures**: Use pytest fixtures for common test data

### Test Organization

```
tests/
├── unit/
│   ├── test_jax_ops.py
│   ├── test_model.py
│   └── test_utils.py
├── integration/
│   ├── test_api.py
│   └── test_workflows.py
└── fixtures/
    ├── sample_data.py
    └── data/
```

### Test Naming

- Test files: `test_<module_name>.py`
- Test functions: `test_<function_name>_<scenario>()`
- Test classes: `Test<ClassName>`

### Running Tests

```bash
# All tests
pytest

# Specific test file
pytest tests/unit/test_model.py

# Specific test
pytest tests/unit/test_model.py::test_build_connectivity_matrix

# With coverage
pytest --cov=src --cov-report=html

# Watch mode (requires pytest-watch)
ptw
```

## Documentation

### When to Update Documentation

- Adding new features
- Changing API endpoints
- Modifying configuration options
- Fixing bugs that affect usage
- Improving performance significantly

### Documentation Structure

- `README.md`: Overview and quick start
- `docs/API_SPECIFICATION.md`: API reference
- `docs/DEVELOPMENT.md`: Development setup
- `docs/DEPLOYMENT.md`: Deployment guide
- `docs/TESTING.md`: Testing guide
- Inline docstrings: All functions and classes

## Reporting Issues

### Bug Reports

When reporting bugs, include:

1. **Description**: Clear description of the issue
2. **Steps to Reproduce**: Numbered steps to reproduce
3. **Expected Behavior**: What should happen
4. **Actual Behavior**: What actually happens
5. **Environment**: Python version, OS, Docker version
6. **Logs**: Relevant log output
7. **Screenshots**: If applicable

### Feature Requests

When requesting features, include:

1. **Use Case**: Why is this feature needed?
2. **Proposed Solution**: How should it work?
3. **Alternatives**: Other approaches considered
4. **Examples**: Examples of similar features elsewhere

## Questions?

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and discussions
- **Email**: justingriffis@wsu.edu

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

Thank you for contributing to DIAS! 🌊

