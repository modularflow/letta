# Workflow System Tests

This directory contains tests for the Letta workflow system, which includes file operations, memory management, and workflow coordination components.

## Setup

1. Install dependencies:
```bash
# Using poetry (recommended)
poetry install

# Using pip
pip install -r requirements.txt
```

2. Install development dependencies:
```bash
poetry install --with dev
```

## Running Tests

### Running all workflow tests:
```bash
# Using poetry
poetry run pytest tests/workflow/ -v

# Using pytest directly
pytest tests/workflow/ -v
```

### Running specific test files:
```bash
# Test file operations
pytest tests/workflow/test_file_ops.py -v

# Test memory management
pytest tests/workflow/test_memory.py -v

# Test workflow coordination
pytest tests/workflow/test_coordinator.py -v
```

### Running with coverage:
```bash
# Generate coverage report
poetry run pytest tests/workflow/ --cov=letta.services.workflow --cov-report=html

# View coverage report
open htmlcov/index.html
```

## Test Structure

The test suite is organized into three main components:

1. `test_file_ops.py`: Tests for file operations
   - File reading/writing
   - Version control
   - Directory management

2. `test_memory.py`: Tests for workflow memory
   - State management
   - Memory sharing
   - State persistence

3. `test_coordinator.py`: Tests for workflow coordination
   - Task management
   - Dependency resolution
   - Error handling
   - Parallel execution

## CI/CD Pipeline

The project uses GitHub Actions for continuous integration and deployment. The workflow is defined in `.github/workflows/workflow-tests.yml`.

The CI pipeline:
1. Runs tests on multiple Python versions (3.9, 3.10, 3.11)
2. Generates coverage reports
3. Uploads test results and coverage data
4. Fails if coverage drops below threshold

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

### Adding New Tests

1. Create a new test file in `tests/workflow/`
2. Use the existing test structure as a template
3. Include both positive and negative test cases
4. Add async tests where appropriate
5. Update the README if needed

### Test Guidelines

1. Use descriptive test names
2. Include docstrings explaining test purpose
3. Use fixtures for common setup
4. Clean up resources in teardown
5. Keep tests focused and atomic

## Common Issues

1. **Async Test Failures**
   - Ensure `pytest-asyncio` is installed
   - Use `@pytest.mark.asyncio` decorator
   - Run with `--asyncio-mode=auto`

2. **Resource Cleanup**
   - Use `tmp_path` fixture for temporary files
   - Clean up resources in `finally` blocks
   - Use context managers when possible

3. **Test Isolation**
   - Don't rely on external services
   - Mock external dependencies
   - Use fresh instances for each test

## Debugging Tests

1. Run with increased verbosity:
```bash
pytest -vv tests/workflow/
```

2. Enable debug logging:
```bash
pytest --log-cli-level=DEBUG tests/workflow/
```

3. Use pytest's debug features:
```bash
pytest --pdb tests/workflow/  # Drops into debugger on failure
```

## Performance Testing

1. Run with timing information:
```bash
pytest --durations=10 tests/workflow/
```

2. Profile tests:
```bash
pytest --profile tests/workflow/
```

## Code Quality

1. Run linting:
```bash
poetry run flake8 letta/services/workflow/
poetry run flake8 tests/workflow/
```

2. Run type checking:
```bash
poetry run mypy letta/services/workflow/
poetry run mypy tests/workflow/
```

## Support

For issues or questions:
1. Check existing issues on GitHub
2. Create a new issue with:
   - Test name and file
   - Expected vs actual behavior
   - Python version and environment details
   - Relevant logs or error messages 