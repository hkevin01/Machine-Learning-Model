# Contributing

We love your input! We want to make contributing to this project as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## Development Process

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

### Pull Requests

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Issue that pull request!

### Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/machine-learning-model.git
cd "Machine Learning Model"

# Create virtual environment (if not exists)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
make install-dev

# Install pre-commit hooks
pre-commit install

# Run tests
make test

# Check code quality
make check
```

### Code Style

We use several tools to maintain code quality:

- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking
- **pytest** for testing
- **ruff** for fast linting

Run all checks with:
```bash
make check
```

### Project-Specific Guidelines

#### Machine Learning Code
- Document all model architectures and hyperparameters
- Include data preprocessing steps
- Add model performance metrics in docstrings
- Use type hints for tensor shapes where applicable

#### Scripts
- All scripts should have proper argument parsing
- Include help text and examples
- Test scripts with edge cases
- Document script dependencies

### Commit Message Convention

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

Types:
- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation only changes
- `style`: Changes that do not affect the meaning of the code
- `refactor`: A code change that neither fixes a bug nor adds a feature
- `perf`: A code change that improves performance
- `test`: Adding missing tests or correcting existing tests
- `chore`: Changes to the build process or auxiliary tools

Examples:
```
feat(model): add transformer architecture
fix(scripts): resolve project creation path issue  
docs(readme): update installation instructions
```

### Testing

- Write tests for new features
- Ensure all tests pass
- Aim for high test coverage
- Include both unit and integration tests
- Test ML models with sample data
- Mock external dependencies

### Documentation

- Update docstrings for new functions/classes
- Update README.md if needed
- Add examples for new features
- Document model architectures
- Include performance benchmarks

### Working with Virtual Environments

Since this project uses virtual environments:

1. Always activate the virtual environment before working
2. Use `pip install -e .` for development installs
3. Update requirements files when adding dependencies
4. Test in clean environments before submitting

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.
