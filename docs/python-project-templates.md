# Modern Python Project Templates

This document provides an overview of popular Python project template tools and how they compare to our custom project generator.

## Table of Contents

- [Cookiecutter PyPackage](#cookiecutter-pypackage)
- [PyScaffold](#pyscaffold)
- [Poetry](#poetry)
- [Copier](#copier)
- [Other Notable Templates](#other-notable-templates)
- [Comparison Matrix](#comparison-matrix)
- [Our Implementation](#our-implementation)

## Cookiecutter PyPackage

**Repository:** https://github.com/audreyfeldroy/cookiecutter-pypackage

Cookiecutter PyPackage is one of the most popular Python project templates, created by Audrey Roy Greenfeld. It uses the Cookiecutter templating engine to generate Python packages with modern best practices.

### Key Features

- **Interactive Setup**: Prompts for project details during generation
- **Multiple License Options**: MIT, BSD, Apache, GPL, etc.
- **Testing Framework**: Pre-configured pytest setup
- **Documentation**: Sphinx documentation with Read the Docs integration
- **CI/CD**: GitHub Actions, Travis CI, or Tox configurations
- **Package Publishing**: Ready for PyPI publishing
- **Code Quality Tools**: flake8, black formatting
- **Versioning**: Automatic version bumping with bump2version

### Example Usage

```bash
# Install cookiecutter
pip install cookiecutter

# Generate project
cookiecutter https://github.com/audreyfeldroy/cookiecutter-pypackage.git
```

### Generated Structure

```
my_package/
├── my_package/
│   ├── __init__.py
│   └── my_package.py
├── tests/
│   └── test_my_package.py
├── docs/
│   ├── conf.py
│   └── index.rst
├── .github/
│   └── workflows/
├── setup.py
├── setup.cfg
├── requirements_dev.txt
├── tox.ini
└── README.rst
```

### Pros

- ✅ Well-established and battle-tested
- ✅ Extensive customization options
- ✅ Great documentation
- ✅ Large community support
- ✅ Multiple CI/CD options

### Cons

- ❌ Uses older setup.py instead of pyproject.toml
- ❌ RST format instead of Markdown
- ❌ Can be overwhelming for beginners
- ❌ Requires interactive input

## PyScaffold

**Repository:** https://github.com/pyscaffold/pyscaffold

PyScaffold is a modern Python project generator that emphasizes simplicity and follows current Python packaging standards.

### Key Features

- **Modern Packaging**: Uses pyproject.toml by default
- **Namespace Packages**: Built-in support for namespace packages
- **Extensions System**: Pluggable architecture with extensions
- **Git Integration**: Automatic git repository initialization
- **Pre-commit Hooks**: Built-in pre-commit configuration
- **Jupyter Support**: Optional Jupyter notebook integration
- **Django Support**: Django project templates
- **Documentation**: Sphinx with modern themes

### Example Usage

```bash
# Install PyScaffold
pip install pyscaffold

# Generate basic project
putup my_project

# Generate with extensions
putup my_project --interactive --pre-commit --github-actions
```

### Generated Structure

```
my_project/
├── src/
│   └── my_project/
│       ├── __init__.py
│       └── skeleton.py
├── tests/
│   ├── conftest.py
│   └── test_skeleton.py
├── docs/
│   ├── conf.py
│   └── index.rst
├── .github/
│   └── workflows/
├── pyproject.toml
├── setup.cfg
├── tox.ini
└── README.rst
```

### Extensions

- `--cirrus`: Cirrus CI support
- `--gitlab`: GitLab CI support  
- `--github-actions`: GitHub Actions
- `--pre-commit`: Pre-commit hooks
- `--cookiecutter`: Cookiecutter integration
- `--django`: Django project structure
- `--jupyter`: Jupyter notebook support

### Pros

- ✅ Modern pyproject.toml configuration
- ✅ Src-layout by default
- ✅ Extensible architecture
- ✅ Git integration
- ✅ Active development

### Cons

- ❌ Less customization than Cookiecutter
- ❌ Smaller community
- ❌ RST documentation format
- ❌ Can be complex for simple projects

## Poetry

**Repository:** https://github.com/python-poetry/poetry

Poetry is both a dependency management tool and project template generator that simplifies Python packaging.

### Key Features

- **Dependency Management**: Lock files for reproducible builds
- **Virtual Environment**: Automatic virtual environment management
- **Publishing**: Easy PyPI publishing
- **Modern Configuration**: pyproject.toml-based
- **Version Management**: Semantic versioning support
- **Build System**: Modern Python packaging

### Example Usage

```bash
# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Create new project
poetry new my-package

# Initialize existing project
poetry init
```

### Generated Structure

```
my-package/
├── my_package/
│   └── __init__.py
├── tests/
│   └── test_my_package.py
├── pyproject.toml
└── README.md
```

### Pros

- ✅ Excellent dependency management
- ✅ Virtual environment integration
- ✅ Modern pyproject.toml
- ✅ Easy publishing workflow
- ✅ Active community

### Cons

- ❌ Minimal project structure
- ❌ Learning curve for Poetry-specific commands
- ❌ No CI/CD templates
- ❌ Limited customization

## Copier

**Repository:** https://github.com/copier-org/copier

Copier is a modern alternative to Cookiecutter with additional features like template updates.

### Key Features

- **Template Updates**: Update projects when templates change
- **Multiple Formats**: YAML, JSON, TOML configuration
- **Jinja2 Templating**: Powerful templating engine
- **Git Integration**: Built-in git support
- **Migrations**: Template migration system
- **Subdirectories**: Generate into existing projects

### Example Usage

```bash
# Install Copier
pip install copier

# Generate project
copier https://github.com/pawamoy/copier-poetry.git my_project

# Update existing project
copier update
```

### Popular Copier Templates

- **copier-poetry**: Modern Python with Poetry
- **copier-pdm**: Python with PDM package manager
- **copier-django**: Django projects
- **copier-fastapi**: FastAPI applications

### Pros

- ✅ Template update capability
- ✅ Modern tooling
- ✅ Flexible configuration
- ✅ Good documentation
- ✅ Active development

### Cons

- ❌ Smaller ecosystem than Cookiecutter
- ❌ Less mature
- ❌ Fewer available templates

## Other Notable Templates

### 1. **Hypermodern Python Cookiecutter**
- **Repository**: https://github.com/cjolowicz/cookiecutter-hypermodern-python
- **Focus**: Cutting-edge Python tooling (Poetry, Nox, pre-commit, GitHub Actions)
- **Features**: Type checking, documentation, testing, linting

### 2. **FastAPI Template**
- **Repository**: https://github.com/tiangolo/full-stack-fastapi-postgresql
- **Focus**: Full-stack FastAPI applications
- **Features**: PostgreSQL, Docker, Celery, Vue.js frontend

### 3. **Django Cookiecutter**
- **Repository**: https://github.com/cookiecutter/cookiecutter-django
- **Focus**: Django web applications
- **Features**: Docker, PostgreSQL, Redis, Celery, AWS deployment

### 4. **Data Science Template**
- **Repository**: https://github.com/drivendata/cookiecutter-data-science
- **Focus**: Data science projects
- **Features**: Jupyter notebooks, data organization, reproducible research

### 5. **Python CLI Template**
- **Repository**: https://github.com/NiklasRosenstein/python-cli-template
- **Focus**: Command-line applications
- **Features**: Click, setuptools, testing

## Comparison Matrix

| Feature | Cookiecutter PyPackage | PyScaffold | Poetry | Copier | Our Template |
|---------|----------------------|------------|--------|--------|--------------|
| **Configuration** | setup.py | pyproject.toml | pyproject.toml | pyproject.toml | pyproject.toml |
| **Documentation** | RST/Sphinx | RST/Sphinx | Minimal | Configurable | Markdown/Sphinx |
| **Testing** | pytest | pytest | pytest | Configurable | pytest + coverage |
| **CI/CD** | Multiple options | GitHub Actions | None | Configurable | GitHub Actions |
| **Code Quality** | flake8 | flake8 | None | Configurable | black, ruff, mypy |
| **Dependencies** | requirements.txt | setup.cfg | Poetry lock | Configurable | requirements.txt |
| **Virtual Env** | Manual | Manual | Automatic | Manual | Automatic |
| **CLI Support** | Basic | Basic | None | Configurable | Typer/Click |
| **Docker** | Optional | Extension | None | Template-specific | Included |
| **Pre-commit** | Optional | Extension | None | Configurable | Included |
| **Logging** | Basic | Basic | None | Configurable | Loguru |
| **Updates** | No | Limited | No | Yes | No |

## Our Implementation

Our custom Python project generator combines the best features from these tools while addressing their limitations:

### Unique Features

1. **Zero Configuration**: Works without interactive prompts
2. **Modern Tooling**: Uses latest Python packaging standards
3. **Rich CLI**: Built-in typer-based command-line interface
4. **Comprehensive Linting**: ruff, black, mypy, flake8
5. **Auto Virtual Environment**: Creates and configures venv automatically
6. **Container Ready**: Docker and docker-compose included
7. **Documentation**: Both Markdown and Sphinx support
8. **Security**: bandit and safety checks included
9. **AI Integration**: GitHub Copilot configuration
10. **Logging**: Advanced logging with loguru

### Inspired By

- **Cookiecutter PyPackage**: Comprehensive feature set
- **PyScaffold**: Modern packaging standards
- **Hypermodern Python**: Cutting-edge tooling
- **Poetry**: Excellent dependency management patterns
- **FastAPI Templates**: Modern development practices

### Design Philosophy

1. **Convention over Configuration**: Sensible defaults
2. **Modern Standards**: Latest Python packaging practices
3. **Developer Experience**: Easy setup and usage
4. **Comprehensive**: Everything needed for production
5. **Extensible**: Easy to modify and extend

## Choosing the Right Template

### Use Cookiecutter PyPackage If:
- You need maximum customization
- You're publishing to PyPI
- You want battle-tested solutions
- You prefer interactive setup

### Use PyScaffold If:
- You want modern packaging standards
- You need namespace packages
- You prefer src-layout
- You want extensibility

### Use Poetry If:
- Dependency management is crucial
- You want simple project structure
- You're comfortable with Poetry workflow
- You need reproducible builds

### Use Our Template If:
- You want zero-configuration setup
- You need comprehensive tooling out-of-the-box
- You prefer modern CLI interfaces
- You want container and AI integration
- You're working on machine learning projects

## Directory Structure Behavior

Our template intelligently handles directory creation:

### Default Behavior
```bash
# If current directory name matches project name, uses current directory
cd /path/to/my-project
./create_python_project.py  # Uses current directory as project root

# If project name differs, creates subdirectory
cd /path/to/workspace
./create_python_project.py my-new-project  # Creates /path/to/workspace/my-new-project/
```

### Force Subdirectory Creation
```bash
# Always create subdirectory, even if names match
./create_python_project.py my-project --force-subdir
```

### Examples

**Scenario 1: Initialize existing directory**
```bash
mkdir my-awesome-project
cd my-awesome-project
/path/to/create_python_project.py
# Result: Uses current directory as project root
```

**Scenario 2: Create new project**
```bash
cd workspace
/path/to/create_python_project.py my-awesome-project
# Result: Creates workspace/my-awesome-project/
```

**Scenario 3: Force subdirectory**
```bash
cd my-awesome-project
/path/to/create_python_project.py my-awesome-project --force-subdir
# Result: Creates my-awesome-project/my-awesome-project/
```

## Best Practices

Regardless of which template you choose, follow these best practices:

1. **Use pyproject.toml** for modern Python packaging
2. **Implement comprehensive testing** with pytest
3. **Set up CI/CD pipelines** for automation
4. **Use code quality tools** (linting, formatting, type checking)
5. **Include proper documentation** with examples
6. **Set up pre-commit hooks** for code quality
7. **Use semantic versioning** for releases
8. **Include security scanning** in your workflow
9. **Containerize your application** for deployment
10. **Monitor dependencies** for security vulnerabilities

## References

- [Python Packaging User Guide](https://packaging.python.org/)
- [PEP 518 - pyproject.toml](https://peps.python.org/pep-0518/)
- [Python Application Layouts](https://realpython.com/python-application-layouts/)
- [Hypermodern Python](https://cjolowicz.github.io/posts/hypermodern-python-01-setup/)
- [Python Project Template Best Practices](https://mitelman.engineering/posts/python-best-practice/)
