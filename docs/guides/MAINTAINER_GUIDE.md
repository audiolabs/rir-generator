# Quick Start Guide for Maintainers

This guide provides quick reference commands for common maintenance tasks after the package restructuring.

## 🚀 Quick Commands

### Development Setup
```bash
# Clone and setup
git clone https://github.com/audiolabs/rir-generator.git
cd rir-generator
pip install -e .[dev]

# Run tests
pytest

# Check code quality
black --check src/ tests/
```

### Building
```bash
# Build package
python -m build

# Check package
twine check dist/*

# Clean build artifacts
rm -rf build/ dist/ *.egg-info src/*.egg-info
```

### Publishing a Release

#### 1. Version Management (Automatic!)
The project uses **setuptools-scm** for automatic versioning from git tags.
**No need to manually update version numbers!**

See [VERSIONING.md](VERSIONING.md) for details.

#### 2. Create Release on GitHub
```bash
# Tag the release (version is determined from this tag)
git tag v0.3.0
git push origin v0.3.0

# Or use GitHub web interface:
# https://github.com/audiolabs/rir-generator/releases/new
```

#### 3. Automatic Publishing
The GitHub Actions workflow automatically:
- ✅ Builds wheels for all platforms
- ✅ Publishes to PyPI
- ✅ No manual intervention needed!

### Testing Before Release

#### Test on Test PyPI
1. Go to: https://github.com/audiolabs/rir-generator/actions/workflows/publish-to-pypi.yml
2. Click "Run workflow"
3. Select: ☑ Publish to Test PyPI instead
4. Click "Run workflow"
5. Test install:
   ```bash
   pip install --index-url https://test.pypi.org/simple/ \
               --extra-index-url https://pypi.org/simple/ \
               rir-generator
   ```

## 📁 Important Files

### Configuration
- `pyproject.toml` - Package metadata and build config
- `.github/workflows/python-package.yml` - CI testing
- `.github/workflows/publish-to-pypi.yml` - PyPI publishing
- `.github/dependabot.yml` - Dependency updates

### Documentation
- `README.md` - Main documentation
- `docs/guides/VERSIONING.md` - Dynamic versioning with setuptools-scm
- `docs/guides/MIGRATION.md` - Migration guide for package restructuring
- `docs/guides/CI_CD_SETUP.md` - Detailed CI/CD instructions
- `docs/guides/MAINTAINER_GUIDE.md` - This file

### Source Code
- `src/rir_generator/__init__.py` - Main module
- `src/rir_generator/_cffi/` - C++ extension bindings
  - `build.py` - CFFI build script
  - `rir_generator_core.cpp` - C++ implementation
  - `rir_generator_core.h` - C++ header

## 🔧 Common Tasks

### Update Dependencies

**Edit `pyproject.toml`:**
```toml
[project]
dependencies = [
    "numpy>=1.17.0",
    "scipy>=1.0.0",
    "cffi>=1.1.0",
]
```

### Add Python Version Support

**Edit `pyproject.toml`:**
```toml
requires-python = ">=3.8"
classifiers = [
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    # Add new version here
]
```

**Update CI workflows:**
```yaml
# .github/workflows/python-package.yml
matrix:
  python-version: ["3.8", "3.9", "3.10", "3.11", "3.12"]
```

```yaml
# .github/workflows/publish-to-pypi.yml
env:
  CIBW_BUILD: cp38-* cp39-* cp310-* cp311-* cp312-*
```

### Update GitHub Actions

Dependabot will create PRs automatically, or update manually:

**Edit workflow files:**
```yaml
- uses: actions/checkout@v4  # Update version
- uses: actions/setup-python@v5  # Update version
```

### Fix Failed CI

**Check logs:**
1. Go to: https://github.com/audiolabs/rir-generator/actions
2. Click on failed workflow
3. Review error logs

**Common fixes:**
```bash
# Reinstall if build fails
pip install -e .[dev] --force-reinstall

# Import issues - verify package structure:
python -c "import rir_generator; print(rir_generator.__file__)"
```

### Handle PyPI Publishing Issues

**If publishing fails:**

1. **Check PyPI Trusted Publishing:**
   - https://pypi.org/manage/project/rir-generator/settings/publishing/
   - Ensure workflow name matches: `publish-to-pypi.yml`

2. **Check permissions:**
   - Workflow needs `id-token: write` permission
   - Already configured in workflow file

3. **Manual publish (if needed):**
   ```bash
   python -m build
   twine upload dist/*
   # Will prompt for PyPI token
   ```

## 📊 Monitoring

### Check Package Status
- **PyPI:** https://pypi.org/project/rir-generator/
- **CI Status:** https://github.com/audiolabs/rir-generator/actions
- **Documentation:** https://rir-generator.readthedocs.io/

### Review Metrics
```bash
# Download stats (install pypinfo)
pip install pypinfo
pypinfo rir-generator

# GitHub stars/forks
# Check: https://github.com/audiolabs/rir-generator
```

## 🐛 Debugging

### Build Issues
```bash
# Verbose build
pip install -e . -v

# Check CFFI compilation
python src/rir_generator/_cffi/build.py

# System requirements
# macOS: xcode-select --install
# Linux: sudo apt-get install build-essential
# Windows: Visual Studio Build Tools
```

### Import Issues
```bash
# Check installation
pip show rir-generator

# Check import path
python -c "import rir_generator; print(rir_generator.__file__)"

# Reinstall
pip uninstall rir-generator -y
pip install -e .[dev]
```

### Test Issues
```bash
# Run specific test
pytest tests/test_example.py::test_parameters -v

# Run with coverage
pytest --cov=rir_generator --cov-report=html

# View coverage
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

## 📚 Resources

### Documentation
- Python Packaging Guide: https://packaging.python.org/
- PyPI Trusted Publishing: https://docs.pypi.org/trusted-publishers/
- GitHub Actions: https://docs.github.com/en/actions
- cibuildwheel: https://cibuildwheel.readthedocs.io/

### Project Links
- GitHub: https://github.com/audiolabs/rir-generator
- PyPI: https://pypi.org/project/rir-generator/
- Docs: https://rir-generator.readthedocs.io/
- DOI: https://doi.org/10.5281/zenodo.4133077

## 🆘 Getting Help

1. **Check documentation:** Start with README.md and docs/guides/
2. **Search issues:** https://github.com/audiolabs/rir-generator/issues
3. **Ask for help:** Open a new issue
4. **ReadTheDocs:** https://rir-generator.readthedocs.io/

## 📝 Notes

- **Version is automatic** - determined by git tags (see docs/guides/VERSIONING.md)
- Tag releases with `v` prefix (e.g., `v0.3.0`)
- Testing on Test PyPI before production is optional but recommended
- GitHub Actions are free for public repositories
- The `.[dev]` extra includes all development dependencies including pytest-cov

## ✅ Pre-Release Checklist

Before each release:
- [ ] Run full test suite: `pytest`
- [ ] Build package: `python -m build`
- [ ] Check package: `twine check dist/*`
- [ ] Verify version: `python -m setuptools_scm`
- [ ] Test on Test PyPI (optional)
- [ ] Create git tag: `git tag vX.Y.Z`
- [ ] Push tag: `git push origin vX.Y.Z`
- [ ] Create GitHub release
- [ ] Verify PyPI publication
- [ ] Test install from PyPI

---

**Last Updated:** October 2025
