# Migration Guide: Package Restructuring

## Overview

The package has been restructured with a modernized layout following current Python packaging best practices. This guide explains the changes and how to migrate.

## What Changed

### 1. Modern Package Structure

The package has been restructured to use the `src/` layout:

```
OLD:
rir_generator/
  __init__.py
  _rir/
    build.py
    rir_generator_core.cpp
    rir_generator_core.h

NEW:
src/
  rir_generator/
    __init__.py
    _cffi/
      build.py
      rir_generator_core.cpp
      rir_generator_core.h
```

**Benefits:**
- Better separation of source code and build artifacts
- Prevents accidental imports from the working directory
- Aligns with modern Python packaging standards (PEP 517/518)

### 2. Build System Modernization

**Old:** `setup.py` + `setup.cfg`

**New:** `pyproject.toml` with full PEP 517/518 support

**Benefits:**
- Single source of truth for project metadata
- Better dependency resolution
- Future-proof configuration
- Easier to maintain

### 3. CI/CD Changes

**Old:** Travis CI (`.travis.yml`)

**New:** GitHub Actions (`.github/workflows/`)

**Changes:**
- `python-package.yml` - Modern CI testing across multiple Python versions and platforms
- `publish-to-pypi.yml` - Automated PyPI publishing on releases

**Benefits:**
- Faster builds
- Better platform coverage (Linux, macOS, Windows)
- Free for open source
- Integrated with GitHub
- Automated wheel building for multiple platforms

### 4. Improved Folder Naming

- `_rir/` → `_cffi/` - Clearer indication that this contains CFFI bindings
- Internal CFFI module renamed from `rir` to `_rir` to indicate it's private/internal

## For Users (Package Installation)

### No Changes Required

If you're installing and using the package, **nothing changes**:

```python
pip install rir-generator

import rir_generator
h = rir_generator.generate(...)  # Works exactly the same
```

The API remains 100% compatible.

## For Contributors (Development)

### Installing for Development

**Old:**
```bash
pip install -e .[tests,docs]
```

**New:**
```bash
pip install -e .[dev]
# or for just tests
pip install -e .[tests]
```

### Running Tests

**No change:**
```bash
pytest
```

### Building the Package

**Old:**
```bash
python setup.py sdist bdist_wheel
```

**New:**
```bash
python -m build
```

### File Paths

If your code or scripts reference internal paths (which is not recommended for external use):

**Old:**
```python
# Internal structure (not part of public API)
rir_generator/_rir/
```

**New:**
```python
# Internal structure renamed for clarity (not part of public API)
rir_generator/_cffi/
```

**Note:** The public API (`rir_generator.generate()`) remains unchanged.

## For Maintainers

### Publishing to PyPI

**Old:** Travis CI with manual configuration

**New:** Automated via GitHub Actions

#### Publishing Process:

1. Create a git tag (e.g., `git tag v0.3.0` and `git push origin v0.3.0`)
2. Create a new release on GitHub selecting that tag
3. GitHub Actions automatically:
   - Builds wheels for all platforms (Linux, macOS, Windows)
   - Builds source distribution
   - Publishes to PyPI using trusted publishing

#### Manual Publishing (if needed):

```bash
python -m build
twine upload dist/*
```

### Testing PyPI Publishing

You can test the publishing workflow:

1. Go to Actions tab
2. Select "Publish to PyPI" workflow
3. Click "Run workflow"
4. Check "Publish to Test PyPI instead"

## Version Support

- **Python 3.8+** (dropped Python 3.6, 3.7 which are EOL)
- **Wheels provided for:** Linux, macOS (Intel & Apple Silicon), Windows
- **Architectures:** x86_64, arm64 (Apple Silicon)

## Dependency Updates

Minimum versions updated for better compatibility:
- `numpy >= 1.17.0` (was 1.6)
- `scipy >= 1.0.0` (was 0.13.0)

These versions are still quite old and should not affect most users.

## Troubleshooting

### Build Issues

If you encounter build issues:

```bash
pip install --upgrade pip setuptools wheel cffi
pip install -e . --no-build-isolation
```

### Import Issues

If imports fail, ensure you're not in the project root directory or uninstall the old version:

```bash
pip uninstall rir-generator
pip install -e .
```

### C Extension Issues

If the C extension fails to build:

```bash
# Check CFFI installation
pip install --upgrade cffi

# On macOS, ensure Xcode Command Line Tools are installed
xcode-select --install

# On Linux, ensure gcc is installed
sudo apt-get install build-essential  # Debian/Ubuntu
sudo yum install gcc gcc-c++          # RedHat/CentOS
```

## Questions?

- Open an issue: https://github.com/audiolabs/rir-generator/issues
- Check the documentation: https://rir-generator.readthedocs.io/

## Backwards Compatibility

The **public API remains unchanged**. All existing code using `rir_generator.generate()` will continue to work without modifications.
