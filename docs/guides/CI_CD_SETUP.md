# CI/CD Setup Guide

This document explains the continuous integration and deployment setup for the rir-generator package.

## Overview

The project uses **GitHub Actions** for:
1. **Continuous Integration (CI)** - Testing on every push/PR
2. **Continuous Deployment (CD)** - Automatic PyPI publishing on releases

## Workflows

### 1. CI Testing (`python-package.yml`)

**Triggers:** Push to master/main, Pull Requests

**What it does:**
- Tests across multiple Python versions (3.8, 3.9, 3.10, 3.11, 3.12)
- Tests on multiple platforms (Linux, macOS, Windows)
- Runs pytest with coverage
- Lints code with black (check only)
- Builds package and verifies distribution
- Attempts to build documentation

**Configuration:**
```yaml
on:
  push:
    branches: [ "master", "main" ]
  pull_request:
    branches: [ "master", "main" ]

# Important: Full git history is fetched for setuptools-scm
steps:
  - uses: actions/checkout@v4
    with:
      fetch-depth: 0  # Required for version detection
```

**Note:** The `fetch-depth: 0` is critical for setuptools-scm to determine the version from git tags. Without it, shallow clones will cause version detection errors.

### 2. PyPI Publishing (`publish-to-pypi.yml`)

**Triggers:** 
- GitHub Release publication (automatic)
- Manual workflow dispatch (for testing with Test PyPI option)

**What it does:**
- Builds wheels for all major platforms using `cibuildwheel`:
  - Linux (x86_64) - ubuntu-22.04
  - macOS Intel (x86_64) - macos-11
  - macOS Apple Silicon (arm64) - macos-14
  - Windows (x64) - windows-2022
- Builds source distribution (sdist)
- Tests built wheels with pytest
- Publishes to PyPI using Trusted Publishing

**Supported Python versions:** 3.8, 3.9, 3.10, 3.11, 3.12

## Setting Up PyPI Trusted Publishing

Trusted Publishing is a secure, token-free method to publish packages to PyPI.

### Initial Setup (One-time)

1. **On PyPI:**
   - Go to https://pypi.org/manage/account/publishing/
   - Or for a specific project: https://pypi.org/manage/project/rir-generator/settings/publishing/
   - Click "Add a new publisher"
   - Fill in:
     - PyPI Project Name: `rir-generator`
     - Owner: `audiolabs`
     - Repository name: `rir-generator`
     - Workflow name: `publish-to-pypi.yml`
     - Environment name: (leave empty)

2. **On GitHub:**
   - No additional setup needed! The workflow uses OIDC tokens automatically.

### Benefits of Trusted Publishing

- ✅ No API tokens to manage or rotate
- ✅ More secure (uses short-lived tokens)
- ✅ Automatic and transparent
- ✅ Works out of the box with GitHub Actions

## Publishing a New Release

### Method 1: GitHub Release (Recommended)

1. **Create a new tag:**
   ```bash
   git tag v0.3.0
   git push origin v0.3.0
   ```

2. **Create a GitHub Release:**
   - Go to: https://github.com/audiolabs/rir-generator/releases/new
   - Select the tag you created
   - Fill in release notes
   - Click "Publish release"

3. **Automatic Publishing:**
   - GitHub Actions automatically triggers
   - Builds wheels for all platforms
   - Publishes to PyPI
   - Monitor progress: https://github.com/audiolabs/rir-generator/actions

### Method 2: Manual Workflow Dispatch

For testing or emergency releases:

1. Go to: https://github.com/audiolabs/rir-generator/actions/workflows/publish-to-pypi.yml
2. Click "Run workflow"
3. Choose options:
   - **test_pypi: false** - Publish to real PyPI
   - **test_pypi: true** - Publish to Test PyPI (for testing)

## Testing Before Release

### Test on Test PyPI

1. Run workflow manually with `test_pypi: true`
2. Install from Test PyPI:
   ```bash
   pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ rir-generator
   ```

### Local Testing

Test the build process locally:

```bash
# Install build tools
pip install build twine cibuildwheel

# Build source distribution
python -m build --sdist

# Build wheel for current platform
python -m build --wheel

# Check the distribution
twine check dist/*

# Test install
pip install dist/rir_generator-*.whl
```

## Wheel Building Details

The workflow uses `cibuildwheel` to build wheels for multiple platforms:

```yaml
CIBW_BUILD: cp38-* cp39-* cp310-* cp311-* cp312-*
CIBW_SKIP: "*-win32 *-manylinux_i686 *-musllinux_*"
```

**Built wheels:**
- `cp38-manylinux_x86_64` - Python 3.8 on Linux
- `cp39-manylinux_x86_64` - Python 3.9 on Linux
- `cp310-manylinux_x86_64` - Python 3.10 on Linux
- `cp311-manylinux_x86_64` - Python 3.11 on Linux
- `cp312-manylinux_x86_64` - Python 3.12 on Linux
- `cp*-macosx_x86_64` - Intel macOS
- `cp*-macosx_arm64` - Apple Silicon macOS
- `cp*-win_amd64` - Windows 64-bit

**Skipped:**
- 32-bit Windows (`*-win32`)
- 32-bit Linux (`*-manylinux_i686`)
- musl Linux (`*-musllinux_*`)

## Monitoring Builds

### Check Build Status

- **CI Tests:** https://github.com/audiolabs/rir-generator/actions/workflows/python-package.yml
- **Publish:** https://github.com/audiolabs/rir-generator/actions/workflows/publish-to-pypi.yml

### Check Published Packages

- **PyPI:** https://pypi.org/project/rir-generator/
- **Test PyPI:** https://test.pypi.org/project/rir-generator/

### Debugging Failed Builds

1. Check the Actions tab for error logs
2. Common issues:
   - **C++ compilation errors:** Check compiler availability
   - **CFFI issues:** Ensure CFFI is installed in build environment
   - **Test failures:** Check test logs for details
   - **Publishing errors:** Verify PyPI trusted publishing setup

## Migrating from Travis CI

The old `.travis.yml` configuration has been replaced with GitHub Actions. Key differences:

| Feature | Travis CI | GitHub Actions |
|---------|-----------|----------------|
| Configuration | `.travis.yml` | `.github/workflows/*.yml` |
| Cost | Free tier limited | Free for public repos |
| Platform support | Limited | Linux, macOS, Windows |
| Speed | Slower | Faster |
| PyPI publishing | Manual tokens | Trusted Publishing |
| Wheel building | Manual | Automated via cibuildwheel |

## Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [cibuildwheel Documentation](https://cibuildwheel.readthedocs.io/)
- [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/)
- [Python Packaging Guide](https://packaging.python.org/)

## Maintenance

### Updating Python Versions

Edit the matrix in both workflows:

```yaml
matrix:
  python-version: ["3.8", "3.9", "3.10", "3.11", "3.12"]
```

And in `cibuildwheel` configuration:

```yaml
CIBW_BUILD: cp38-* cp39-* cp310-* cp311-* cp312-*
```

### Updating Dependencies

Dependencies are specified in `pyproject.toml`:

```toml
[project]
dependencies = [
    "numpy>=1.17.0",
    "scipy>=1.0.0",
    "cffi>=1.1.0",
]
```

### Updating GitHub Actions

Keep actions up to date by reviewing:
- `actions/checkout` version
- `actions/setup-python` version
- `pypa/cibuildwheel` version
- `pypa/gh-action-pypi-publish` version

Use Dependabot to automate this (add `.github/dependabot.yml`).
