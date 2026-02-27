# Room Impulse Response Generator

[![Documentation Status](https://readthedocs.org/projects/rir-generator/badge/?version=latest)](https://rir-generator.readthedocs.io/en/latest/?badge=latest)
[![CI Tests](https://github.com/audiolabs/rir-generator/actions/workflows/python-package.yml/badge.svg)](https://github.com/audiolabs/rir-generator/actions/workflows/python-package.yml)
[![PyPI version](https://badge.fury.io/py/rir-generator.svg)](https://badge.fury.io/py/rir-generator)
[![Python versions](https://img.shields.io/pypi/pyversions/rir-generator.svg)](https://pypi.org/project/rir-generator/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4133077.svg)](https://doi.org/10.5281/zenodo.4133077)

Python- and C-based [room impulse response](https://en.wikipedia.org/wiki/Impulse_response#Acoustic_and_audio_applications) generator, for use in [convolutional reverb](https://en.wikipedia.org/wiki/Convolution_reverb).

Official Python port of https://github.com/ehabets/RIR-Generator.

Two backends are available:

| Backend | Module | Speed (CPU) | Speed (GPU) | Extra dependency |
|---------|--------|-------------|-------------|------------------|
| C++ (default) | `rir_generator` | ~2× RTF | — | none |
| PyTorch (GPU) | `rir_generator.gpu` | ~1× RTF | ~10–100× RTF | `torch>=2.0` |

Both backends implement the same Image Source Method and produce numerically
identical results (difference < 1e-15).

## Installation

### Default (C++ backend)

```sh
pip install rir-generator
```

### With GPU support (PyTorch backend)

```sh
pip install rir-generator[gpu]
# or
pip install rir-generator
pip install -r requirements-gpu.txt
```

## Usage

### C++ backend

```python
import numpy as np
import scipy.signal as ss
import soundfile as sf
import rir_generator as rir

signal, fs = sf.read("bark.wav", always_2d=True)

h = rir.generate(
    c=340,                  # Sound velocity (m/s)
    fs=fs,                  # Sample frequency (samples/s)
    r=[                     # Receiver position(s) [x y z] (m)
        [2, 1.5, 1],
        [2, 1.5, 2],
        [2, 1.5, 3]
    ],
    s=[2, 3.5, 2],          # Source position [x y z] (m)
    L=[5, 4, 6],            # Room dimensions [x y z] (m)
    reverberation_time=0.4, # Reverberation time (s)
    nsample=4096,           # Number of output samples
)

print(h.shape)              # (4096, 3)
print(signal.shape)         # (11462, 2)

# Convolve 2-channel signal with 3 impulse responses
signal = ss.convolve(h[:, None, :], signal[:, :, None])

print(signal.shape)         # (15557, 2, 3)
```

### GPU/PyTorch backend

The GPU backend is a drop-in replacement with two additional keyword
arguments: `device` and `chunk_size`.

```python
from rir_generator import gpu as rir_gpu

h = rir_gpu.generate(
    c=340,
    fs=fs,
    r=[
        [2, 1.5, 1],
        [2, 1.5, 2],
        [2, 1.5, 3]
    ],
    s=[2, 3.5, 2],
    L=[5, 4, 6],
    reverberation_time=0.4,
    nsample=4096,
    device="cuda",      # "cuda" | "cpu" | torch.device(...)
                        # default: CUDA if available, else CPU
    chunk_size=4096,    # image sources per batch; increase on large GPUs
)

print(h.shape)          # (4096, 3) — identical layout to C++ backend
```

All other parameters (`beta`, `mtype`, `order`, `dim`, `orientation`,
`hp_filter`) are shared between both backends.

## Development

### Installation for Development

```sh
git clone https://github.com/audiolabs/rir-generator.git
cd rir-generator
pip install -e .[dev]
```

### Running Tests

```sh
pytest
```

### Comparing Backends

```sh
python examples/compare_gpu.py
```

This prints a numerical and timing comparison and saves a `rir_comparison.png`
figure showing both impulse responses and their difference.

### Building from Source

```sh
python -m build
```

### Project Structure

The project follows the `src/` layout with **automatic versioning from git tags**:

```
src/
  rir_generator/
    __init__.py               # C++ backend (default)
    gpu.py                    # PyTorch GPU backend
    _cffi/                    # CFFI C++ extension bindings
      build.py                # CFFI build configuration
      rir_generator_core.cpp  # C++ implementation
      rir_generator_core.h    # C++ header
examples/
  compare_gpu.py              # Numerical, visual and timing comparison
```
