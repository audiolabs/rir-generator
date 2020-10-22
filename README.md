# rir_generator

[![Documentation Status](https://readthedocs.org/projects/rir-generator/badge/?version=latest)](https://rir-generator.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://travis-ci.org/audiolabs/rir-generator.svg?branch=master)](https://travis-ci.org/audiolabs/rir-generator)

This is a python port of the RIR-Generator code from https://github.com/ehabets/RIR-Generator

This package contains:

 - C implementation of the RIR code and the corresponding .h file
 - A cffi wrapper

Python dependencies: cffi, numpy

## Installation

```sh
pip install rir-generator
```

## Usage

Example

```python
import rir_generator

h = rir_generator.generate(
    c=340,                  # Sound velocity (m/s)
    fs=16000,               # Sample frequency (samples/s)
    r=[                     # Receiver position(s) [x y z] (m)
        [2, 1.5, 2],
        [2, 1.5, 3]
    ],
    s=[2, 3.5, 2],          # Source position [x y z] (m)
    L=[5, 4, 6],            # Room dimensions [x y z] (m)
    reverberation_time=0.4, # Reverberation time (s)
    nsample=4096,           # Number of output samples
)
```
