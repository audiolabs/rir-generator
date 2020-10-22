# Room Impulse Response Generator

[![Documentation Status](https://readthedocs.org/projects/rir-generator/badge/?version=latest)](https://rir-generator.readthedocs.io/en/latest/?badge=latest)
[![Build Status](https://travis-ci.org/audiolabs/rir-generator.svg?branch=master)](https://travis-ci.org/audiolabs/rir-generator)

This is a Python port of the RIR-Generator code from https://github.com/ehabets/RIR-Generator

## Installation

```sh
pip install rir-generator
```

## Usage

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
