"""CFFI build script for rir_generator C++ extension."""
import os
from cffi import FFI

rir_generator = FFI()

compile_extras = {
    "libraries": ["m"],
}

# MSVC does not like -march=native -lm
import sys
if sys.platform == "win32":
    compile_extras = {}

# Use relative path from project root
# During build, sources should be relative to setup.py location
rir_generator.set_source(
    "rir_generator._rir",
    '#include "rir_generator_core.h"',
    sources=["src/rir_generator/_cffi/rir_generator_core.cpp"],
    include_dirs=["src/rir_generator/_cffi"],
    **compile_extras
)

rir_generator.cdef(
    """
void computeRIR(
    double* imp,
    double c,
    double fs,
    double* rr,
    int nr_of_mics,
    int nsamples,
    double* ss,
    double* LL,
    double* beta,
    char mtype,
    int order,
    double* angle,
    int hp_filter
);
"""
)

if __name__ == "__main__":
    rir_generator.compile()
