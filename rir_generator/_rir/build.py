import os
import sys
from cffi import FFI

rir_generator = FFI()

compile_extras = {
    "extra_compile_args": ["-march=native"],
    "libraries": ["m"],
}

# MSVC does not like -march=native -lm
if sys.platform == "win32":
    compile_extras = {}

rir_generator.set_source(
    "rir_generator.rir",
    '#include "rir_generator_core.h"',
    sources=["rir_generator/_rir/rir_generator_core.cpp"],
    include_dirs=["rir_generator/_rir"],
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
