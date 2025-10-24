#!/usr/bin/env python
"""
Setup script for rir-generator package.

This handles CFFI extension compilation.
The main configuration is in pyproject.toml.
"""

from setuptools import setup

if __name__ == "__main__":
    setup(
        cffi_modules=["src/rir_generator/_cffi/build.py:rir_generator"],
    )
