#!/usr/bin/env python

import setuptools


if __name__ == "__main__":
    setuptools.setup(
        name="rir-generator",
        version="0.1.0",
        description="Room Impulse Response Generator.",
        author="Nils Werner",
        author_email="nils.werner@fau.de",
        license="MIT",
        packages=setuptools.find_packages(),
        setup_requires=["cffi>=1.1.0"],
        install_requires=[
            "numpy>=1.6",
            "scipy>=0.13.0",
            "cffi>=1.1.0",
        ],
        cffi_modules=[
            "rir_generator/build.py:rir_generator",
        ],
        extras_require={
            "tests": [
                "pytest",
                "pytest-cov",
                "pytest-black",
                "sphinx",
                "sphinxcontrib-napoleon",
                "sphinx_rtd_theme",
                "numpydoc",
            ],
            "docs": [
                "sphinx",
                "sphinxcontrib-napoleon",
                "sphinxcontrib-bibtex",
                "sphinx_rtd_theme",
                "numpydoc",
            ],
        },
        zip_safe=False,
        include_package_data=True,
    )
