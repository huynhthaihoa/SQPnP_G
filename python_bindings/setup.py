#!/usr/bin/env python3
"""
Setup script for SQPnP Python bindings using pybind11.
Auto-discovers pybind11's headers under site-packages or $CONDA_PREFIX/include,
and its library under $CONDA_PREFIX/lib.
Finds Eigen3 headers in standard locations.
"""

import os
import sys
import site
from setuptools import setup, Extension
import pybind11
import numpy as np

# Get the directory containing this setup.py
current_dir = os.path.dirname(os.path.abspath(__file__))
sqpnp_dir = os.path.join(current_dir, "..", "sqpnp")

# Define the extension module
ext_modules = [
    Extension(
        "sqpnp_python",
        ["sqpnp_bindings.cpp", os.path.join(sqpnp_dir, "sqpnp.cpp")],
        include_dirs=[
            sqpnp_dir,
            "/usr/include/eigen3",  # Add Eigen3 include path
            pybind11.get_include(),
            np.get_include(),
        ],
        language="c++",
        extra_compile_args=[
            "-Wall",
            "-Wextra",
            "-O3",
            "-std=c++17",
            "-DEIGEN_MPL2_ONLY",  # Use MPL2 license for Eigen
        ],
        extra_link_args=[],
    ),
]

setup(
    name="sqpnp_python",
    version="1.0.0",
    author="Manoj Pandey",
    author_email="pandeymsp16@gmail.com",
    description="Python bindings for SQPnP - Efficient Perspective-n-Point Algorithm",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    ext_modules=ext_modules,
    zip_safe=False,
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "pybind11>=2.6.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Computer Vision",
    ],
)
