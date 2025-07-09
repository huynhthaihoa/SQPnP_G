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

ext_modules = [
    Extension(
        "sqpnp_python.sqpnp_python",  # <package>.<module>
        ["sqpnp_bindings.cpp", os.path.join("..", "sqpnp", "sqpnp.cpp")],
        include_dirs=[
            os.path.join("..", "sqpnp"),
            "/usr/include/eigen3",
            pybind11.get_include(),
            np.get_include(),
        ],
        language="c++",
        extra_compile_args=[
            "-Wall",
            "-Wextra",
            "-O3",
            "-std=c++17",
            "-DEIGEN_MPL2_ONLY",
        ],
        extra_link_args=[],
    ),
]

setup(
    name="sqpnp_python",
    version="1.0.0",
    packages=["sqpnp_python"],
    package_dir={"sqpnp_python": "sqpnp_python"},
    ext_modules=ext_modules,
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "pybind11>=2.6.0",
    ],
    zip_safe=False,
)
