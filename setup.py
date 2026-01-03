"""
setup.py - Build script untuk kompilasi modul Cython ENTRAP (Hybrid Version)

Pure Python: topological_energy.py, dek_selector.py
Cython: kernels, utilities, ebm_engine

Usage:
    python setup.py build_ext --inplace
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# Compiler directives untuk optimasi
compiler_directives = {
    'language_level': 3,
    'boundscheck': False,
    'wraparound': False,
    'cdivision': True,
    'initializedcheck': False,
    'embedsignature': True,
}

# Define extensions - ONLY Cython modules
extensions = [
    # 1. Kernels - tidak ada dependency internal
    Extension(
        name="entrap.kernels",
        sources=["entrap/kernels.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=['-O3', '-ffast-math'],
        extra_link_args=['-O3'],
    ),
    
    # 2. Utilities - tidak ada dependency internal
    Extension(
        name="entrap.utilities",
        sources=["entrap/utilities.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=['-O3', '-ffast-math'],
        extra_link_args=['-O3'],
    ),
    
    # 3. EBM Engine - depends on kernels & utilities
    #    Uses pure Python topological_energy & dek_selector
    Extension(
        name="entrap.ebm_engine",
        sources=["entrap/ebm_engine.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=['-O3', '-ffast-math'],
        extra_link_args=['-O3'],
    ),
]

setup(
    name="ENTRAP",
    version="1.0",
    description="ENergy-based Topological Rescue of Ambiguous Points",
    author="Muhammad Akmal Husain",
    packages=['entrap'],
    ext_modules=cythonize(
        extensions,
        compiler_directives=compiler_directives,
        annotate=True,  # Generate HTML annotation files
    ),
    install_requires=[
        'numpy>=1.19.0',
        'scipy>=1.5.0',
        'scikit-learn>=0.23.0',
        'hdbscan>=0.8.27',
        'ripser>=0.6.0',
        'cython>=0.29.0',
    ],
    python_requires='>=3.7',
    zip_safe=False,
)