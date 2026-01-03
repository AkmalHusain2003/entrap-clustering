"""
ENTRAP: ENergy-based Topological Rescue of Ambiguous Points

A clustering refinement algorithm combining persistent homology with geometric
energy minimization to rescue noise points misclassified by density-based methods.

Author: Muhammad Akmal Husain
Version: 1.0
"""

from .entrap_main import ENTRAP, ENTRAP_Results

__version__ = "1.0"
__author__ = "Muhammad Akmal Husain"
__all__ = ["ENTRAP", "ENTRAP_Results"]