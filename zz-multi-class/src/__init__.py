# src/__init__.py**
"""
Multiclass Attack Graph Classification Package

A comprehensive framework for multiclass classification on cybersecurity attack graphs
using Graph Neural Networks and traditional machine learning approaches.
"""

__version__ = "1.0.0"
__author__ = "Cybersecurity ML Team"

from . import ag_utils
from . import data_utils
from . import models
from . import train
from . import evaluate

__all__ = ["ag_utils", "data_utils", "models", "train", "evaluate"]