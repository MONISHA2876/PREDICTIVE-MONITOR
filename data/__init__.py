# data/__init__.py
"""
Data management package.
Handles dataset generation, loading, preprocessing and sequence creation.
"""

from .generator import DatasetGenerator
from .preprocessor import DataPreprocessor

__all__ = ["DatasetGenerator", "DataPreprocessor"]
