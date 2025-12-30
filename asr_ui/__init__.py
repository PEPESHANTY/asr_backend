"""
Modular ASR UI package.

This package provides a modular architecture for Automatic Speech Recognition (ASR)
with interchangeable models, a FastAPI backend, and a Streamlit frontend.
"""

__version__ = "0.1.0"
__author__ = "AA Moonshine"
__email__ = ""

from . import core
from . import models
from . import api
from . import ui

__all__ = ["core", "models", "api", "ui"]
