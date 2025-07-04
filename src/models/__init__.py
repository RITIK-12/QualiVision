"""
Model architectures for QualiVision
"""

from .dover_model import DOVERModel, QualityAwareFusion
from .vjepa_model import VJEPAModel, OptimizedMOSHead

__all__ = [
    'DOVERModel',
    'QualityAwareFusion', 
    'VJEPAModel',
    'OptimizedMOSHead'
] 