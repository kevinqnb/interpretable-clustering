import numpy as np
from ._splitter import AxisAlignedSplitter

"""
NOTE: These splitters are designed to be stand-ins, when a real splitter 
isn't needed, or as simple easy calculation splitters used for testing purposes. 
"""

class DummySplitter(AxisAlignedSplitter):
    pass

class SimpleSplitter(AxisAlignedSplitter):
    def __init__(self, min_points_leaf : int = 1):
        super().__init__(min_points_leaf = min_points_leaf)
    
    def cost(self, indices : np.ndarray) -> float:
        return np.max(self.X[indices, :])