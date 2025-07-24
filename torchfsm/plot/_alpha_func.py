import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import copy
from vape4d.utils import diverging_alpha, linear_increasing_alpha
from matplotlib.colors import Colormap


class AlphaFunction:
    
    
    def segment_alpha(self)-> np.ndarray:
        """returns the alpha segment data for the colormap"""
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def alpha_func(self,i,N)-> float:
        """returns the alpha value for the given index and total number of segments"""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def __call__(self, cmap: Colormap):
        if isinstance(cmap, ListedColormap):
            colors = copy.deepcopy(cmap.colors)
            for i, a in enumerate(colors):
                current_alpha = self.alpha_func(i, cmap.N)
                if len(a) == 3:
                    a.append(current_alpha)
                elif len(a) == 4:
                    a[3] = current_alpha
            return ListedColormap(colors, cmap.name)
        elif isinstance(cmap, LinearSegmentedColormap):
            segmentdata = copy.deepcopy(cmap._segmentdata)
            segmentdata["alpha"] = self.segment_alpha()
            return LinearSegmentedColormap(cmap.name, segmentdata)
        else:
            raise TypeError(
                "cmap must be either a ListedColormap or a LinearSegmentedColormap"
            )
            
class DivergingAlpha(AlphaFunction):
    """changes the alpha channel of a colormap to be diverging (0->1, 0.5 > 0, 1->1)"""

    def segment_alpha(self) -> np.ndarray:
        return np.array([[0.0, 1.0, 1.0], [0.5, 0.0, 0.0], [1.0, 1.0, 1.0]])

    def alpha_func(self, i: int, N: int) -> float:
        return 2 * abs(i / N - 0.5)


class LinearIncreasingAlpha(AlphaFunction):
    """changes the alpha channel of a colormap to be linear (0->0, 1->1)"""

    def segment_alpha(self) -> np.ndarray:
        return np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])

    def alpha_func(self, i: int, N: int) -> float:
        return i / (N - 1)
    
class LinearDecreasingAlpha(AlphaFunction):
    """changes the alpha channel of a colormap to be linear (1->1, 0->0)"""

    def segment_alpha(self) -> np.ndarray:
        return np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 0.0]])

    def alpha_func(self, i: int, N: int) -> float:
        return 1 - (i / (N - 1))
    
class ZigzagAlpha(AlphaFunction):
    
    def __init__(self, min_alpha: float = 0.0):
        super().__init__()
        self.min_alpha = min_alpha
        
    def segment_alpha(self) -> np.ndarray:
        return np.array(
            [
                [0.0, 0.0, 0.0],
                [0.25, 1.0, 1.0],
                [0.5, 0.0, 0.0],
                [0.75, 1.0, 1.0],
                [1.0, 0.0, 0.0],
            ]
        )
    
    def alpha_func(self, i: int, N: int) -> float:
        """returns the alpha value for the given index and total number of segments"""
        return (self.triangle_wave(i / (N - 1), 0.5) * (1 - self.min_alpha)) + self.min_alpha

    # "triangle_wave" and "zigzag_alpha" functions are copied from exponax(https://github.com/Ceyron/exponax) exponax/exponax/viz/_volume.py
    def triangle_wave(self, x, p):
        return 2 * np.abs(x / p - np.floor(x / p + 0.5))
