import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import copy
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

class CentralValleyAlpha(AlphaFunction):
    """changes the alpha channel of a colormap to be diverging (0->1, 0.5 > 0, 1->1)"""
    
    def __init__(self, min_alpha: float = 0.0, max_alpha: float = 1.0):
        super().__init__()
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha

    def segment_alpha(self) -> np.ndarray:
        return np.array([[0.0, self.max_alpha , self.max_alpha ], [0.5, self.min_alpha, self.min_alpha], [1.0, self.max_alpha , self.max_alpha ]])

    def alpha_func(self, i: int, N: int) -> float:
        return 2 * abs(i / N - 0.5) * (self.max_alpha - self.min_alpha) + self.min_alpha

class CentralPeakAlpha(AlphaFunction):
    
    def __init__(self, min_alpha: float = 0.0, max_alpha: float = 1.0):
        super().__init__()
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha

    def segment_alpha(self) -> np.ndarray:
        return np.array([[0.0, self.min_alpha, self.min_alpha], [0.5, self.max_alpha, self.max_alpha], [1.0, self.min_alpha, self.min_alpha]])

    def alpha_func(self, i: int, N: int) -> float:
        return 1 - abs(i / (N - 1) - 0.5) * 2 * (self.max_alpha - self.min_alpha) + self.min_alpha

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
    
class LuminanceAlpha(AlphaFunction):
    
    def luminance_alpha(self,r,g,b):
        return 1-(0.2126 * r + 0.7152 * g + 0.0722 * b)

    def __call__(self, cmap: Colormap):
        if isinstance(cmap, ListedColormap):
            colors = copy.deepcopy(cmap.colors)
            for i, a in enumerate(colors):
                current_alpha = self.luminance_alpha(a[0],a[1],a[2])
                if len(a) == 3:
                    a.append(current_alpha)
                elif len(a) == 4:
                    a[3] = current_alpha
            return ListedColormap(colors, cmap.name)
        elif isinstance(cmap, LinearSegmentedColormap):
            segmentdata = copy.deepcopy(cmap._segmentdata)
            len_segmentdata = len(segmentdata["red"])
            alpha=[]
            for i in range(len_segmentdata):
                index= segmentdata["red"][i][0]
                r1, g1, b1 = segmentdata["red"][i][1], segmentdata["green"][i][1], segmentdata["blue"][i][1]
                r2, g2, b2 = segmentdata["red"][i][2], segmentdata["green"][i][2], segmentdata["blue"][i][2]
                current_alpha_1 = self.luminance_alpha(r1, g1, b1)
                current_alpha_2 = self.luminance_alpha(r2, g2, b2)
                alpha.append((index, current_alpha_1, current_alpha_2))
            segmentdata["alpha"] = alpha
            return LinearSegmentedColormap(cmap.name, segmentdata)
        else:
            raise TypeError(
                "cmap must be either a ListedColormap or a LinearSegmentedColormap"
            )
            
class ZigzagAlpha(AlphaFunction):
    
    def __init__(self, 
                 boundary_alpha: float = 0.0,
                 central_alpha: float = 0.0,
                 peak_alpha: float = 1.0):
        super().__init__()
        self.boundary_alpha = boundary_alpha
        self.central_alpha = central_alpha
        self.peak_alpha = peak_alpha
        
    def segment_alpha(self) -> np.ndarray:
        return np.array(
            [
                [0.0, self.boundary_alpha, self.boundary_alpha],
                [0.25, self.peak_alpha, self.peak_alpha],
                [0.5, self.central_alpha, self.central_alpha],
                [0.75, self.peak_alpha, self.peak_alpha],
                [1.0, self.boundary_alpha, self.boundary_alpha],
            ]
        )
    
    def alpha_func(self, i: int, N: int) -> float:
        if i/N <= 0.25:
            return self.boundary_alpha + (self.peak_alpha - self.boundary_alpha) * (i / (0.25 * N))
        elif i/N <= 0.5:
            return self.peak_alpha + (self.central_alpha - self.peak_alpha) * ((i - 0.25 * N) / (0.25 * N))
        elif i/N <= 0.75:
            return self.central_alpha + (self.peak_alpha - self.central_alpha) * ((i - 0.5 * N) / (0.25 * N))
        else:
            return self.peak_alpha + (self.boundary_alpha - self.peak_alpha) * ((i - 0.75 * N) / (0.25 * N))
