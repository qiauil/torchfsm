import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import copy
from matplotlib.colors import Colormap
import matplotlib as mlp
from typing import Union, Optional, Literal
from vape4d import render


class AlphaFunction:

    def segment_alpha(self) -> np.ndarray:
        """returns the alpha segment data for the colormap"""
        raise NotImplementedError("This method should be implemented by subclasses.")

    def alpha_func(self, i, N) -> float:
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
        return np.array(
            [
                [0.0, self.max_alpha, self.max_alpha],
                [0.5, self.min_alpha, self.min_alpha],
                [1.0, self.max_alpha, self.max_alpha],
            ]
        )

    def alpha_func(self, i: int, N: int) -> float:
        return 2 * abs(i / N - 0.5) * (self.max_alpha - self.min_alpha) + self.min_alpha


class CentralPeakAlpha(AlphaFunction):

    def __init__(self, min_alpha: float = 0.0, max_alpha: float = 1.0):
        super().__init__()
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha

    def segment_alpha(self) -> np.ndarray:
        return np.array(
            [
                [0.0, self.min_alpha, self.min_alpha],
                [0.5, self.max_alpha, self.max_alpha],
                [1.0, self.min_alpha, self.min_alpha],
            ]
        )

    def alpha_func(self, i: int, N: int) -> float:
        return (
            1
            - abs(i / (N - 1) - 0.5) * 2 * (self.max_alpha - self.min_alpha)
            + self.min_alpha
        )


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

    def luminance_alpha(self, r, g, b):
        return 1 - (0.2126 * r + 0.7152 * g + 0.0722 * b)

    def __call__(self, cmap: Colormap):
        if isinstance(cmap, ListedColormap):
            colors = copy.deepcopy(cmap.colors)
            for i, a in enumerate(colors):
                current_alpha = self.luminance_alpha(a[0], a[1], a[2])
                if len(a) == 3:
                    a.append(current_alpha)
                elif len(a) == 4:
                    a[3] = current_alpha
            return ListedColormap(colors, cmap.name)
        elif isinstance(cmap, LinearSegmentedColormap):
            segmentdata = copy.deepcopy(cmap._segmentdata)
            len_segmentdata = len(segmentdata["red"])
            alpha = []
            for i in range(len_segmentdata):
                index = segmentdata["red"][i][0]
                r1, g1, b1 = (
                    segmentdata["red"][i][1],
                    segmentdata["green"][i][1],
                    segmentdata["blue"][i][1],
                )
                r2, g2, b2 = (
                    segmentdata["red"][i][2],
                    segmentdata["green"][i][2],
                    segmentdata["blue"][i][2],
                )
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

    def __init__(
        self,
        boundary_alpha: float = 0.0,
        central_alpha: float = 0.0,
        peak_alpha: float = 1.0,
    ):
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
        if i / N <= 0.25:
            return self.boundary_alpha + (self.peak_alpha - self.boundary_alpha) * (
                i / (0.25 * N)
            )
        elif i / N <= 0.5:
            return self.peak_alpha + (self.central_alpha - self.peak_alpha) * (
                (i - 0.25 * N) / (0.25 * N)
            )
        elif i / N <= 0.75:
            return self.central_alpha + (self.peak_alpha - self.central_alpha) * (
                (i - 0.5 * N) / (0.25 * N)
            )
        else:
            return self.peak_alpha + (self.boundary_alpha - self.peak_alpha) * (
                (i - 0.75 * N) / (0.25 * N)
            )


def rendering_cmap(
    cmap: Union[str, Colormap],
    alpha_func: Union[
        Literal[
            "zigzag",
            "central_peak",
            "central_valley",
            "linear_increase",
            "linear_decrease",
            "luminance",
        ],
        AlphaFunction,
    ] = "zigzag",
) -> Colormap:
    if isinstance(cmap, str):
        cmap = mlp.colormaps[cmap]
    if isinstance(alpha_func, AlphaFunction):
        return alpha_func(cmap)
    elif alpha_func == "zigzag":
        cmap = ZigzagAlpha()(cmap)
    elif alpha_func == "central_peak":
        cmap = CentralPeakAlpha()(cmap)
    elif alpha_func == "central_valley":
        cmap = CentralValleyAlpha()(cmap)
    elif alpha_func == "linear_increase":
        cmap = LinearIncreasingAlpha()(cmap)
    elif alpha_func == "linear_decrease":
        cmap = LinearDecreasingAlpha()(cmap)
    elif alpha_func == "luminance":
        cmap = LuminanceAlpha()(cmap)
    else:
        raise ValueError(
            "The alpha function should be 'zigzag', 'central_peak', 'central_valley', 'linear_increase', 'linear_decrease', or 'luminance' or an instance of AlphaFunction."
        )
    return cmap


def render_3d_field(
    data: np.ndarray,
    cmap: Union[str, Colormap],
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    distance_scale: float = 10,
    background=(0, 0, 0, 0),
    width=512,
    height=512,
    alpha_func: Literal[
        "zigzag", "diverging", "linear_increase", "linear_decrease"
    ] = "zigzag",
    gamma_correction: float = 2.4,
    **kwargs,
) -> np.ndarray:
    cmap = rendering_cmap(cmap, alpha_func)
    additional_kwargs = {}
    keys=["time","distance_scale","spatial_interpolation","temporal_interpolation"]
    for key in keys:
        if key in kwargs:
            additional_kwargs[key] = kwargs.pop(key)
    img = render(
        data.astype(np.float32),  # expects float32
        cmap=cmap,  # zigzag alpha
        width=width,
        height=height,
        distance_scale=distance_scale,
        background=background,  # transparent background
        vmin=vmin,
        vmax=vmax,
        **additional_kwargs,
    )
    img = ((img / 255.0) ** (gamma_correction) * 255).astype(np.uint8)
    return img

def add_3d_coord(ax2d: mlp.axes.Axes, 
                 elev:int=45, 
                 azim:int=45,
                 roll:int=0,
                 size:float=0.1, 
                 length:float=0.6, 
                 x_loc:float=-0.05,
                 y_loc:float=-0.05,
                 x_label:str='x',
                 y_label:str='y',
                 z_label:str='z',
                 x_color:str='r',
                 y_color:str='g',
                 z_color:str='b',
                 arrow_length_ratio:float=0.25,
                 linewidth:int=1
                 ) -> mlp.axes.Axes:
    """
    Add a small 3D coordinate system (XYZ arrows)

    Args:
        ax2d (mlp.axes.Axes): The 2D axes to which the 3D coordinate system will be added
        elev (int, optional): Elevation angle for the 3D view. Defaults to 45.
        azim (int, optional): Azimuth angle for the 3D view. Defaults to 45.
        roll (int, optional): Roll angle for the 3D view. Defaults to 0.
        size (float, optional): Size of the 3D axes relative to the 2D axes. Defaults to 0.1.
        length (float, optional): Length of the arrows. Defaults to 0.6.
        x_loc (float, optional): X location offset for the 3D axes. Defaults to -0.05.
        y_loc (float, optional): Y location offset for the 3D axes. Defaults to -0.05.
        x_label (str, optional): Label for the X axis. Defaults to 'x'.
        y_label (str, optional): Label for the Y axis. Defaults to 'y'.
        z_label (str, optional): Label for the Z axis. Defaults to 'z'.
        x_color (str, optional): Color for the X axis arrow. Defaults to 'r'.
        y_color (str, optional): Color for the Y axis arrow. Defaults to 'g'.
        z_color (str, optional): Color for the Z axis arrow. Defaults to 'b'.
        arrow_length_ratio (float, optional): Ratio of the arrow head length to the total arrow length. Defaults to 0.25.

    Returns:
      The newly created 3D axes object
    """
    fig = ax2d.figure
    bbox = ax2d.get_position()  # in figure coords
    #width = bbox.width * size
    #height = bbox.height * size
    x0 = bbox.x0 + x_loc
    y0 = bbox.y0 + y_loc

    ax3 = fig.add_axes([x0, y0, size, size], projection='3d', facecolor='none')
    ax3.view_init(elev=elev, azim=azim, roll=roll)
    ax3.set_axis_off()

    vectors = np.array([[length, 0, 0],
                        [0, length, 0],
                        [0, 0, length]])
    ax3.quiver([0, 0, 0], [0, 0, 0], [0, 0, 0],
               vectors[:, 0], vectors[:, 1], vectors[:, 2],
               color=[z_color, x_color, y_color], 
               arrow_length_ratio=arrow_length_ratio, 
               linewidth=linewidth)

    ax3.set_xlim(0, length); ax3.set_ylim(0, length); ax3.set_zlim(0, length)

    ax3.text(1.2*length, 0, 0, z_label, color=z_color, fontsize=8, horizontalalignment='right', verticalalignment='center')
    ax3.text(0, 1.2*length, 0, x_label, color=x_color, fontsize=8, horizontalalignment='left', verticalalignment='center')
    ax3.text(0, 0, 1.2*length, y_label, color=y_color, fontsize=8, horizontalalignment='left', verticalalignment='bottom')

    try:
        ax3.dist = 7
    except Exception:
        pass

    return ax3