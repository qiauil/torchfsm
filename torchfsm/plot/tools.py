import numpy as np
import matplotlib as mlp
import matplotlib.colors as colors
from typing import Callable


def sym_colormap(d_min, d_max, d_cen=0, cmap="twilight", cmapname="sym_map"):
    """
    Generate a symmetric colormap.

    Args:
        d_min (float): The minimum value of the colormap.
        d_max (float): The maximum value of the colormap.
        d_cen (float, optional): The center value of the colormap. Defaults to 0.
        cmap (str, optional): The colormap to use. Defaults to "twilight".
        cmapname (str, optional): The name of the colormap. Defaults to "sym_map".

    Returns:
        matplotlib.colors.LinearSegmentedColormap: The generated colormap.
    """
    if abs(d_max - d_cen) > abs(d_min - d_cen):
        max_v = 1
        low_v = 0.5 - (d_cen - d_min) / (d_max - d_cen) * 0.5
    else:
        low_v = 0
        max_v = 0.5 + (d_max - d_cen) / (d_cen - d_min) * 0.5
    if isinstance(cmap, str):
        cmap = mlp.colormaps[cmap]
    return colors.LinearSegmentedColormap.from_list(
        cmapname, cmap(np.linspace(low_v, max_v, 100))
    )


def generate_uniform_ticks(
    start: float, end: float, n_tick, label_func: Callable[[np.number], str]
):
    """
    Generate uniform ticks for a plot.

    Args:
        start (float): The start value of the ticks.
        end (float): The end value of the ticks.
        n_tick (int): The number of ticks to generate.
        label_func (Callable[[np.number], str]): A function to format the tick labels.

    Returns:
        Tuple[Sequence[float], Sequence[str]]: A tuple containing the tick positions and labels.
    """
    ticks = np.linspace(start, end, n_tick)
    return ticks, [label_func(tick) for tick in ticks]

