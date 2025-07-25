import numpy as np
import matplotlib as mlp
from matplotlib.colors import Colormap
import os
from .alpha_func import *
from typing import Union, Optional, Sequence, Literal
from vape4d import render
from IPython.display import HTML


def _find_min_max(
    traj: np.ndarray,
    vmin: Union[float, Sequence[Optional[float]]],
    vmax: Union[float, Sequence[Optional[float]]],
):
    axis = tuple([0, 1] + [i + 3 for i in range(len(traj.shape) - 3)])
    vmins = np.min(traj, axis=axis)
    vmaxs = np.max(traj, axis=axis)
    if vmin is not None:
        if isinstance(vmin, float) or isinstance(vmin, int):
            vmin = [vmin] * len(vmins)
        elif len(vmin) != len(vmins):
            raise ValueError(
                "The number of vmin values should be equal to the number of channels in the input trajectory."
            )
        vmins = np.asarray(
            [vmin[i] if vmin[i] is not None else vmins[i] for i in range(len(vmins))]
        )
    if vmax is not None:
        if isinstance(vmax, float) or isinstance(vmax, int):
            vmax = [vmax] * len(vmaxs)
        elif len(vmax) != len(vmaxs):
            raise ValueError(
                "The number of vmax values should be equal to the number of channels in the input trajectory."
            )
        vmaxs = np.asarray(
            [vmax[i] if vmax[i] is not None else vmaxs[i] for i in range(len(vmaxs))]
        )
    return vmins, vmaxs


def _data_plot(
    i: int,
    fields: np.ndarray,
    n_dim: int,
    n_channel: int,
    batch_size: int,
    channel_names: Sequence[str],
    batch_names: Sequence[str],
    animation: bool = True,
):
    i_row = i // n_channel
    i_column = i % n_channel

    if n_dim == 1:
        if animation:
            y_label = (
                batch_names[i_row] + os.linesep + "value"
                if len(batch_names) > 1
                else "value"
            )
            x_label = (
                "x" + os.linesep + channel_names[i_column]
                if len(channel_names) > 1
                else "x"
            )
            data_i = fields[i_row, :, i_column, :]
        else:
            y_label = "x"
            if len(batch_names) > 1:
                y_label = batch_names[i_row] + os.linesep + y_label
            x_label = "t"
            data_i = fields[i_row, :, i_column, :]
    if n_dim == 2:
        if animation:
            y_label = "y"
            if len(batch_names) > 1:
                y_label = batch_names[i_row] + os.linesep + y_label
            x_label = "x"
            data_i = fields[i_row, :, i_column, ...]
        else:
            x_label = channel_names[i_column] if len(channel_names) > 1 else None
            y_label = batch_names[i_row] if len(batch_names) > 1 else None
            data_i = fields[i_row, :, i_column, ...]
    elif n_dim == 3:
        x_label = (
            channel_names[i_column]
            if len(channel_names) > 1 and i_row == batch_size - 1
            else None
        )
        y_label = batch_names[i_row] if len(batch_names) > 1 and i_column == 0 else None
        data_i = None
    return data_i, x_label, y_label, i_column, i_row


def _to_rendering_cmap(
    cmap: Union[str, Colormap],
    alpha_func: Union[
        Literal[
            "zigzag",
            "central_peak",
            "central_valley",
            "linear_increase",
            "linear_decrease",
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


def _render(
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
):
    cmap = _to_rendering_cmap(cmap, alpha_func)
    img = render(
        data.astype(np.float32),  # expects float32
        cmap=cmap,  # zigzag alpha
        width=width,
        height=height,
        distance_scale=distance_scale,
        background=background,  # transparent background
        vmin=vmin,
        vmax=vmax,
        **kwargs,
    )
    img = ((img / 255.0) ** (gamma_correction) * 255).astype(np.uint8)
    return img
