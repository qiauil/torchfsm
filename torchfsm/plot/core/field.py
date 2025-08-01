import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap
import torch
from ..render import AlphaFunction, render_3d_field
from typing import Union, Optional, Sequence, Tuple, Literal

def plot_1D_field(
    ax: plt.Axes,
    data: Union[np.ndarray, torch.Tensor],
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    title: Optional[str] = None,
    title_loc="center",
    show_ticks=True,
    ticks_x: Tuple[Sequence[float], Sequence[str]] = None,
    ticks_y: Tuple[Sequence[float], Sequence[str]] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    extend_value_range: bool = True,
    grid=True,
    **kwargs,
):
    """
    Plot a 1D field.

    Args:
        ax (plt.Axes): The axes to plot on.
        data (Union[np.ndarray, torch.Tensor]): The data to plot.
        x_label (Optional[str], optional): The label for the x-axis. Defaults to None.
        y_label (Optional[str], optional): The label for the y-axis. Defaults to None.
        title (Optional[str], optional): The title of the plot. Defaults to None.
        title_loc (str, optional): The location of the title. Defaults to "center".
        show_ticks (bool, optional): Whether to show ticks. Defaults to True.
        ticks_x (Tuple[Sequence[float], Sequence[str]], optional): Custom ticks for the x-axis. Defaults to None.
        ticks_y (Tuple[Sequence[float], Sequence[str]], optional): Custom ticks for the y-axis. Defaults to None.
        vmin (Optional[float], optional): The minimum value for the color scale. Defaults to None.
        vmax (Optional[float], optional): The maximum value for the color scale. Defaults to None.
        extend_value_range (bool, optional): Whether to extend the value range. Defaults to True.
        grid (bool, optional): Whether to show grid lines. Defaults to True.
        **kwargs: Additional keyword arguments for the plot.

    """
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    elif not isinstance(data, np.ndarray):
        data = np.asarray(data)
    if len(data.shape) != 1:
        raise ValueError("Only support 1D data.")
    ax.plot(data, **kwargs)
    if not show_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        if ticks_x is not None:
            ax.set_xticks(ticks_x[0], labels=ticks_x[1])
        if ticks_y is not None:
            ax.set_yticks(ticks_y[0], labels=ticks_y[1])
    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)
    if title is not None:
        ax.set_title(title, loc=title_loc)
    if vmin is not None and vmax is not None:
        if extend_value_range:
            ax.set_ylim(vmin * 1.05, vmax * 1.05)
        else:
            ax.set_ylim(vmin, vmax)
    if grid:
        ax.grid()


def plot_2D_field(
    ax: plt.Axes,
    data: Union[np.ndarray, torch.Tensor],
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    title: Optional[str] = None,
    title_loc="center",
    interpolation="none",
    aspect="auto",
    cmap: Union[str, Colormap] = "coolwarm",
    show_ticks=True,
    ticks_x: Tuple[Sequence[float], Sequence[str]] = None,
    ticks_y: Tuple[Sequence[float], Sequence[str]] = None,
    **kwargs,
):
    """
    Plot a 2D field.

    Args:
        ax (plt.Axes): The axes to plot on.
        data (Union[np.ndarray, torch.Tensor]): The data to plot.
        x_label (Optional[str], optional): The label for the x-axis. Defaults to None.
        y_label (Optional[str], optional): The label for the y-axis. Defaults to None.
        title (Optional[str], optional): The title of the plot. Defaults to None.
        title_loc (str, optional): The location of the title. Defaults to "center".
        interpolation (str, optional): The interpolation method. Defaults to "none".
        aspect (str, optional): The aspect ratio. Defaults to "auto".
        cmap (Union[str, Colormap], optional): The colormap to use. Defaults to "coolwarm".
        show_ticks (bool, optional): Whether to show ticks. Defaults to True.
        ticks_x (Tuple[Sequence[float], Sequence[str]], optional): Custom ticks for the x-axis. Defaults to None.
        ticks_y (Tuple[Sequence[float], Sequence[str]], optional): Custom ticks for the y-axis. Defaults to None.
        **kwargs: Additional keyword arguments for the plot.
    """
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    elif not isinstance(data, np.ndarray):
        data = np.asarray(data)
    if len(data.shape) != 2:
        raise ValueError("Only support 2D data.")
    im = ax.imshow(
        data.T,
        interpolation=interpolation,
        cmap=cmap,
        origin="lower",
        aspect=aspect,
        **kwargs,
    )
    if not show_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)
    if title is not None:
        ax.set_title(title, loc=title_loc)
    if ticks_x is not None:
        ax.set_xticks(ticks_x[0], labels=ticks_x[1])
    if ticks_y is not None:
        ax.set_yticks(ticks_y[0], labels=ticks_y[1])
    return im


def _plot_3D_field(
    ax: plt.Axes,
    img: np.ndarray,
    bottom_label: Optional[str] = None,
    left_label: Optional[str] = None,
    title: Optional[str] = None,
    title_loc="center",
    aspect="auto",
):
    """
    Plot a 3D field.

    Args:
        ax (plt.Axes): The axes to plot on.
        img (np.ndarray): The image to plot.
        bottom_label (Optional[str], optional): The label for the bottom axis. Defaults to None.
        left_label (Optional[str], optional): The label for the left axis. Defaults to None.
        title (Optional[str], optional): The title of the plot. Defaults to None.
        title_loc (str, optional): The location of the title. Defaults to "center".
        aspect (str, optional): The aspect ratio. Defaults to "auto".
    """
    im = ax.imshow(img, aspect=aspect)
    ax.set_xticks([])
    ax.set_yticks([])
    if bottom_label is not None:
        ax.set_xlabel(bottom_label)
    if left_label is not None:
        ax.set_ylabel(left_label)
    if title is not None:
        ax.set_title(title, loc=title_loc)
    for loc in ["bottom", "top", "right", "left"]:
        ax.spines[loc].set_color("white")
    return im


def plot_3D_field(
    ax: plt.Axes,
    data: Union[np.ndarray, torch.Tensor],
    bottom_label: Optional[str] = None,
    left_label: Optional[str] = None,
    title: Optional[str] = None,
    title_loc="center",
    aspect="auto",
    cmap: Union[str, Colormap] = "coolwarm",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    return_cmap: bool = False,
    distance_scale: float = 10,
    background=(0, 0, 0, 0),
    width=512,
    height=512,
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
    gamma_correction: float = 2.4,
    **kwargs,
):
    """
    Plot a 3D field.
    Powered by https://github.com/KeKsBoTer/vape4d

    Args:
        ax (plt.Axes): The axes to plot on.
        data (Union[np.ndarray, torch.Tensor]): The data to plot.
        bottom_label (Optional[str], optional): The label for the bottom axis. Defaults to None.
        left_label (Optional[str], optional): The label for the left axis. Defaults to None.
        title (Optional[str], optional): The title of the plot. Defaults to None.
        title_loc (str, optional): The location of the title. Defaults to "center".
        aspect (str, optional): The aspect ratio. Defaults to "auto".
        cmap (Union[str, Colormap], optional): The colormap to use. Defaults to "coolwarm".
        vmin (Optional[float], optional): The minimum value for the color scale. Defaults to None.
        vmax (Optional[float], optional): The maximum value for the color scale. Defaults to None.
        return_cmap (bool, optional): Whether to return the colormap. Defaults to False.
        distance_scale (float, optional): The distance scale for rendering. Defaults to 10.
        background (tuple, optional): The background color. Defaults to (0, 0, 0, 0).
        width (int, optional): The width of the rendered image. Defaults to 512.
        height (int, optional): The height of the rendered image. Defaults to 512.
        alpha_func (Union[Literal["zigzag","central_peak","central_valley","linear_increase","linear_decrease", "luminance",],AlphaFunction,], optional): The alpha function. Defaults to "zigzag".
        gamma_correction (float, optional): The gamma correction factor. Defaults to 2.4.
        **kwargs: Additional keyword arguments for the plot.
    """
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    elif not isinstance(data, np.ndarray):
        data = np.asarray(data)
    if len(data.shape) == 3:
        data = np.expand_dims(data, 0)
    elif not (len(data.shape) == 4 and data.shape[0] == 1):
        raise ValueError("Only support 3D data with shape of [X,Y,Z] or [1,X,Y,Z].")
    img = render_3d_field(
        data,
        cmap,
        vmin,
        vmax,
        distance_scale,
        background,
        width,
        height,
        alpha_func,
        gamma_correction,
        **kwargs,
    )
    im = _plot_3D_field(
        ax,
        img,
        bottom_label=bottom_label,
        left_label=left_label,
        title=title,
        title_loc=title_loc,
        aspect=aspect,
    )
    if return_cmap:
        return im, cmap
    return im

