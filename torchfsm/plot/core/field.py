import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap
import torch
from ..render import AlphaFunction, render_3d_field, add_3d_coord
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
    cmap: Union[str, Colormap] = "twilight",
    show_ticks=True,
    ticks_x: Tuple[Sequence[float], Sequence[str]] = None,
    ticks_y: Tuple[Sequence[float], Sequence[str]] = None,
    rasterized: bool = True,
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
        cmap (Union[str, Colormap], optional): The colormap to use. Defaults to "twilight".
        show_ticks (bool, optional): Whether to show ticks. Defaults to True.
        ticks_x (Tuple[Sequence[float], Sequence[str]], optional): Custom ticks for the x-axis. Defaults to None.
        ticks_y (Tuple[Sequence[float], Sequence[str]], optional): Custom ticks for the y-axis. Defaults to None.
        rasterized (bool, optional): Whether to rasterize the image. Defaults to True.
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
        rasterized=rasterized,
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
    show_3d_coordinates: bool = True,
    coordinates_size: float = 0.1,
    coordinates_x_loc: float = -0.05,
    coordinates_y_loc: float = -0.05,
    arrow_length: float = 0.6,
    x_arrow_label:str='x',
    y_arrow_label:str='y',
    z_arrow_label:str='z',
    x_arrow_color:str='r',
    y_arrow_color:str='g',
    z_arrow_color:str='b',
    arrow_length_ratio:float=0.25,
    arrow_linewidth:int=1,
    rasterized: bool = True,
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
        show_3d_coordinates (bool, optional): Whether to show 3D coordinates. Defaults to True.
                ax2d (mlp.axes.Axes): The 2D axes to which the 3D coordinate system will be added
        coordinates_size (float, optional): Size of the 3D axes relative to the 2D axes. Defaults to 0.1.
        coordinates_x_loc (float, optional): X location offset for the 3D axes. Defaults to -0.05.
        coordinates_y_loc (float, optional): Y location offset for the 3D axes. Defaults to -0.05.
        arrow_length (float, optional): Length of the arrows. Defaults to 0.6.
        x_arrow_label (str, optional): Label for the X axis. Defaults to 'x'.
        y_arrow_label (str, optional): Label for the Y axis. Defaults to 'y'.
        z_arrow_label (str, optional): Label for the Z axis. Defaults to 'z'.
        x_arrow_color (str, optional): Color for the X axis arrow. Defaults to 'r'.
        y_arrow_color (str, optional): Color for the Y axis arrow. Defaults to 'g'.
        z_arrow_color (str, optional): Color for the Z axis arrow. Defaults to 'b'.
        arrow_length_ratio (float, optional): Ratio of the arrow head length to the total arrow length. Defaults to 0.25.
        arrow_linewidth (int, optional): Line width of the arrows. Defaults to 1.
        rasterized (bool, optional): Whether to rasterize the image. Defaults to True.
    """
    im = ax.imshow(img, aspect=aspect,rasterized=rasterized)
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
    if show_3d_coordinates:
        add_3d_coord(ax,  
                     size=coordinates_size, 
                     length=arrow_length, 
                     x_loc=coordinates_x_loc,
                     y_loc=coordinates_y_loc,
                     x_label=x_arrow_label,
                     y_label=y_arrow_label,
                     z_label=z_arrow_label,
                     x_color=x_arrow_color,
                     y_color=y_arrow_color,
                     z_color=z_arrow_color,
                     arrow_length_ratio=arrow_length_ratio,
                     linewidth=arrow_linewidth)
    return im


def plot_3D_field(
    ax: plt.Axes,
    data: Union[np.ndarray, torch.Tensor],
    bottom_label: Optional[str] = None,
    left_label: Optional[str] = None,
    title: Optional[str] = None,
    title_loc="center",
    aspect="auto",
    cmap: Union[str, Colormap] = "twilight",
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
    show_3d_coordinates: bool = True,
    coordinates_size: float = 0.1,
    coordinates_x_loc: float = -0.05,
    coordinates_y_loc: float = -0.05,
    arrow_length: float = 0.6,
    x_arrow_label:str='x',
    y_arrow_label:str='y',
    z_arrow_label:str='z',
    x_arrow_color:str='r',
    y_arrow_color:str='g',
    z_arrow_color:str='b',
    arrow_length_ratio:float=0.25,
    arrow_linewidth:int=1,
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
        cmap (Union[str, Colormap], optional): The colormap to use. Defaults to "twilight".
        vmin (Optional[float], optional): The minimum value for the color scale. Defaults to None.
        vmax (Optional[float], optional): The maximum value for the color scale. Defaults to None.
        return_cmap (bool, optional): Whether to return the colormap. Defaults to False.
        distance_scale (float, optional): The distance scale for rendering. Defaults to 10.
        background (tuple, optional): The background color. Defaults to (0, 0, 0, 0).
        width (int, optional): The width of the rendered image. Defaults to 512.
        height (int, optional): The height of the rendered image. Defaults to 512.
        alpha_func (Union[Literal["zigzag","central_peak","central_valley","linear_increase","linear_decrease", "luminance",],AlphaFunction,], optional): The alpha function. Defaults to "zigzag".
        gamma_correction (float, optional): The gamma correction factor. Defaults to 2.4.
        show_3d_coordinates (bool, optional): Whether to show 3D coordinates. Defaults to False.
                ax2d (mlp.axes.Axes): The 2D axes to which the 3D coordinate system will be added
        coordinates_size (float, optional): Size of the 3D axes relative to the 2D axes. Defaults to 0.1.
        coordinates_x_loc (float, optional): X location offset for the 3D axes. Defaults to -0.05.
        coordinates_y_loc (float, optional): Y location offset for the 3D axes. Defaults to -0.05.
        arrow_length (float, optional): Length of the arrows. Defaults to 0.6.
        x_arrow_label (str, optional): Label for the X axis. Defaults to 'x'.
        y_arrow_label (str, optional): Label for the Y axis. Defaults to 'y'.
        z_arrow_label (str, optional): Label for the Z axis. Defaults to 'z'.
        x_arrow_color (str, optional): Color for the X axis arrow. Defaults to 'r'.
        y_arrow_color (str, optional): Color for the Y axis arrow. Defaults to 'g'.
        z_arrow_color (str, optional): Color for the Z axis arrow. Defaults to 'b'.
        arrow_length_ratio (float, optional): Ratio of the arrow head length to the total arrow length. Defaults to 0.25.
        arrow_linewidth (int, optional): Line width of the arrows. Defaults to 1.
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
        show_3d_coordinates=show_3d_coordinates,
        coordinates_size=coordinates_size,
        coordinates_x_loc=coordinates_x_loc,
        coordinates_y_loc=coordinates_y_loc,
        arrow_length=arrow_length,
        x_arrow_label=x_arrow_label,
        y_arrow_label=y_arrow_label,
        z_arrow_label=z_arrow_label,
        x_arrow_color=x_arrow_color,
        y_arrow_color=y_arrow_color,
        z_arrow_color=z_arrow_color,
        arrow_length_ratio=arrow_length_ratio,
        arrow_linewidth=arrow_linewidth,
    )
    if return_cmap:
        return im, cmap
    return im

