import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mlp
import matplotlib.colors as colors
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Colormap, LinearSegmentedColormap, ListedColormap
import torch, os, copy
from .utils import default
from ._type import SpatialTensor
from typing import Union, Optional, Sequence, Tuple, Callable, Literal, Annotated
from mpl_toolkits.axes_grid1 import ImageGrid
from warnings import warn
from vape4d import render
from vape4d.utils import diverging_alpha, linear_increasing_alpha
from IPython.display import HTML


# "triagnle_wave" and "zigzag_alpha" functions are copied from exponax(https://github.com/Ceyron/exponax) exponax/exponax/viz/_volume.py
def triangle_wave(x, p):
    return 2 * np.abs(x / p - np.floor(x / p + 0.5))


def zigzag_alpha(cmap, min_alpha=0.0):
    """changes the alpha channel of a colormap to be linear (0->0, 1->1)

    Args:
        cmap (Colormap): colormap

    Returns:a
        Colormap: new colormap
    """
    if isinstance(cmap, ListedColormap):
        colors = copy.deepcopy(cmap.colors)
        for i, a in enumerate(colors):
            a.append(
                (triangle_wave(i / (cmap.N - 1), 0.5) * (1 - min_alpha)) + min_alpha
            )
        return ListedColormap(colors, cmap.name)
    elif isinstance(cmap, LinearSegmentedColormap):
        segmentdata = copy.deepcopy(cmap._segmentdata)
        segmentdata["alpha"] = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.25, 1.0, 1.0],
                [0.5, 0.0, 0.0],
                [0.75, 1.0, 1.0],
                [1.0, 0.0, 0.0],
            ]
        )
        return LinearSegmentedColormap(cmap.name, segmentdata)
    else:
        raise TypeError(
            "cmap must be either a ListedColormap or a LinearSegmentedColormap"
        )


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
    n_batch: int,
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
            if len(channel_names) > 1 and i_row == n_batch - 1
            else None
        )
        y_label = batch_names[i_row] if len(batch_names) > 1 and i_column == 0 else None
        data_i = None
    return data_i, x_label, y_label, i_column, i_row


def sym_colormap(d_min, d_max, d_cen=0, cmap="coolwarm", cmapname="sym_map"):
    """
    Generate a symmetric colormap.

    Args:
        d_min (float): The minimum value of the colormap.
        d_max (float): The maximum value of the colormap.
        d_cen (float, optional): The center value of the colormap. Defaults to 0.
        cmap (str, optional): The colormap to use. Defaults to "coolwarm".
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
    **kwargs
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
    **kwargs
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
        **kwargs
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


def _render(
    data: np.ndarray,
    cmap: Union[str, Colormap],
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    distance_scale: float = 10,
    background=(0, 0, 0, 0),
    width=512,
    height=512,
    alpha_func: Literal["zigzag", "diverging", "linear"] = "zigzag",
    gamma_correction: float = 2.4,
    **kwargs
):
    if isinstance(cmap, str):
        cmap = mlp.colormaps[cmap]
    if alpha_func == "zigzag":
        cmap = zigzag_alpha(cmap)
    elif alpha_func == "diverging":
        cmap = diverging_alpha(cmap)
    elif alpha_func == "linear":
        cmap = linear_increasing_alpha(cmap)
    else:
        raise ValueError(
            "The alpha function should be 'zigzag', 'diverging', or 'linear'."
        )
    img = render(
        data.astype(np.float32),  # expects float32
        cmap=cmap,  # zigzag alpha
        width=width,
        height=height,
        distance_scale=distance_scale,
        background=background,  # transparent background
        vmin=vmin,
        vmax=vmax,
        **kwargs
    )
    img = ((img / 255.0) ** (gamma_correction) * 255).astype(np.uint8)
    return img


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
    alpha_func: Literal["zigzag", "diverging", "linear"] = "zigzag",
    gamma_correction: float = 2.4,
    **kwargs
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
        alpha_func (str, optional): The alpha function. Defaults to "zigzag".
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
    img = _render(
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
        **kwargs
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


def plot_traj(
    traj: Union[
        SpatialTensor["B T C H ..."], Annotated[np.ndarray, "Spatial, B T C H ..."]
    ],
    channel_names: Optional[Sequence[str]] = None,
    batch_names: Optional[Sequence[str]] = None,
    vmin: Union[float, Sequence[Optional[float]]] = None,
    vmax: Union[float, Sequence[Optional[float]]] = None,
    subfig_size: float = 3.5,
    x_space: float = 0.7,
    y_space: float = 0.1,
    cbar_pad: float = 0.1,
    aspect: Literal["auto", "equal"] = "auto",
    num_colorbar_value: int = 4,
    ctick_format: Optional[str] = "%.1f",
    show_ticks: Union[Literal["auto"], bool] = "auto",
    show_time_index: bool = True,
    use_sym_colormap: bool = True,
    cmap: Union[str, Colormap] = "coolwarm",
    ticks_t: Tuple[Sequence[float], Sequence[str]] = None,
    ticks_x: Tuple[Sequence[float], Sequence[str]] = None,
    ticks_y: Tuple[Sequence[float], Sequence[str]] = None,
    ticks_z: Tuple[Sequence[float], Sequence[str]] = None,
    animation: bool = True,
    fps=30,
    show_in_notebook: bool = True,
    animation_engine: Literal["jshtml", "html5"] = "html5",
    save_name: Optional[str] = None,
    **kwargs
):
    """
    Plot a trajectory. The dimension of the trajectory can be 1D, 2D, or 3D.

    Args:
        traj (Union[SpatialTensor["B T C H ...], Annotated[np.ndarray, "Spatial, B T C H ..."]]): The trajectory to plot.
        channel_names (Optional[Sequence[str]], optional): The names of the channels. Defaults to None.
        batch_names (Optional[Sequence[str]], optional): The names of the batches. Defaults to None.
        vmin (Union[float, Sequence[Optional[float]]], optional): The minimum value for the color scale. Defaults to None.
        vmax (Union[float, Sequence[Optional[float]]], optional): The maximum value for the color scale. Defaults to None.
        subfig_size (float, optional): The size of the subfigures. Defaults to 3.5.
        x_space (float, optional): The space between subfigures in the x direction. Defaults to 0.7.
        y_space (float, optional): The space between subfigures in the y direction. Defaults to 0.1.
        cbar_pad (float, optional): The padding for the colorbar. Defaults to 0.1.
        aspect (Literal["auto", "equal"], optional): The aspect ratio. Defaults to "auto".
        num_colorbar_value (int, optional): The number of values for the colorbar. Defaults to 4.
        ctick_format (Optional[str], optional): The format for the colorbar ticks. Defaults to "%.1f".
        show_ticks (Union[Literal["auto"], bool], optional): Whether to show ticks. Defaults to "auto".
        show_time_index (bool, optional): Whether to show time index in the title. Defaults to True.
        use_sym_colormap (bool, optional): Whether to use a symmetric colormap. Defaults to True.
        cmap (Union[str, Colormap], optional): The colormap to use. Defaults to "coolwarm".
        ticks_t (Tuple[Sequence[float], Sequence[str]], optional): Custom ticks for the t-axis. Defaults to None.
        ticks_x (Tuple[Sequence[float], Sequence[str]], optional): Custom ticks for the x-axis. Defaults to None.
        ticks_y (Tuple[Sequence[float], Sequence[str]], optional): Custom ticks for the y-axis. Defaults to None.
        ticks_z (Tuple[Sequence[float], Sequence[str]], optional): Custom ticks for the z-axis. Defaults to None.
        animation (bool, optional): Whether to animate the plot. Defaults to True
            This only works for 1D and 2D data. If set to False, the 1d trajectory will be plotted as a 2D plot and the 2D trajectory will be plotted as a 3D plot.
        fps (int, optional): The frames per second for the animation. Defaults to 30.
        show_in_notebook (bool, optional): Whether to show the plot in a notebook. Defaults to True.
        animation_engine (Literal["jshtml", "html5"], optional): The engine for the animation. Defaults to "html5".
        save_name (Optional[str], optional): The name of the file to save the plot. Defaults to None.
        **kwargs: Additional keyword arguments for the plot.
    """

    if isinstance(traj, torch.Tensor):
        traj = traj.cpu().detach().numpy()
    n_batch, n_frame, n_channel = traj.shape[0], traj.shape[1], traj.shape[2]
    n_dim = len(traj.shape) - 3
    channel_names = default(
        channel_names, ["channel {}".format(i) for i in range(n_channel)]
    )
    batch_names = default(batch_names, ["batch {}".format(i) for i in range(n_batch)])
    if len(channel_names) != n_channel:
        raise ValueError(
            "The number of channel names should be equal to the number of channels in the input trajectory."
        )
    if len(batch_names) != n_batch:
        raise ValueError(
            "The number of batch names should be equal to the number of batches in the input trajectory."
        )
    vmins, vmaxs = _find_min_max(traj, vmin, vmax)
    if n_batch == 1:
        cbar_location = "right"
        cbar_mode = "each"
        ticklocation = "right"
    else:
        cbar_location = "top"
        cbar_mode = "edge"
        ticklocation = "top"
    cmaps = [
        sym_colormap(vmins[i], vmaxs[i], cmap=cmap) if use_sym_colormap else cmap
        for i in range(n_channel)
    ]
    if show_ticks == "auto":
        show_ticks = True if (n_dim == 1 and animation) else False
    subfig_h = subfig_size
    if n_dim == 1:
        if not animation:
            subfig_w = subfig_size * n_frame / traj.shape[-1]
        else:
            subfig_w = subfig_size * 2
            cbar_mode = None
    elif n_dim == 2:
        subfig_w = subfig_size * traj.shape[-2] / traj.shape[-1]
    elif n_dim == 3:
        h = traj.shape[-3] + traj.shape[-3]
        w = traj.shape[-2] + traj.shape[-1]
        subfig_w = subfig_size * w / h
        if (
            ticks_x is not None
            or ticks_y is not None
            or ticks_z is not None
            or show_ticks
        ):
            warn("Ticks are not supported for 3D trajectories.")
        cmaps = [diverging_alpha(cmap) for cmap in cmaps]
    else:
        raise ValueError("Only support 1D, 2D, and 3D trajectories.")
    fig = plt.figure(figsize=(subfig_w * n_channel, subfig_h * n_batch))
    # fig=plt.figure()
    grid = ImageGrid(
        fig,
        111,
        nrows_ncols=(n_batch, n_channel),
        axes_pad=(x_space, y_space),
        share_all=True,
        cbar_location=cbar_location,
        cbar_mode=cbar_mode,
        direction="row",
        cbar_pad=cbar_pad,
        aspect=False,
    )

    def set_colorbar():
        for i in range(n_channel):
            cb = grid.cbar_axes[i].colorbar(
                mlp.cm.ScalarMappable(
                    colors.Normalize(vmin=vmins[i], vmax=vmaxs[i]), cmap=cmaps[i]
                ),
                ticklocation=ticklocation,
                label=channel_names[i],
                format=ctick_format,
            )
            cb.ax.minorticks_on()
            cb.set_ticks(
                np.linspace(vmins[i], vmaxs[i], num_colorbar_value, endpoint=True)
            )

    def title_t(i):
        if show_time_index:
            if ticks_t is not None:
                if i in ticks_t[0]:
                    fig.suptitle("t={}".format(ticks_t[1][i]))
            else:
                fig.suptitle("t={}".format(i))

    if n_dim == 1:
        if animation:

            def ani_func(i):
                for j, ax_j in enumerate(grid):
                    ax_j.clear()
                    data_i, x_label, y_label, i_column, i_row = _data_plot(
                        j,
                        traj,
                        n_dim,
                        n_channel,
                        n_batch,
                        channel_names,
                        batch_names,
                        animation=animation,
                    )
                    plot_1D_field(
                        ax=ax_j,
                        data=data_i[i],
                        show_ticks=show_ticks,
                        x_label=x_label,
                        y_label=y_label,
                        ticks_x=ticks_t,
                        ticks_y=ticks_x,
                        vmin=vmins[i_column],
                        vmax=vmaxs[i_column],
                        **kwargs
                    )
                title_t(i)

        else:
            for i, ax_i in enumerate(grid):
                data_i, x_label, y_label, i_column, i_row = _data_plot(
                    i,
                    traj,
                    n_dim,
                    n_channel,
                    n_batch,
                    channel_names,
                    batch_names,
                    animation=animation,
                )
                plot_2D_field(
                    ax=ax_i,
                    data=data_i,
                    show_ticks=show_ticks,
                    x_label=x_label,
                    y_label=y_label,
                    cmap=cmaps[i_column],
                    vmin=vmins[i_column],
                    vmax=vmaxs[i_column],
                    ticks_x=ticks_t,
                    ticks_y=ticks_x,
                    aspect=aspect,
                    **kwargs
                )
            set_colorbar()
            if save_name is not None:
                plt.savefig(save_name)
            plt.show()
            return None
    elif n_dim == 2:
        if animation:

            def ani_func(i):
                for j, ax_j in enumerate(grid):
                    ax_j.clear()
                    data_j, x_label, y_label, j_column, j_row = _data_plot(
                        j,
                        traj,
                        n_dim,
                        n_channel,
                        n_batch,
                        channel_names,
                        batch_names,
                        animation=animation,
                    )
                    plot_2D_field(
                        ax=ax_j,
                        data=data_j[i],
                        show_ticks=show_ticks,
                        x_label=x_label,
                        y_label=y_label,
                        cmap=cmaps[j_column],
                        vmin=vmins[j_column],
                        vmax=vmaxs[j_column],
                        ticks_x=ticks_x,
                        ticks_y=ticks_y,
                        aspect=aspect,
                        **kwargs
                    )
                set_colorbar()
                title_t(i)

        else:
            for i, ax_i in enumerate(grid):
                data_i, x_label, y_label, i_column, i_row = _data_plot(
                    i,
                    traj,
                    n_dim,
                    n_channel,
                    n_batch,
                    channel_names,
                    batch_names,
                    animation=animation,
                )
                plot_3D_field(
                    ax=ax_i,
                    data=data_i,
                    bottom_label=x_label,
                    left_label=y_label,
                    aspect=aspect,
                    cmap=cmaps[i_column],
                    **kwargs
                )
            if save_name is not None:
                plt.savefig(save_name)
            set_colorbar()
            plt.show()
            return None
    elif n_dim == 3:
        imgs = []
        if n_frame == 1:
            t = [0, 1]
        else:
            t = np.linspace(0, 1, n_frame)
        for b in range(n_batch):
            for c in range(n_channel):
                imgs.append(
                    _render(
                        traj[b, :, c, ...].astype(np.float32),
                        cmaps[c],
                        time=t,
                        **kwargs
                    )
                )

        def ani_func(i):
            for j, ax_j in enumerate(grid):
                ax_j.clear()
                _, x_label, y_label, i_column, i_row = _data_plot(
                    j, traj, n_dim, n_channel, n_batch, channel_names, batch_names
                )
                _plot_3D_field(
                    ax_j,
                    imgs[j][i],
                    bottom_label=x_label,
                    left_label=y_label,
                    aspect=aspect,
                    **kwargs
                )
            title_t(i)
            set_colorbar()

    if n_frame != 1:
        ani = FuncAnimation(
            fig, ani_func, frames=n_frame, repeat=False, interval=1000 / fps
        )
        if show_in_notebook:
            plt.close()
            if animation_engine == "jshtml":
                return HTML(ani.to_jshtml())
            elif animation_engine == "html5":
                try:
                    return HTML(ani.to_html5_video())
                except Exception as e:
                    warn_msg = (
                        "Error occurs when generating html5 video, use jshtml instead."
                        + os.linesep
                    )
                    warn_msg += "Error message: {}".format(e) + os.linesep
                    warn_msg += (
                        "This is probably due to the `ffmpeg` is not properly installed."
                        + os.linesep
                    )
                    warn_msg += "Please install `ffmpeg` and try again." + os.linesep
                    warn(warn_msg)
                    return HTML(ani.to_jshtml())
            else:
                raise ValueError("The animation engine should be 'jshtml' or 'html5'.")
        else:
            return ani
    else:
        ani_func(0)
        if save_name is not None:
            plt.savefig(save_name)
        plt.show()


def plot_field(
    field: Union[
        SpatialTensor["B C H ..."], Annotated[np.ndarray, "Spatial, B C H ..."]
    ],
    channel_names: Optional[Sequence[str]] = None,
    batch_names: Optional[Sequence[str]] = None,
    vmin: Union[float, Sequence[Optional[float]]] = None,
    vmax: Union[float, Sequence[Optional[float]]] = None,
    subfig_size: float = 3.5,
    x_space: float = 0.7,
    y_space: float = 0.1,
    cbar_pad: float = 0.1,
    aspect: Literal["auto", "equal"] = "auto",
    num_colorbar_value: int = 4,
    ctick_format: Optional[str] = "%.1f",
    show_ticks: Union[Literal["auto"], bool] = "auto",
    use_sym_colormap: bool = True,
    cmap: Union[str, Colormap] = "coolwarm",
    ticks_x: Tuple[Sequence[float], Sequence[str]] = None,
    ticks_y: Tuple[Sequence[float], Sequence[str]] = None,
    ticks_z: Tuple[Sequence[float], Sequence[str]] = None,
    save_name: Optional[str] = None,
    **kwargs
):
    """
    Plot a field. The dimension of the field can be 1D, 2D, or 3D.

    Args:
        field (Union[SpatialTensor["B C H ...], Annotated[np.ndarray, "Spatial, B C H ..."]]): The field to plot.
        channel_names (Optional[Sequence[str]], optional): The names of the channels. Defaults to None.
        batch_names (Optional[Sequence[str]], optional): The names of the batches. Defaults to None.
        vmin (Union[float, Sequence[Optional[float]]], optional): The minimum value for the color scale. Defaults to None.
        vmax (Union[float, Sequence[Optional[float]]], optional): The maximum value for the color scale. Defaults to None.
        subfig_size (float, optional): The size of the subfigures. Defaults to 3.5.
        x_space (float, optional): The space between subfigures in the x direction. Defaults to 0.7.
        y_space (float, optional): The space between subfigures in the y direction. Defaults to 0.1.
        cbar_pad (float, optional): The padding for the colorbar. Defaults to 0.1.
        aspect (Literal["auto", "equal"], optional): The aspect ratio. Defaults to "auto".
        num_colorbar_value (int, optional): The number of values for the colorbar. Defaults to 4.
        ctick_format (Optional[str], optional): The format for the colorbar ticks. Defaults to "%.1f".
        show_ticks (Union[Literal["auto"], bool], optional): Whether to show ticks. Defaults to "auto".
        use_sym_colormap (bool, optional): Whether to use a symmetric colormap. Defaults to True.
        cmap (Union[str, Colormap], optional): The colormap to use. Defaults to "coolwarm".
        ticks_x (Tuple[Sequence[float], Sequence[str]], optional): Custom ticks for the x-axis. Defaults to None.
        ticks_y (Tuple[Sequence[float], Sequence[str]], optional): Custom ticks for the y-axis. Defaults to None.
        ticks_z (Tuple[Sequence[float], Sequence[str]], optional): Custom ticks for the z-axis. Defaults to None.
        save_name (Optional[str], optional): The name of the file to save the plot. Defaults to None.
        **kwargs: Additional keyword arguments for the plot.
    """

    if isinstance(field, torch.Tensor):
        field = field.cpu().detach().numpy()
    field = np.expand_dims(field, 1)
    plot_traj(
        field,
        channel_names=channel_names,
        batch_names=batch_names,
        vmin=vmin,
        vmax=vmax,
        subfig_size=subfig_size,
        x_space=x_space,
        y_space=y_space,
        cbar_pad=cbar_pad,
        aspect=aspect,
        num_colorbar_value=num_colorbar_value,
        ctick_format=ctick_format,
        show_ticks=show_ticks,
        use_sym_colormap=use_sym_colormap,
        cmap=cmap,
        ticks_x=ticks_x,
        ticks_y=ticks_y,
        ticks_z=ticks_z,
        animation=True,
        show_time_index=False,
        save_name=save_name,
        **kwargs
    )
