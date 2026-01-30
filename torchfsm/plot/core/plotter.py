import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mlp
import matplotlib.colors as colors
import torch, os
from ..._utils import default
from ..._type import SpatialTensor, SpatialArray
from ..render import AlphaFunction, rendering_cmap, render_3d_field
from ..tools import sym_colormap
from .field import plot_1D_field, plot_2D_field, plot_3D_field, _plot_3D_field
from typing import Union, Optional, Sequence, Tuple, Literal, Callable
from warnings import warn
from IPython.display import HTML
from functools import partial
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Colormap


class ChannelWisedPlotter:
    """
    A plotter that plots trajectories with channel-wise data arrangement.

    In channel-wise plotting, each column represents a different channel,
    and each row represents a different batch.
    """

    def min_max(
        self,
        traj: np.ndarray,
        vmin: Union[float, Sequence[Optional[float]]],
        vmax: Union[float, Sequence[Optional[float]]],
        universal_minmax: bool = False,
    ):
        """
        Calculate the minimum and maximum values for the trajectory.

        Args:
            traj (np.ndarray): The trajectory data.
            vmin (Union[float, Sequence[Optional[float]]]): Minimum values for color scale.
            vmax (Union[float, Sequence[Optional[float]]]): Maximum values for color scale.
            universal_minmax (bool): Whether to use universal min-max across all channels.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Arrays of minimum and maximum values.
        """
        axis = tuple([0, 1] + [i + 3 for i in range(len(traj.shape) - 3)])
        vmins = np.min(traj, axis=axis)
        vmaxs = np.max(traj, axis=axis)
        if vmin is not None:
            if isinstance(vmin, float) or isinstance(vmin, int):
                vmin = [vmin] * len(vmins)
            elif len(vmin) != len(vmins):
                raise ValueError(
                    "The number of vmin values should be equal to the number of columns of the plot."
                )
            vmins = np.asarray(
                [
                    vmin[i] if vmin[i] is not None else vmins[i]
                    for i in range(len(vmins))
                ]
            )
        if vmax is not None:
            if isinstance(vmax, float) or isinstance(vmax, int):
                vmax = [vmax] * len(vmaxs)
            elif len(vmax) != len(vmaxs):
                raise ValueError(
                    "The number of vmax values should be equal to the number of columns of the plot."
                )
            vmaxs = np.asarray(
                [
                    vmax[i] if vmax[i] is not None else vmaxs[i]
                    for i in range(len(vmaxs))
                ]
            )
        if universal_minmax:
            vmin = np.min(vmins)
            vmax = np.max(vmaxs)
            vmins = np.full_like(vmins, vmin)
            vmaxs = np.full_like(vmaxs, vmax)
        return vmins, vmaxs

    def setup_colorbar_location(
        self, batch_size, universal_minmax=False, rotate_cbar_for_single_batch=True
    ):
        """
        Setup the location and mode of the colorbar.

        Args:
            batch_size (int): Number of batches in the data.
            universal_minmax (bool): Whether using universal min-max values.
            rotate_cbar_for_single_batch (bool): Whether to rotate colorbar for single batch.

        Returns:
            Tuple[str, str, str]: Colorbar location, mode, and tick location.
        """
        if batch_size == 1 and rotate_cbar_for_single_batch:
            cbar_location = "right"
            cbar_mode = "each"
            ticklocation = "right"
        else:
            cbar_location = "top"
            cbar_mode = "edge"
            ticklocation = "top"
        if universal_minmax:
            cbar_mode = "single"
        return cbar_location, cbar_mode, ticklocation

    def _select_values(self, values, i_column, i_row):
        """
        Select values based on column and row indices.

        Args:
            values: Array of values to select from.
            i_column (int): Column index.
            i_row (int): Row index.

        Returns:
            Selected value.
        """
        return values[i_column]

    def setup_pad_size(self, space_x, space_y, cbar_pad):
        """
        Setup padding sizes for the plot layout.

        Args:
            space_x: Horizontal spacing between subplots.
            space_y: Vertical spacing between subplots.
            cbar_pad: Padding for colorbar.

        Returns:
            Tuple: Processed spacing values.
        """
        return default(space_x, 0.7), default(space_y, 0.1), default(cbar_pad, 0.1)

    def select_vmin(
        self,
        i_column,
        i_row,
        vmins,
    ):
        """
        Select minimum value for a specific subplot.

        Args:
            i_column (int): Column index.
            i_row (int): Row index.
            vmins: Array of minimum values.

        Returns:
            Selected minimum value.
        """
        return self._select_values(vmins, i_column, i_row)

    def select_vmax(self, i_column, i_row, vmaxs):
        """
        Select maximum value for a specific subplot.

        Args:
            i_column (int): Column index.
            i_row (int): Row index.
            vmaxs: Array of maximum values.

        Returns:
            Selected maximum value.
        """
        return self._select_values(vmaxs, i_column, i_row)

    def select_cmap(self, i_column, i_row, cmaps):
        """
        Select colormap for a specific subplot.

        Args:
            i_column (int): Column index.
            i_row (int): Row index.
            cmaps: Array of colormaps.

        Returns:
            Selected colormap.
        """
        return self._select_values(cmaps, i_column, i_row)

    def select_data(
        self,
        i: int,
        fields: np.ndarray,
        n_dim: int,
        n_channel: int,
        batch_size: int,
        channel_names: Sequence[str],
        batch_names: Sequence[str],
        animation: bool = True,
        label_x: Optional[str] = None,
        label_y: Optional[str] = None,
        label_t: Optional[str] = None,
        hide_batch_channel_name_for_single_plot: bool = True,
    ):
        """
        Select and prepare data for a specific subplot.

        Args:
            i (int): Linear index of the subplot.
            fields (np.ndarray): The trajectory data.
            n_dim (int): Number of spatial dimensions.
            n_channel (int): Number of channels.
            batch_size (int): Number of batches.
            channel_names (Sequence[str]): Names of channels.
            batch_names (Sequence[str]): Names of batches.
            animation (bool): Whether creating animation.
            label_x (Optional[str]): Label for x-axis.
            label_y (Optional[str]): Label for y-axis.
            label_t (Optional[str]): Label for time axis.
            hide_batch_channel_name_for_single_plot (bool): Whether to hide batch name for single batch.

        Returns:
            Tuple: Data, x_label, y_label, column_index, row_index.
        """
        i_row = i // n_channel
        i_column = i % n_channel

        if n_dim == 1:
            if animation:
                y_label = (
                    batch_names[i_row] + os.linesep + "value"
                    if len(batch_names) > 1
                    or not hide_batch_channel_name_for_single_plot
                    else "value"
                )
                x_label = (
                    label_x + os.linesep + channel_names[i_column]
                    if len(channel_names) > 1
                    or not hide_batch_channel_name_for_single_plot
                    else label_x
                )
                data_i = fields[i_row, :, i_column, :]
            else:
                y_label = label_x
                if len(batch_names) > 1 or not hide_batch_channel_name_for_single_plot:
                    y_label = batch_names[i_row] + os.linesep + y_label
                x_label = label_t
                if len(channel_names) > 1:
                    x_label = channel_names[i_column] + os.linesep + x_label
                data_i = fields[i_row, :, i_column, :]
        if n_dim == 2:
            if animation:
                y_label = label_y
                if len(batch_names) > 1 or not hide_batch_channel_name_for_single_plot:
                    y_label = batch_names[i_row] + os.linesep + y_label
                if len(channel_names) > 1:
                    x_label = label_x + os.linesep + channel_names[i_column]
                else:
                    x_label = label_x
                data_i = fields[i_row, :, i_column, ...]
            else:
                x_label = channel_names[i_column] if len(channel_names) > 1 else None
                y_label = (
                    batch_names[i_row]
                    if len(batch_names) > 1
                    or not hide_batch_channel_name_for_single_plot
                    else None
                )
                data_i = fields[i_row, :, i_column, ...]
        elif n_dim == 3:
            x_label = (
                channel_names[i_column]
                if len(channel_names) > 1 and i_row == batch_size - 1
                else None
            )
            y_label = (
                batch_names[i_row]
                if (len(batch_names) > 1 or not hide_batch_channel_name_for_single_plot)
                and i_column == 0
                else None
            )
            data_i = None
        return data_i, x_label, y_label, i_column, i_row

    def prepare_cmaps(
        self,
        cmap,
        vmins,
        vmaxs,
        use_sym_colormap: bool,
        n_dim: int,
        alpha_func,
        animation: bool,
    ):
        """
        Prepare colormaps for plotting.

        Args:
            cmap: Base colormap.
            vmins: Minimum values.
            vmaxs: Maximum values.
            use_sym_colormap (bool): Whether to use symmetric colormap.
            n_dim (int): Number of spatial dimensions.
            alpha_func: Alpha function for 3D rendering.
            animation (bool): Whether creating animation.

        Returns:
            List of prepared colormaps.
        """
        cmap = mlp.colormaps[cmap] if isinstance(cmap, str) else cmap
        cmaps = [
            (
                sym_colormap(vmins[i], vmaxs[i], cmap=cmap, cmapname=f"sym_cmap_{i}")
                if use_sym_colormap
                else cmap
            )
            for i in range(len(vmins))
        ]
        if n_dim == 3 or (n_dim == 2 and not animation):
            cmaps = [rendering_cmap(c, alpha_func) for c in cmaps]
        return cmaps

    def traj_info(
        self,
        traj: Union[SpatialTensor["B T C H ..."], SpatialArray["B T C H ..."]],
    ):
        """
        Extract trajectory information.

        Args:
            traj: The trajectory data.

        Returns:
            Tuple: n_dim, batch_size, n_frame, n_channel, x_shape, y_shape, z_shape.
        """
        n_dim = len(traj.shape) - 3
        batch_size = traj.shape[0]
        n_frame = traj.shape[1]
        n_channel = traj.shape[2]
        x_shape = traj.shape[3]
        y_shape = traj.shape[4] if n_dim > 1 else None
        z_shape = traj.shape[5] if n_dim > 2 else None
        return n_dim, batch_size, n_frame, n_channel, x_shape, y_shape, z_shape

    def plot_size(
        self,
        subfig_size,
        traj,
        animation,
        batch_size,
        n_channel,
        space_x,
        space_y,
        width_correction,
        height_correction,
        cbar_mode,
        cbar_location,
        cbar_pad,
    ):
        """
        Calculate the total plot size.

        Args:
            subfig_size (float): Size of each subplot.
            traj: Trajectory data.
            animation (bool): Whether creating animation.
            batch_size (int): Number of batches.
            n_channel (int): Number of channels.
            space_x: Horizontal spacing.
            space_y: Vertical spacing.

        Returns:
            Tuple[float, float]: Total width and height.
        """
        n_dim, batch_size, n_frame, n_channel, x_shape, y_shape, z_shape = (
            self.traj_info(traj)
        )
        subfig_h = subfig_size
        if n_dim == 1:
            if not animation:
                subfig_w = subfig_size * n_frame / x_shape
            else:
                subfig_w = subfig_size * 2
        elif n_dim == 2:
            if not animation:
                w = (n_frame + y_shape) * 0.784
                h = x_shape / 1.4142 + max(n_frame, y_shape) * 0.555
                subfig_w = subfig_size * w / h
            else:
                subfig_w = subfig_size * y_shape / x_shape
        elif n_dim == 3:
            w = (x_shape + z_shape) * 0.784
            h = y_shape / 1.4142 + max(x_shape, z_shape) * 0.555
            subfig_w = subfig_size * w / h
        else:
            raise ValueError("Only support 1D, 2D, and 3D trajectories.")
        subfig_w = max(
            subfig_w, subfig_size / 4
        )
        total_height = subfig_h * batch_size
        total_height += space_y * (batch_size - 1)
        total_width = subfig_w * n_channel
        total_width += space_x * (n_channel - 1)
        total_width *= width_correction
        total_height *= height_correction
        # the size calculation with colorbar is still very confusing when using ImageGrid
        # need to be improved in the future
        if cbar_mode == "single" and cbar_location == "right":
            c_bar_size = f"{0.15/total_height*100}%"
            total_width = 1.15*total_width + cbar_pad
        elif cbar_mode == "single" and cbar_location == "top":
            c_bar_size = f"{0.15/total_width*100}%"
            total_height = 1.15*total_height + cbar_pad
        else:
            c_bar_size = "5%"
            if cbar_location == "top":
                total_height = total_height + 0.05*subfig_h + cbar_pad
            elif cbar_location == "right":
                total_width = (
                    total_width + (0.05 * batch_size*subfig_w) + cbar_pad * n_channel
                )
        return total_width, total_height, c_bar_size

    def setup_colorbar(
        self,
        vmins,
        vmaxs,
        grid,
        cmaps,
        ticklocation,
        ctick_format,
        num_colorbar_value,
        c_bar_labels,
        universal_minmax=False,
    ):
        """
        Setup colorbars for the plot.

        Args:
            vmins: Minimum values.
            vmaxs: Maximum values.
            grid: ImageGrid object.
            cmaps: Colormaps.
            ticklocation (str): Location of ticks.
            ctick_format (str): Format for colorbar ticks.
            num_colorbar_value (int): Number of colorbar values.
            c_bar_labels: Labels for colorbars.
            universal_minmax (bool): Whether using universal min-max.
        """
        if universal_minmax:
            vmins = vmins[0:1]
            vmaxs = vmaxs[0:1]
        if c_bar_labels is None:
            c_bar_labels = [None] * len(vmins)
        elif len(c_bar_labels) != len(vmins):
            raise ValueError(
                f"The number of colorbar labels {len(c_bar_labels)} should be equal to the number of colorbars {len(vmins)} of the plot."
            )
        for i in range(len(vmins)):
            grid.cbar_axes[i].clear()
            cb = grid.cbar_axes[i].colorbar(
                mlp.cm.ScalarMappable(
                    colors.Normalize(vmin=vmins[i], vmax=vmaxs[i]), cmap=cmaps[i]
                ),
                ticklocation=ticklocation,
                label=c_bar_labels[i],
                format=ctick_format,
            )
            cb.ax.minorticks_on()
            cb.set_ticks(
                np.linspace(vmins[i], vmaxs[i], num_colorbar_value, endpoint=True)
            )

    def add_text_plot(
        self, text_ax, string, location: Literal["center", "top", "bottom"] = "center"
    ):
        """
        Add text to a text axis.

        Args:
            text_ax: Matplotlib axis for text.
            string (str): Text string to display.
        """
        text_ax.clear()
        if string is not None:
            if location == "top":
                y = 1.0
            elif location == "bottom":
                y = 0.0
            elif location == "center":
                y = 0.5
            else:
                raise ValueError("The location should be 'center', 'top', or 'bottom'.")
            text_ax.text(
                0.5,
                y,
                string,
                transform=text_ax.transAxes,  # Coordinates relative to the axes (0,0 bottom-left, 1,1 top-right)
                horizontalalignment="center",
                fontsize=14,
                verticalalignment="center",
            )
        text_ax.set_xticks([])
        text_ax.set_yticks([])
        text_ax.spines["top"].set_visible(False)
        text_ax.spines["right"].set_visible(False)
        text_ax.spines["bottom"].set_visible(False)
        text_ax.spines["left"].set_visible(False)

    def title_t(self, i, title, ticks_t, show_time_index, ax_t, ax_title, label_t):
        """
        Update title and time index display.

        Args:
            i (int): Current time index.
            title (str): Plot title.
            ticks_t: Time ticks.
            show_time_index (bool): Whether to show time index.
            ax_t: Axis for time display.
            ax_title: Axis for title display.
            label_t (str): Label for time.
        """
        if ax_t is not None:
            if show_time_index:
                if ticks_t is not None:
                    if i in ticks_t[0]:
                        self.add_text_plot(ax_t, ticks_t[1][i])
                else:
                    self.add_text_plot(ax_t, f"{label_t}={i}", location="top")
            else:
                self.add_text_plot(ax_t, None)
        elif show_time_index:
            warn(
                "It seems you require showing time index, but axis for time index is None. "
                + "Please provide `rect_t_label` to the `plot` function."
            )
        if ax_title is not None:
            self.add_text_plot(ax_title, title, location="bottom")
        elif title is not None:
            warn(
                "It seems you require showing title, but axis for title is None. "
                + "Please provide `rect_title` to the `plot` function."
            )

    def build_animation(
        self,
        fig: plt.Figure,
        ani_func: Callable[[int], None],
        n_frame: int,
        fps: int,
        show_in_notebook: bool = True,
        animation_engine: Literal["jshtml", "html5"] = "html5",
    ):
        """
        Build animation from the plot.

        Args:
            fig (plt.Figure): Matplotlib figure.
            ani_func (Callable[[int], None]): Animation function.
            n_frame (int): Number of frames.
            fps (int): Frames per second.
            show_in_notebook (bool): Whether to show in notebook.
            animation_engine (Literal["jshtml", "html5"]): Animation engine.

        Returns:
            Animation object or HTML display.
        """
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

    def prefix_names(self, name: str, n: int, space: str = " "):
        """
        Generate prefixed names for batches or channels.

        Args:
            name (str): Base name.
            n (int): Number of names to generate.
            space (str): Separator between name and index.

        Returns:
            List[str]: Generated names.
        """
        return [f"{name}{space}{i}" for i in range(n)]

    def create_figure(
        self,
        total_width,
        total_height,
        title=None,
        show_time_index=False,
        rect_t_label=None,
        rect_title=None,
    ):
        if rect_t_label is not None or rect_title is not None:
            raise ValueError(
                "If `rect_t_label` or `rect_title` is provided, `fig` should not be None."
            )
        # the pad and height for title and time label are hard coded for now
        # need to be improved in the future
        pad_left = 0.5 / total_width
        pad_top = 0.2 / total_height
        height_title = 1.0
        height_t_label = 0.5
        total_width += pad_left * 2
        total_height += pad_top * 2
        if title is None and not show_time_index:
            fig = plt.figure(figsize=(total_width, total_height))
            rect_t_label = None
            rect_title = None
            gs = fig.add_gridspec(
                1,
                1,
                hspace=0,
                top=1.0 - pad_top,
                bottom=pad_top,
                left=pad_left,
                right=1.0 - pad_left,
            )
            rect_plot = gs[0]
        elif title is not None and not show_time_index:
            fig = plt.figure(figsize=(total_width, total_height + height_title))
            rect_t_label = None
            gs = fig.add_gridspec(
                2,
                1,
                height_ratios=[
                    1 - height_title / total_height,
                    height_title / total_height,
                ],
                hspace=0,
                top=1.0 - pad_top,
                bottom=pad_top,
                left=pad_left,
                right=1.0 - pad_left,
            )
            rect_plot = gs[0]
            rect_title = gs[1]
        elif title is None and show_time_index:
            fig = plt.figure(figsize=(total_width, total_height + height_t_label))
            rect_title = None
            gs = fig.add_gridspec(
                2,
                1,
                height_ratios=[
                    height_t_label / total_height,
                    1 - height_t_label / total_height,
                ],
                hspace=0,
                top=1.0 - pad_top,
                bottom=pad_top,
                left=pad_left,
                right=1.0 - pad_left,
            )
            rect_t_label = gs[0]
            rect_plot = gs[1]
        else:
            fig = plt.figure(
                figsize=(total_width, total_height + height_t_label + height_title)
            )
            gs = fig.add_gridspec(
                3,
                1,
                height_ratios=[
                    height_t_label / total_height,
                    1 - height_t_label / total_height - height_title / total_height,
                    height_title / total_height,
                ],
                hspace=0,
                top=1.0 - pad_top,
                bottom=pad_top,
                left=pad_left,
                right=1.0 - pad_left,
            )
            rect_t_label = gs[0]
            rect_plot = gs[1]
            rect_title = gs[2]
        return fig, rect_plot, rect_t_label, rect_title

    def plot(
        self,
        traj: Union[SpatialTensor["B T C H ..."], SpatialArray["B T C H ..."]],
        fig: Optional[plt.Figure] = None,
        rect_plot: Optional[Union[Tuple[int, int, int, int]]] = None,
        rect_t_label: Optional[Union[Tuple[int, int, int, int]]] = None,
        rect_title: Optional[Union[Tuple[int, int, int, int]]] = None,
        channel_names: Optional[Sequence[str]] = None,
        batch_names: Optional[Sequence[str]] = None,
        hide_batch_channel_name_for_single_plot: bool = True,
        title: Optional[str] = None,
        vmin: Optional[Union[float, Sequence[float]]] = None,
        vmax: Optional[Union[float, Sequence[float]]] = None,
        cmap: Union[str, Colormap] = "twilight",
        use_sym_colormap: bool = False,
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
        universal_minmax: bool = False,
        num_colorbar_value: int = 4,
        c_bar_labels: Optional[Sequence[str]] = None,
        cbar_pad: Optional[float] = 0.1,
        ctick_format: Optional[str] = "%.1f",
        rotate_cbar_for_single_batch: bool = True,
        subfig_size: float = 2.5,
        real_size_ratio: bool = False,
        width_correction: float = 1.0,
        height_correction: float = 1.0,
        space_x: Optional[float] = 0.7,
        space_y: Optional[float] = 0.1,
        label_x: Optional[str] = "x",
        label_y: Optional[str] = "y",
        label_t: Optional[str] = "t",
        ticks_t: Tuple[Sequence[float], Sequence[str]] = None,
        ticks_x: Tuple[Sequence[float], Sequence[str]] = None,
        ticks_y: Tuple[Sequence[float], Sequence[str]] = None,
        show_ticks: Union[Literal["auto"], bool] = "auto",
        show_time_index: bool = True,
        animation: bool = True,
        fps=30,
        show_in_notebook: bool = True,
        animation_engine: Literal["jshtml", "html5"] = "html5",
        return_ani_func: bool = False,
        save_name: Optional[str] = None,
        show_plot: bool = True,
        show_3d_coordinates: bool = True,
        **kwargs,
    ) -> Optional[FuncAnimation]:
        """
        Plot a trajectory. The dimension of the trajectory can be 1D, 2D, or 3D.

        Args:
            traj (Union[SpatialTensor["B T C H ..."], SpatialArray["B T C H ..."]]): The trajectory to plot.
            fig (Optional[plt.Figure], optional): The figure to plot on. If None, a new figure will be created. Defaults to None.
            channel_names (Optional[Sequence[str]], optional): The names of the channels. Defaults to None.
            batch_names (Optional[Sequence[str]], optional): The names of the batches. Defaults to None.
            hide_batch_channel_name_for_single_plot (bool, optional): Whether to hide the batch and channel name for a single batch or channel. Defaults to True.
            vmin (Optional[Union[float, Sequence[float]]], optional): The minimum value for the color scale. Defaults to None.
            vmax (Optional[Union[float, Sequence[float]]], optional): The maximum value for the color scale. Defaults to None.
            cmap (Union[str, Colormap], optional): The colormap to use. Defaults to "twilight".
            use_sym_colormap (bool, optional): Whether to use a symmetric colormap. Defaults to False.
            alpha_func (Union[Literal["zigzag", "central_peak", "central_valley", "linear_increase", "linear_decrease"], AlphaFunction], optional): The alpha function to use for 3D rendering. Defaults to "zigzag".
            universal_minmax (bool, optional): Whether to use a universal min-max for the color scale. Defaults to False.
            num_colorbar_value (int, optional): The number of values to show on the colorbar. Defaults to 4.
            c_bar_labels (Optional[Sequence[str]], optional): The labels for the colorbar. Defaults to None.
            cbar_pad (Optional[float], optional): The padding for the colorbar. Defaults to 0.1.
            ctick_format (Optional[str], optional): The format for the colorbar ticks. Defaults to "%.1f".
            rotate_cbar_for_single_batch (bool, optional): Whether to rotate the colorbar for a single colorbar. Defaults to True.
            subfig_size (float, optional): The size of each subplot. Defaults to 2.5.
            real_size_ratio (bool, optional): Whether to use the real size ratio for the subfigures. Defaults to False.
            width_correction (float, optional): The correction factor for the width of the subfigures. Defaults to 1.0.
            height_correction (float, optional): The correction factor for the height of the subfigures. Defaults to 1.0.
            space_x (Optional[float], optional): The space between subfigures in the x direction. Defaults to 0.7.
            space_y (Optional[float], optional): The space between subfigures in the y direction. Defaults to 0.1.
            label_x (Optional[str], optional): The label for the x axis. Defaults to "x".
            label_y (Optional[str], optional): The label for the y axis. Defaults to "y".
            label_t (Optional[str], optional): The label for the time axis. Defaults to "t".
            ticks_t (Tuple[Sequence[float], Sequence[str]], optional): The ticks for the time axis. Defaults to None.
            ticks_x (Tuple[Sequence[float], Sequence[str]], optional): The ticks for the x axis. Defaults to None.
            ticks_y (Tuple[Sequence[float], Sequence[str]], optional): The ticks for the y axis. Defaults to None.
            show_ticks (Union[Literal["auto"], bool], optional): Whether to show ticks on the axes. Defaults to "auto".
            show_time_index (bool, optional): Whether to show the time index in the title. Defaults to True.
            animation (bool, optional): Whether to create an animation. Defaults to True.
            fps (int, optional): The frames per second for the animation. Defaults to 30.
            show_in_notebook (bool, optional): Whether to show the plot in a Jupyter notebook. Defaults to True.
            animation_engine (Literal["jshtml", "html5"], optional): The engine to use for the animation. Defaults to "html5".
            return_ani_func (bool, optional): Whether to return the animation function. Defaults to False.
            save_name (Optional[str], optional): The name of the file to save the plot. Defaults to None.
            show_plot (bool, optional): Whether to show the plot. Defaults to True.
            show_3d_coordinates (bool, optional): Whether to show 3D coordinates for 3D plots. Defaults to True.
            **kwargs: Additional keyword arguments to pass to the plotting functions.
        Returns:
            Optional[FuncAnimation]: If `animation` is True and not show_in_notebook, returns a `FuncAnimation` object.
        """
        if isinstance(traj, torch.Tensor):
            traj = traj.cpu().detach().numpy()
        n_dim, batch_size, n_frame, n_channel, x_shape, y_shape, z_shape = (
            self.traj_info(traj)
        )
        if n_dim not in [1, 2, 3]:
            raise ValueError(
                "Only support 1D, 2D, and 3D trajectories. Currently got {}D.".format(
                    n_dim
                )
            )
        channel_names = default(channel_names, self.prefix_names("channel", n_channel))
        batch_names = default(batch_names, self.prefix_names("batch", batch_size))
        if len(channel_names) != n_channel:
            raise ValueError(
                f"The number of channel names {channel_names} should be equal to the number of channels {n_channel} in the input trajectory."
            )
        if len(batch_names) != batch_size:
            raise ValueError(
                f"The number of batch names {batch_names} should be equal to the number of batches {batch_size} in the input trajectory."
            )
        vmins, vmaxs = self.min_max(traj, vmin, vmax, universal_minmax)
        cbar_location, cbar_mode, ticklocation = self.setup_colorbar_location(
            batch_size, universal_minmax, rotate_cbar_for_single_batch
        )
        cmaps = self.prepare_cmaps(
            cmap, vmins, vmaxs, use_sym_colormap, n_dim, alpha_func, animation
        )
        if show_ticks == "auto":
            show_ticks = True if (n_dim == 1 and animation) else False
        if n_dim == 1 and animation:
            cbar_mode = None
        if n_dim == 3 and (ticks_x is not None or ticks_y is not None or show_ticks):
            warn("Ticks are not supported for 3D trajectories.")
        if not animation:
            show_time_index = False
        space_x, space_y, cbar_pad = self.setup_pad_size(space_x, space_y, cbar_pad)
        total_width, total_height, c_bar_size = self.plot_size(
            subfig_size,
            traj,
            animation,
            batch_size,
            n_channel,
            space_x,
            space_y,
            width_correction,
            height_correction,
            cbar_mode,
            cbar_location,
            cbar_pad,
        )
        if fig is None:
            fig, rect_plot, rect_t_label, rect_title = self.create_figure(
                total_width,
                total_height,
                title,
                show_time_index,
                rect_t_label,
                rect_title,
            )
        if rect_t_label is not None:
            ax_t_label = fig.add_subplot(rect_t_label)
        else:
            ax_t_label = None
        if rect_title is not None:
            ax_title = fig.add_subplot(rect_title)
        else:
            ax_title = None

        grid = ImageGrid(
            fig,
            rect=rect_plot,
            nrows_ncols=(batch_size, n_channel),
            axes_pad=(space_x, space_y),
            share_all=True,
            cbar_location=cbar_location,
            cbar_mode=cbar_mode,
            direction="row",
            cbar_pad=cbar_pad,
            aspect=real_size_ratio,
            cbar_size=c_bar_size,
        )
        set_colorbar = partial(
            self.setup_colorbar,
            vmins=vmins,
            vmaxs=vmaxs,
            grid=grid,
            cmaps=cmaps,
            ticklocation=ticklocation,
            ctick_format=ctick_format,
            num_colorbar_value=num_colorbar_value,
            c_bar_labels=c_bar_labels,
        )
        title_t = partial(
            self.title_t,
            title=title,
            ticks_t=ticks_t,
            show_time_index=show_time_index,
            ax_t=ax_t_label,
            ax_title=ax_title,
            label_t=label_t,
        )
        select_vmin = partial(self.select_vmin, vmins=vmins)
        select_vmax = partial(self.select_vmax, vmaxs=vmaxs)
        select_cmap = partial(self.select_cmap, cmaps=cmaps)
        select_data = partial(
            self.select_data,
            fields=traj,
            n_dim=n_dim,
            n_channel=n_channel,
            batch_size=batch_size,
            channel_names=channel_names,
            batch_names=batch_names,
            label_x=label_x,
            label_y=label_y,
            label_t=label_t,
            hide_batch_channel_name_for_single_plot=hide_batch_channel_name_for_single_plot,
            animation=animation,
        )
        if n_dim == 1:
            if animation:

                def ani_func(i):
                    for j, ax_j in enumerate(grid):
                        ax_j.clear()
                        data_i, x_label, y_label, i_column, i_row = select_data(j)
                        plot_1D_field(
                            ax=ax_j,
                            data=data_i[i],
                            show_ticks=show_ticks,
                            x_label=x_label,
                            y_label=y_label,
                            ticks_x=ticks_t,
                            ticks_y=ticks_x,
                            vmin=select_vmin(i_column, i_row),
                            vmax=select_vmax(i_column, i_row),
                            **kwargs,
                        )
                    title_t(i)

            else:
                for i, ax_i in enumerate(grid):
                    data_i, x_label, y_label, i_column, i_row = select_data(i)
                    plot_2D_field(
                        ax=ax_i,
                        data=data_i,
                        show_ticks=show_ticks,
                        x_label=x_label,
                        y_label=y_label,
                        cmap=select_cmap(i_column, i_row),
                        vmin=select_vmin(i_column, i_row),
                        vmax=select_vmax(i_column, i_row),
                        ticks_x=ticks_t,
                        ticks_y=ticks_x,
                        **kwargs,
                    )
                set_colorbar()
                title_t(0)
                if save_name is not None:
                    plt.savefig(save_name, bbox_inches="tight")
                if show_plot:
                    plt.show()
                return None
        elif n_dim == 2:
            if animation:

                def ani_func(i):
                    for j, ax_j in enumerate(grid):
                        ax_j.clear()
                        data_j, x_label, y_label, j_column, j_row = select_data(j)
                        plot_2D_field(
                            ax=ax_j,
                            data=data_j[i],
                            show_ticks=show_ticks,
                            x_label=x_label,
                            y_label=y_label,
                            cmap=select_cmap(j_column, j_row),
                            vmin=select_vmin(j_column, j_row),
                            vmax=select_vmax(j_column, j_row),
                            ticks_x=ticks_x,
                            ticks_y=ticks_y,
                            **kwargs,
                        )
                    set_colorbar()
                    title_t(i)

            else:
                for i, ax_i in enumerate(grid):
                    data_i, x_label, y_label, i_column, i_row = select_data(i)
                    plot_3D_field(
                        ax=ax_i,
                        data=data_i,
                        cmap=select_cmap(i_column, i_row),
                        bottom_label=x_label,
                        left_label=y_label,
                        show_3d_coordinates=(
                            show_3d_coordinates if i == len(grid) - 1 else False
                        ),
                        x_arrow_label="t",
                        y_arrow_label="x",
                        z_arrow_label="y",
                        **kwargs,
                    )
                title_t(0)
                if save_name is not None:
                    plt.savefig(save_name, bbox_inches="tight")
                set_colorbar()
                if show_plot:
                    plt.show()
                return None
        elif n_dim == 3:
            imgs = []
            if n_frame == 1:
                t = [0, 1]
            else:
                t = np.linspace(0, 1, n_frame)
            for b in range(batch_size):
                for c in range(n_channel):
                    imgs.append(
                        render_3d_field(
                            traj[b, :, c, ...].astype(np.float32),
                            select_cmap(c, b),
                            time=t,
                            alpha_func=alpha_func,
                            vmin=select_vmin(c, b),
                            vmax=select_vmax(c, b),
                            **kwargs,
                        )
                    )

            def ani_func(i):
                for j, ax_j in enumerate(grid):
                    ax_j.clear()
                    _, x_label, y_label, i_column, i_row = select_data(j)
                    _plot_3D_field(
                        ax_j,
                        imgs[j][i],
                        bottom_label=x_label,
                        left_label=y_label,
                        show_3d_coordinates=(
                            show_3d_coordinates
                            if (j == len(grid) - 1 and i == 0)
                            else False
                        ),
                        **kwargs,
                    )
                title_t(i)
                set_colorbar()

        if n_frame != 1:
            if not show_in_notebook and return_ani_func:
                return ani_func
            if show_in_notebook and return_ani_func:
                warn(
                    "The `return_ani_func` argument is ignored when `show_in_notebook` is True. The animation will be displayed in the notebook."
                )
            return self.build_animation(
                fig,
                ani_func,
                n_frame,
                fps,
                show_in_notebook=show_in_notebook,
                animation_engine=animation_engine,
            )
        else:
            ani_func(0)
            if save_name is not None:
                plt.savefig(save_name, bbox_inches="tight")
            if show_plot:
                plt.show()


class BatchWisedPlotter(ChannelWisedPlotter):
    """
    A plotter that plots trajectories with batch-wise data.
    It inherits from ChannelWisedPlotter and overrides some methods to handle batch-wise data.
    """

    def min_max(
        self,
        traj: np.ndarray,
        vmin: Union[float, Sequence[Optional[float]]],
        vmax: Union[float, Sequence[Optional[float]]],
        universal_minmax: bool = False,
    ):
        """
        Calculate the minimum and maximum values for the trajectory.

        Args:
            traj (np.ndarray): The trajectory data.
            vmin (Union[float, Sequence[Optional[float]]]): Minimum values for color scale.
            vmax (Union[float, Sequence[Optional[float]]]): Maximum values for color scale.
            universal_minmax (bool): Whether to use universal min-max across all channels.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Arrays of minimum and maximum values.
        """
        axis = tuple([1, 2] + [i + 3 for i in range(len(traj.shape) - 3)])
        vmins = np.min(traj, axis=axis)
        vmaxs = np.max(traj, axis=axis)
        if vmin is not None:
            if isinstance(vmin, float) or isinstance(vmin, int):
                vmin = [vmin] * len(vmins)
            elif len(vmin) != len(vmins):
                raise ValueError(
                    "The number of vmin values should be equal to the number of rows of the plot."
                )
            vmins = np.asarray(
                [
                    vmin[i] if vmin[i] is not None else vmins[i]
                    for i in range(len(vmins))
                ]
            )
        if vmax is not None:
            if isinstance(vmax, float) or isinstance(vmax, int):
                vmax = [vmax] * len(vmaxs)
            elif len(vmax) != len(vmaxs):
                raise ValueError(
                    "The number of vmax values should be equal to the number of rows of the plot."
                )
            vmaxs = np.asarray(
                [
                    vmax[i] if vmax[i] is not None else vmaxs[i]
                    for i in range(len(vmaxs))
                ]
            )
        if universal_minmax:
            vmin = np.min(vmins)
            vmax = np.max(vmaxs)
            vmins = np.full_like(vmins, vmin)
            vmaxs = np.full_like(vmaxs, vmax)
        return vmins, vmaxs

    def setup_colorbar_location(
        self, batch_size, universal_minmax, rotate_cbar_for_single_batch
    ):
        """
        Setup the location and mode of the colorbar.

        Args:
            batch_size (int): Number of batches in the data.
            universal_minmax (bool): Whether using universal min-max values.
            rotate_cbar_for_single_batch (bool): Whether to rotate colorbar for single batch.

        Returns:
            Tuple[str, str, str]: Colorbar location, mode, and tick location.
        """
        if batch_size == 1 and rotate_cbar_for_single_batch:
            warn(
                "The `rotate_cbar_for_single_batch` argument is ignored when `BatchWisedPlotter` is used."
            )
        cbar_location = "right"
        cbar_mode = "edge"
        ticklocation = "right"
        if universal_minmax:
            cbar_mode = "single"
        return cbar_location, cbar_mode, ticklocation

    def _select_values(self, values, i_column, i_row):
        """
        Select values based on column and row indices.

        Args:
            values: Array of values to select from.
            i_column (int): Column index.
            i_row (int): Row index.

        Returns:
            Selected value.
        """
        return values[i_row]

    def setup_pad_size(self, space_x, space_y, cbar_pad):
        """
        Setup padding sizes for the plot layout.

        Args:
            space_x: Horizontal spacing between subplots.
            space_y: Vertical spacing between subplots.
            cbar_pad: Padding for colorbar.

        Returns:
            Tuple: Processed spacing values.
        """
        return default(space_x, 0.2), default(space_y, 0.2), default(cbar_pad, 0.1)
