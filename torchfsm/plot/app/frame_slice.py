from ..._type import SpatialTensor, SpatialArray
from ..._utils import default
from ...utils.slice import traj_slices
from ..render import AlphaFunction
from .frame import _plot_traj_frame_group
from .slice import _get_slice_names
from typing import Union, Optional, Sequence, Tuple, Literal
from matplotlib.colors import Colormap


def plot_traj_frame_slice(
    traj: Union[SpatialTensor["B T C H ..."], SpatialArray["B T C H ..."]],
    slice_control: Sequence[Optional[Union[int, float]]] = None,
    n_frames: int = 5,
    channel_names: Optional[Sequence[str]] = None,
    batch_names: Optional[Sequence[str]] = None,
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
    num_colorbar_value: int = 4,
    c_bar_labels: Optional[Sequence[str]] = None,
    cbar_pad: Optional[float] = 0.1,
    ctick_format: Optional[str] = "%.1f",
    subfig_size: float = 2.5,
    real_size_ratio: bool = False,
    width_correction: float = 1.0,
    height_correction: float = 1.0,
    space_x: Optional[float] = 0.7,
    space_y: Optional[float] = 0.1,
    label_x: Optional[str] = "x",
    label_y: Optional[str] = "y",
    label_z: Optional[str] = "z",
    label_t: Optional[str] = "t",
    ticks_x: Tuple[Sequence[float], Sequence[str]] = None,
    ticks_y: Tuple[Sequence[float], Sequence[str]] = None,
    ticks_z: Tuple[Sequence[float], Sequence[str]] = None,
    show_ticks: Union[Literal["auto"], bool] = "auto",
    save_name: Optional[str] = None,
    compare_mode: Literal[
        "t_wised", "channel_wised", "channel_wised_universal"
    ] = "channel_wised_universal",
    frame_start_index: int = 0,
    **kwargs,
):
    """
    Plot the trajectory slices.

    Args:
        traj (Union[SpatialTensor["B C H ..."], SpatialArray["B C H ..."]]): The trajectory to plot.
        slice_control (Sequence[Optional[Union[int, float]]], optional): The control points for slicing the trajectory. Defaults to None.
            If None, it will slice at the middle of each dimension.
        n_frames (int): The number of frames to plot.
        channel_names (Optional[Sequence[str]], optional): The names of the channels. Defaults to None.
        batch_names (Optional[Sequence[str]], optional): The names of the batches. Defaults to None.
        title (Optional[str], optional): The title of the plot. Defaults to None.
        vmin (Optional[Union[float, Sequence[float]]], optional): The minimum value for the color scale. Defaults to None. If a sequence is provided, it should have the same length as the number of channels.
        vmax (Optional[Union[float, Sequence[float]]], optional): The maximum value for the color scale. Defaults to None. If a sequence is provided, it should have the same length as the number of channels.
        cmap (Union[str, Colormap], optional): The colormap to use. Defaults to "twilight".
        use_sym_colormap (bool, optional): Whether to use a symmetric colormap. Defaults to False.
        alpha_func (Union[Literal["zigzag","central_peak","central_valley","linear_increase","linear_decrease",],AlphaFunction,], optional): The alpha function for the colormap when plot 3D data. Defaults to "zigzag".
        num_colorbar_value (int, optional): The number of values for the colorbar. Defaults to 4.
        c_bar_labels (Optional[Sequence[str]], optional): The labels for the colorbar. Defaults to None.
            If provided, it should have the same length as the number of channels.
            If not provided, the colorbar will not have labels.
        cbar_pad (Optional[float], optional): The padding for the colorbar. Defaults to 0.1.
        ctick_format (Optional[str], optional): The format for the colorbar ticks. Defaults to "%.1f".
        subfig_size (float, optional): The size of the subfigures. Defaults to 2.5.
        real_size_ratio (bool, optional): Whether to use the real size ratio for the subfigures. Defaults to False.
        width_correction (float, optional): The correction factor for the width of the subfigures. Defaults to 1.0.
        height_correction (float, optional): The correction factor for the height of the subfigures. Defaults to 1.0.
        space_x (Optional[float], optional): The space between subfigures in the x direction. Defaults to 0.7.
        space_y (Optional[float], optional): The space between subfigures in the y direction. Defaults to 0.1.
        label_x (Optional[str], optional): The label for the x-axis. Defaults to "x".
        label_y (Optional[str], optional): The label for the y-axis. Defaults to "y".
        label_t (Optional[str], optional): The label for the time index. Defaults to "t".
        ticks_t (Tuple[Sequence[float], Sequence[str]], optional): Custom ticks for the time index. Defaults to None.
        ticks_x (Tuple[Sequence[float], Sequence[str]], optional): Custom ticks for the x-axis. Defaults to None.
        ticks_y (Tuple[Sequence[float], Sequence[str]], optional): Custom ticks for the y-axis. Defaults to None.
        show_ticks (Union[Literal["auto"], bool], optional): Whether to show ticks. Defaults to "auto".
        save_name (Optional[str], optional): The name of the file to save the plot. Defaults to None.
        compare_mode (Literal["t_wised", "channel_wised", "channel_wised_universal"], optional): The mode to compare the data. Defaults to "channel_wised_universal".
            - "t_wised": Compare the data across time.
            - "channel_wised": Compare the data across channels.
            - "channel_wised_universal": Compare the data across channels with universal min-max scaling
        frame_start_index (int): The starting index for the frame numbers. Defaults to 0.
    Returns:
        FuncAnimation: If `animation` is True and not show_in_notebook, returns a `FuncAnimation` object.
    """
    n_dim = len(traj.shape) - 3
    if n_dim != 2 and n_dim != 3:
        raise ValueError(
            f"Trajectory must have 2 or 3 spatial dimensions, but got {n_dim}."
        )
    if traj.shape[1] < n_frames:
        raise ValueError(
            f"Trajectory has only {traj.shape[1]} frames, but {n_frames} are requested."
        )
    # make_slices:
    if slice_control is None:
        slice_control = [0.5] * n_dim
    slices = traj_slices(traj, slice_control)
    slice_names, label_xs, label_ys, ticks_xs, ticks_ys = _get_slice_names(
        slice_control=slice_control,
        label_x=label_x,
        label_y=label_y,
        label_z=label_z,
        ticks_x=ticks_x,
        ticks_y=ticks_y,
        ticks_z=ticks_z,
    )
    batch_names = default(batch_names, [f"batch {i}" for i in range(traj.shape[0])])
    batch_names = [
        [f"{batch_names[j]}, {slice_names[i]}" for j in range(len(batch_names))]
        for i in range(len(slice_names))
    ]
    return _plot_traj_frame_group(
        trajs=slices,
        n_frames=n_frames,
        channel_names=channel_names,
        batch_names=batch_names,
        title=title,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        use_sym_colormap=use_sym_colormap,
        alpha_func=alpha_func,
        num_colorbar_value=num_colorbar_value,
        c_bar_labels=c_bar_labels,
        cbar_pad=cbar_pad,
        ctick_format=ctick_format,
        subfig_size=subfig_size,
        real_size_ratio=real_size_ratio,
        width_correction=width_correction,
        height_correction=height_correction,
        space_x=space_x,
        space_y=space_y,
        label_x=label_xs,
        label_y=label_ys,
        label_t=label_t,
        ticks_x=ticks_xs,
        ticks_y=ticks_ys,
        show_ticks=show_ticks,
        save_name=save_name,
        compare_mode=compare_mode,
        frame_start_index=frame_start_index,
        **kwargs,
    )
