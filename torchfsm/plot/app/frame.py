from ..core import (
    ChannelWisedPlotter,
    BatchWisedPlotter,
    concate_fields_plot,
)
from ..._type import SpatialTensor, SpatialArray, ValueList
from ...utils.traj_manipulate import uniformly_select_frames
from ..._utils import default
from ..render import AlphaFunction
from typing import Union, Optional, Sequence, Tuple, Literal
from matplotlib.colors import Colormap
import torch
import numpy as np

def plot_traj_frame(
    traj: Union[SpatialTensor["B T C H ..."], SpatialArray["B T C H ..."]],
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
    label_t: Optional[str] = "t",
    ticks_x: Tuple[Sequence[float], Sequence[str]] = None,
    ticks_y: Tuple[Sequence[float], Sequence[str]] = None,
    show_ticks: Union[Literal["auto"], bool] = "auto",
    save_name: Optional[str] = None,
    compare_mode: Literal[
        "t_wised", "channel_wised", "channel_wised_universal"
    ] = "channel_wised_universal",
    frame_start_index: int = 0,
    show_3d_coordinates: bool = True,
    **kwargs,
):
    """
    Plot frames of a single trajectory. The dimension of the trajectory can be 1D, 2D, or 3D.

    Args:
        traj (Union[SpatialTensor["B C H ..."], SpatialArray["B C H ..."]]): The trajectory to plot.
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
        ticks_x (Tuple[Sequence[float], Sequence[str]], optional): Custom ticks for the x-axis. Defaults to None.
        ticks_y (Tuple[Sequence[float], Sequence[str]], optional): Custom ticks for the y-axis. Defaults to None.
        show_ticks (Union[Literal["auto"], bool], optional): Whether to show ticks. Defaults to "auto".
        frame_start_index: (int): The starting index for the frame numbers. Defaults to 0.
        show_3d_coordinates (bool, optional): Whether to show 3D coordinates for 3D plots. Defaults to True.
        **kwargs: Additional keyword arguments for the plot.
    """
    if isinstance(traj, torch.Tensor):
        traj = traj.cpu().detach().numpy()
    if traj.ndim < 4:
        raise ValueError("Trajectory must have at least 4 dimensions (B, T, C, H, ...)")
    if traj.shape[1] < n_frames:
        raise ValueError(
            f"Trajectory has only {traj.shape[1]} frames, but {n_frames} are requested."
        )
    frames, frame_indices = uniformly_select_frames(traj, n_frames, True)
    channel_wised_data = []
    for c_i in range(frames.shape[2]):
        temp = []
        for b_i in range(frames.shape[0]):
            temp.append(frames[b_i, :, c_i, ...])  # T, H, W, ...
        channel_wised_data.append(np.stack(temp, axis=0))  # [B, T, H, W, ...]
    if compare_mode == "t_wised":
        ploter = ChannelWisedPlotter()
        space_x = default(space_x, 0.7)
        space_y = default(space_y, 0.1)
        universal_minmax = False
    elif compare_mode == "channel_wised" or compare_mode == "channel_wised_universal":
        ploter = BatchWisedPlotter()
        space_x = default(space_x, 0.2)
        space_y = default(space_y, 0.2)
        if compare_mode == "channel_wised_universal":
            universal_minmax = True
    else:
        raise ValueError(
            f"Unknown compare_mode: {compare_mode}. Must be 't_wised', 'channel_wised' or 'channel_wised_universal'."
        )
    time_names = [f"{label_t}={i + frame_start_index}" for i in frame_indices]
    channel_names = default(
        channel_names, [f"channel {i}" for i in range(len(channel_wised_data))]
    )
    batch_names = default(
        batch_names, [f"batch {i} " for i in range(channel_wised_data[0].shape[0])]
    )
    batch_names = [
        [f"{batch_names[j]}, {channel_names[i]}" for j in range(len(batch_names))]
        for i in range(len(channel_names))
    ]
    return concate_fields_plot(
        ploters=ploter,
        fields=channel_wised_data,
        channel_names=time_names,
        batch_names=batch_names,
        vmin=vmin,
        vmax=vmax,
        universal_minmax=universal_minmax,
        subfig_size=subfig_size,
        space_x=space_x,
        space_y=space_y,
        cbar_pad=cbar_pad,
        c_bar_labels=c_bar_labels,
        real_size_ratio=real_size_ratio,
        num_colorbar_value=num_colorbar_value,
        ctick_format=ctick_format,
        show_ticks=show_ticks,
        cmap=cmap,
        use_sym_colormap=use_sym_colormap,
        ticks_x=ticks_x,
        ticks_y=ticks_y,
        save_name=save_name,
        alpha_func=alpha_func,
        show_time_index=False,
        title=title,
        width_correction=width_correction,
        height_correction=height_correction,
        label_x=label_x,
        label_y=label_y,
        label_t=label_t,
        show_3d_coordinates=show_3d_coordinates,
        **kwargs,
    )

def _plot_traj_frame_group(
    trajs: Sequence[Union[SpatialTensor["B T C H ..."], SpatialArray["B T C H ..."]]],
    n_frames: int = 5,
    channel_names: Optional[Sequence[str]] = None,
    batch_names: Optional[ValueList[Sequence[str]]] = None,
    hide_batch_channel_name_for_single_plot: ValueList[bool] = True,
    title: Optional[str] = None,
    vmin: Optional[ValueList[Sequence[Union[float, Sequence[float]]]]] = None,
    vmax: Optional[ValueList[Sequence[Union[float, Sequence[float]]]]] = None,
    cmap: ValueList[Union[str, Colormap]] = "twilight",
    use_sym_colormap: ValueList[bool] = False,
    alpha_func: ValueList[
        Union[
            Literal[
                "zigzag",
                "central_peak",
                "central_valley",
                "linear_increase",
                "linear_decrease",
                "luminance",
            ],
            AlphaFunction,
        ]
    ] = "zigzag",
    universal_minmax: ValueList[bool] = False,
    num_colorbar_value: int = 4,
    c_bar_labels: Optional[ValueList[Sequence[str]]] = None,
    cbar_pad: Optional[float] = 0.1,
    ctick_format: Optional[str] = "%.1f",
    rotate_cbar_for_single_batch: ValueList[bool] = True,
    subfig_size: float = 2.5,
    real_size_ratio: bool = False,
    width_correction: float = 1.0,
    height_correction: float = 1.0,
    space_x: float = 0.7,
    space_y: float = 0.1,
    label_x: ValueList[str] = "x",
    label_y: ValueList[str] = "y",
    label_t: str = "t",
    ticks_x: ValueList[Tuple[Sequence[float], Sequence[str]]] = None,
    ticks_y: ValueList[Tuple[Sequence[float], Sequence[str]]] = None,
    show_ticks: Union[Literal["auto"], bool] = "auto",
    save_name: Optional[str] = None,
    compare_mode: Literal[
        "t_wised", "channel_wised", "channel_wised_universal"
    ] = "channel_wised_universal",
    frame_start_index: int = 0,
    show_3d_coordinates: bool = True,
    **kwargs,
):
    """
    Plot multiple trajectories with different plotters. This function is a wrapper for the `plot` method of `ChannelWisedPlotter` and `BatchWisedPlotter`.

    Args:
        trajs (Sequence[Union[SpatialTensor["B T C H ..."], SpatialArray["B T C H ..."]]]): A list of trajectories to plot. Each trajectory should be a `SpatialTensor` or `SpatialArray` with shape `(B, T, C, H, ...)`,
            where `B` is the batch size, `T` is the number of time steps, `C` is the number of channels, and `H` is the height (and possibly width or depth) of the spatial dimensions.
        n_frames (int): The number of frames to plot for each trajectory.
        channel_names (Optional[Sequence[str]], optional): The names of the channels. Defaults to None.
        batch_names (Optional[ValueList[Sequence[str]]], optional): The names of the batches. Defaults to None.
        hide_batch_channel_name_for_single_plot (ValueList[bool], optional): Whether to hide the batch and channel name for a single batch or channel. If a single value is provided, it will be used for all trajectories. If a list is provided, it should have the same length as `trajs`. Defaults to True.
        title (Optional[str], optional): The title of the plot. Defaults to None.
        vmin (Optional[ValueList[Sequence[Union[float, Sequence[float]]]]], optional): The minimum value for the color scale. If a single value is provided, it will be used for all trajectories. If a list is provided, it should have the same length as `trajs`. Defaults to None.
        vmax (Optional[ValueList[Sequence[Union[float, Sequence[float]]]], optional): The maximum value for the color scale. If a single value is provided, it will be used for all trajectories. If a list is provided, it should have the same length as `trajs`. Defaults to None.
        cmap (ValueList[Union[str, Colormap]], optional): The colormap to use for the plot. If a single value is provided, it will be used for all trajectories. If a list is provided, it should have the same length as `trajs`. Defaults to "twilight".
        use_sym_colormap (ValueList[bool], optional): Whether to use a symmetric colormap. If a single value is provided, it will be used for all trajectories. If a list is provided, it should have the same length as `trajs`. Defaults to False.
        alpha_func (ValueList[Union[Literal["zigzag", "central_peak", "central_valley", "linear_increase", "linear_decrease", "luminance"], AlphaFunction]], optional): The alpha function to use for 3D rendering. If a single value is provided, it will be used for all trajectories. If a list is provided, it should have the same length as `trajs`. Defaults to "zigzag".
        universal_minmax (ValueList[bool], optional): Whether to use a universal min-max for the color scale. If a single value is provided, it will be used for all trajectories. If a list is provided, it should have the same length as `trajs`. Defaults to False.
        num_colorbar_value (int, optional): The number of values to show on the colorbar. Defaults to 4.
        c_bar_labels (Optional[ValueList[Sequence[str]]], optional): The labels for the colorbar. If a single value is provided, it will be used for all trajectories. If a list is provided, it should have the same length as `trajs`. Defaults to None.
        cbar_pad (Optional[float], optional): The padding for the colorbar. Defaults to 0.1.
        ctick_format (Optional[str], optional): The format for the colorbar ticks. Defaults to "%.1f".
        subfig_size (float, optional): The size of the subfigures. Defaults to 2.5.
        real_size_ratio (bool, optional): Whether to use the real size ratio for the subfigures. Defaults to False.
        width_correction (float, optional): The correction factor for the width of the subfigures. Defaults to 1.0.
        height_correction (float, optional): The correction factor for the height of the subfigures. Defaults to 1.0.
        space_x (float, optional): The space between subfigures in the x direction. Defaults to 0.7.
        space_y (float, optional): The space between subfigures in the y direction. Defaults to 0.1.
        label_x (ValueList[str], optional): The label for the x axis. If a single value is provided, it will be used for all trajectories. If a list is provided, it should have the same length as `trajs`. Defaults to "x".
        label_y (ValueList[str], optional): The label for the y axis. If a single value is provided, it will be used for all trajectories. If a list is provided, it should have the same length as `trajs`. Defaults to "y".
        label_t (str, optional): The label for the time axis. Defaults to "t".
        ticks_x (ValueList[Tuple[Sequence[float], Sequence[str]]], optional): The ticks for the x axis. If a single value is provided, it will be used for all trajectories. If a list is provided, it should have the same length as `trajs`. Defaults to None.
        ticks_y (ValueList[Tuple[Sequence[float], Sequence[str]]], optional): The ticks for the y axis. If a single value is provided, it will be used for all trajectories. If a list is provided, it should have the same length as `trajs`. Defaults to None.
        show_ticks (Union[Literal["auto"], bool], optional): Whether to show ticks on the axes. If set to "auto", ticks will be shown for 1D trajectories with animation. Defaults to "auto".
        save_name (Optional[str], optional): The name of the file to save the plot. Defaults to None.
        compare_mode: Literal[
            "t_wised", "channel_wised", "channel_wised_universal"
        ] = "channel_wised_universal",
        frame_start_index (int, optional): The starting index for the time axis. Defaults to 0.
        show_3d_coordinates (bool, optional): Whether to show 3D coordinates for 3D plots. Defaults to True.
        **kwargs: Additional keyword arguments to pass to the plotting functions.
    """
    n_channel = trajs[0].shape[2]

    def check_value_list(value_list):
        n = len(trajs)
        if not isinstance(value_list, Sequence) or isinstance(value_list, str):
            value_list = [value_list] * n
        elif len(value_list) != n:
            raise ValueError(
                f"The length of the value list {value_list} should be equal to the number of trajectories {n}."
            )
        return value_list * n_channel

    vmin = check_value_list(vmin)
    vmax = check_value_list(vmax)
    c_bar_labels = check_value_list(c_bar_labels)
    use_sym_colormap = check_value_list(use_sym_colormap)
    cmap = check_value_list(cmap)
    ticks_x = check_value_list(ticks_x)
    ticks_y = check_value_list(ticks_y)
    alpha_func = check_value_list(alpha_func)
    universal_minmax = check_value_list(universal_minmax)
    label_x = check_value_list(label_x)
    label_y = check_value_list(label_y)
    rotate_cbar_for_single_batch = check_value_list(rotate_cbar_for_single_batch)
    hide_batch_channel_name_for_single_plot = check_value_list(
        hide_batch_channel_name_for_single_plot
    )
    channel_names = default(channel_names, [f"channel {i}" for i in range(n_channel)])
    if batch_names is None:
        batch_names = []
        for traj in trajs:
            batch_names.append([f"batch {i}" for i in range(traj.shape[0])])
    full_batch_names = []
    for i in range(n_channel):
        for group_i in batch_names:
            full_batch_names.append(
                [batch_name_i + ", " + channel_names[i] for batch_name_i in group_i]
            )
    channel_wised_data = []
    for c_i in range(n_channel):
        for traj in trajs:
            if traj.ndim < 4:
                raise ValueError(
                    "Trajectory must have at least 4 dimensions (B, T, C, H, ...)"
                )
            if traj.shape[1] < n_frames:
                raise ValueError(
                    f"Trajectory has only {traj.shape[1]} frames, but {n_frames} are requested."
                )
            if traj.shape[2] != n_channel:
                raise ValueError(
                    f"All trajectories must have the same number of channels. Expected {n_channel}, but got {traj.shape[2]}."
                )
            frames, frame_indices = uniformly_select_frames(
                traj[:, :, c_i, ...], n_frames, True
            )
            channel_wised_data.append(frames)  # [B, T, H, W, ...]
    time_names = [f"{label_t}={i + frame_start_index}" for i in frame_indices]
    if compare_mode == "t_wised":
        ploter = ChannelWisedPlotter()
        space_x = default(space_x, 0.7)
        space_y = default(space_y, 0.1)
        universal_minmax = False
    elif compare_mode == "channel_wised" or compare_mode == "channel_wised_universal":
        ploter = BatchWisedPlotter()
        space_x = default(space_x, 0.2)
        space_y = default(space_y, 0.2)
        if compare_mode == "channel_wised_universal":
            universal_minmax = True
    else:
        raise ValueError(
            f"Unknown compare_mode: {compare_mode}. Must be 't_wised', 'channel_wised' or 'channel_wised_universal'."
        )
    return concate_fields_plot(
        ploters=ploter,
        fields=channel_wised_data,
        channel_names=time_names,
        batch_names=full_batch_names,
        vmin=vmin,
        vmax=vmax,
        universal_minmax=universal_minmax,
        subfig_size=subfig_size,
        space_x=space_x,
        space_y=space_y,
        cbar_pad=cbar_pad,
        c_bar_labels=c_bar_labels,
        real_size_ratio=real_size_ratio,
        num_colorbar_value=num_colorbar_value,
        ctick_format=ctick_format,
        show_ticks=show_ticks,
        cmap=cmap,
        use_sym_colormap=use_sym_colormap,
        ticks_x=ticks_x,
        ticks_y=ticks_y,
        save_name=save_name,
        alpha_func=alpha_func,
        show_time_index=False,
        title=title,
        width_correction=width_correction,
        height_correction=height_correction,
        label_x=label_x,
        label_y=label_y,
        label_t=label_t,
        animation=True,
        rotate_cbar_for_single_batch=rotate_cbar_for_single_batch,
        hide_batch_channel_name_for_single_plot=hide_batch_channel_name_for_single_plot,
        show_3d_coordinates=show_3d_coordinates,
        **kwargs,
    )
