from ...._type import SpatialTensor, SpatialArray
from ...render import AlphaFunction
from ..frame import _plot_traj_frame_group
from .traj_field import _process_error
from typing import Union, Optional, Sequence, Tuple, Literal, Callable
from matplotlib.colors import Colormap

def compare_error_traj_frame(
    traj_1: Union[SpatialTensor["1 T C H ..."], SpatialArray["1 T C H ..."]],
    traj_2: Union[SpatialTensor["1 T C H ..."], SpatialArray["1 T C H ..."]],
    error_func: Optional[
        Callable[
            [
                Union[SpatialTensor["1 T C H ..."], SpatialArray["1 T C H ..."]],
                Union[SpatialTensor["1 T C H ..."], SpatialArray["1 T C H ..."]],
            ],
            Union[SpatialTensor["1 T C H ..."], SpatialArray["1 T C H ..."]],
        ]
    ] = None,
    n_frames: int = 5,
    name_traj_1: str = "traj 1",
    name_traj_2: str = "traj 2",
    error_name: str = "error",
    channel_names: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
    vmin_traj: Optional[Union[float, Sequence[float]]] = None,
    vmax_traj: Optional[Union[float, Sequence[float]]] = None,
    vmin_error: Optional[Union[float, Sequence[float]]] = None,
    vmax_error: Optional[Union[float, Sequence[float]]] = None,
    cmap_traj: Union[str, Colormap] = "twilight",
    cmap_error: Union[str, Colormap] = "Reds",
    use_sym_colormap_traj: bool = False,
    use_sym_colormap_error: bool = False,
    alpha_func_traj: Union[
        Literal[
            "zigzag",
            "central_peak",
            "central_valley",
            "linear_increase",
            "linear_decrease",
        ],
        AlphaFunction,
    ] = "zigzag",
    alpha_func_error: Union[
        Literal[
            "zigzag",
            "central_peak",
            "central_valley",
            "linear_increase",
            "linear_decrease",
        ],
        AlphaFunction,
    ] = "linear_increase",
    num_colorbar_value: int = 4,
    c_bar_labels_traj: Optional[Sequence[str]] = None,
    c_bar_labels_error: Optional[Sequence[str]] = None,
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
    frame_start_index: int = 0,
    show_3d_coordinates: bool = True,
    **kwargs,
):
    """
    Compare two trajectories and plot the error between them.

    Args:
        traj_1 (Union[SpatialTensor["B T C H ..."], SpatialArray["B T C H ..."]]): The first trajectory to compare.
        traj_2 (Union[SpatialTensor["B T C H ..."], SpatialArray["B T C H ..."]]): The second trajectory to compare.
        error_func (Optional[Callable[[Union[SpatialTensor["B T C H ..."], SpatialArray["B T C H ..."]], Union[SpatialTensor["B T C H ..."], SpatialArray["B T C H ..."]]], Union[SpatialTensor["
        B T C H ..."], SpatialArray["B T C H ..."]]]], optional): The function to compute the error between the two trajectories. Defaults to None.
            If None, it will use the absolute difference.
        n_frames (int, optional): The number of frames to select from the trajectories. Defaults to 5.
        name_traj_1 (str, optional): The name of the first trajectory. Defaults to "traj_1".
        name_traj_2 (str, optional): The name of the second trajectory. Defaults to "traj_2".
        error_name (str, optional): The name for the error trajectory. Defaults to "error".
        channel_names (Optional[Sequence[str]], optional): The names of the channels. Defaults to None.
            If None, it will use default channel names like "channel 0", "channel 1", etc.
        title (Optional[str], optional): The title of the plot. Defaults to None.
        vmin_traj (Optional[Union[float, Sequence[float]]], optional): The minimum value for the color scale of the trajectories. Defaults to None.
            If a sequence is provided, it should have the same length as the number of channels.
        vmax_traj (Optional[Union[float, Sequence[float]]], optional): The maximum value for the color scale of the trajectories. Defaults to None.
            If a sequence is provided, it should have the same length as the number of channels.
        vmin_error (Optional[Union[float, Sequence[float]]], optional): The minimum value for the color scale of the error. Defaults to None.
            If a sequence is provided, it should have the same length as the number of channels.
        vmax_error (Optional[Union[float, Sequence[float]]], optional): The maximum value for the color scale of the error. Defaults to None.
            If a sequence is provided, it should have the same length as the number of channels.
        cmap_traj (Union[str, Colormap], optional): The colormap to use for the trajectories. Defaults to "twilight".
        cmap_error (Union[str, Colormap], optional): The colormap to use for the error. Defaults to "Reds".
        use_sym_colormap_traj (bool, optional): Whether to use a symmetric colormap for the trajectories. Defaults to False.
        use_sym_colormap_error (bool, optional): Whether to use a symmetric colormap for the error. Defaults to False
        alpha_func_traj (Union[Literal["zigzag","central_peak","central_valley","linear_increase","linear_decrease",],AlphaFunction,], optional): The alpha function for the colormap of the trajectories. Defaults to "zigzag".
        alpha_func_error (Union[Literal["zigzag","central_peak","central_valley","linear_increase","linear_decrease",],AlphaFunction,], optional): The alpha function for the colormap of the error. Defaults to "linear_increase".
        num_colorbar_value (int, optional): The number of values for the colorbar. Defaults to 4.
        c_bar_labels_traj (Optional[Sequence[str]], optional): The labels for the colorbar of the trajectories. Defaults to None.
            If provided, it should have the same length as the number of channels.
            If not provided, the colorbar will not have labels.
        c_bar_labels_error (Optional[Sequence[str]], optional): The labels for the colorbar of the error. Defaults to None.
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
            If provided, it should be a tuple of two sequences:
            - The first sequence contains the tick positions.
            - The second sequence contains the tick labels.
        ticks_y (Tuple[Sequence[float], Sequence[str]], optional): Custom ticks for the y-axis. Defaults to None.
            If provided, it should be a tuple of two sequences:
            - The first sequence contains the tick positions.
            - The second sequence contains the tick labels.
        show_ticks (Union[Literal["auto"], bool], optional): Whether to show ticks. Defaults to "auto".
            If "auto", ticks will be shown if the number of channels is greater than 1.
            If True, ticks will always be shown.
            If False, ticks will never be shown.
        save_name (Optional[str], optional): The name of the file to save the plot. Defaults to None.
            If provided, the plot will be saved to the specified file.
            If not provided, the plot will not be saved.
        frame_start_index (int, optional): The starting index for the frames to be compared. Defaults to 0.
        show_3d_coordinates (bool, optional): Whether to show 3D coordinates for 3D plots. Defaults to True.
    """
    (
        traj,
        error,
        batch_names,
        cmap,
        use_sym_colormap,
        alpha_func,
        c_bar_labels,
        vmin,
        vmax,
    ) = _process_error(
        traj_1=traj_1,
        traj_2=traj_2,
        error_func=error_func,
        name_traj_1=name_traj_1,
        name_traj_2=name_traj_2,
        error_name=error_name,
        cmap_traj=cmap_traj,
        cmap_error=cmap_error,
        use_sym_colormap_traj=use_sym_colormap_traj,
        use_sym_colormap_error=use_sym_colormap_error,
        alpha_func_traj=alpha_func_traj,
        alpha_func_error=alpha_func_error,
        vmin_traj=vmin_traj,
        vmax_traj=vmax_traj,
        vmin_error=vmin_error,
        vmax_error=vmax_error,
        c_bar_labels_traj=c_bar_labels_traj,
        c_bar_labels_error=c_bar_labels_error,
    )
    return _plot_traj_frame_group(
        trajs=[traj, error],
        batch_names=batch_names,
        cmap=cmap,
        use_sym_colormap=use_sym_colormap,
        alpha_func=alpha_func,
        c_bar_labels=c_bar_labels,
        vmin=vmin,
        vmax=vmax,
        n_frames=n_frames,
        subfig_size=subfig_size,
        real_size_ratio=real_size_ratio,
        width_correction=width_correction,
        height_correction=height_correction,
        space_x=space_x,
        space_y=space_y,
        label_x=label_x,
        label_y=label_y,
        label_t=label_t,
        ticks_x=ticks_x,
        ticks_y=ticks_y,
        show_ticks=show_ticks,
        save_name=save_name,
        title=title,
        channel_names=channel_names,
        frame_start_index=frame_start_index,
        rotate_cbar_for_single_batch=False,
        hide_batch_channel_name_for_single_plot=False,
        num_colorbar_value=num_colorbar_value,
        ctick_format=ctick_format,
        cbar_pad=cbar_pad,
        show_3d_coordinates=show_3d_coordinates,
        **kwargs,
    )
