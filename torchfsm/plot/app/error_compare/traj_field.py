from ...core import (
    ChannelWisedPlotter,
    concate_traj_plots,
)
from ...._type import SpatialTensor, SpatialArray
from ...render import AlphaFunction
from typing import Union, Optional, Sequence, Tuple, Literal, Callable
from matplotlib.colors import Colormap
from matplotlib.animation import FuncAnimation
import torch
import numpy as np

def _process_error(
    traj_1: Union[SpatialTensor["1 T C H ..."], SpatialArray["1 T C H ..."]],
    traj_2: Union[SpatialTensor["1 T C H ..."], SpatialArray["1 T C H ..."]],
    error_func: Optional[
        Callable[
            [
                Union[SpatialTensor["B T C H ..."], SpatialArray["B T C H ..."]],
                Union[SpatialTensor["B T C H ..."], SpatialArray["B T C H ..."]],
            ],
            Union[SpatialTensor["B T C H ..."], SpatialArray["B T C H ..."]],
        ]
    ] = None,
    name_traj_1: str = "traj_1",
    name_traj_2: str = "traj_2",
    error_name: str = "error",
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
    vmin_traj: Optional[Union[float, Sequence[float]]] = None,
    vmax_traj: Optional[Union[float, Sequence[float]]] = None,
    vmin_error: Optional[Union[float, Sequence[float]]] = None,
    vmax_error: Optional[Union[float, Sequence[float]]] = None,
    c_bar_labels_traj: Optional[Sequence[str]] = None,
    c_bar_labels_error: Optional[Sequence[str]] = None,
):
    if traj_1.shape != traj_2.shape:
        raise ValueError(
            f"Trajectories must have the same shape, but got {traj_1.shape} and {traj_2.shape}."
        )
    if traj_1.shape[0] != 1 or traj_2.shape[0] != 1:
        raise ValueError(
            f"Trajectories must have batch size of 1, but got {traj_1.shape[0]} and {traj_2.shape[0]}."
        )
    if error_func is None:
        error_func = lambda x, y: ((x - y) ** 2) ** 0.5
    if isinstance(traj_1, torch.Tensor):
        traj_1 = traj_1.cpu().detach().numpy()
    if isinstance(traj_2, torch.Tensor):
        traj_2 = traj_2.cpu().detach().numpy()
    error = error_func(traj_1, traj_2)
    traj = np.concatenate([traj_1, traj_2], axis=0)
    batch_names = [[name_traj_1, name_traj_2], [error_name]]
    cmap = [cmap_traj, cmap_error]
    use_sym_colormap = [use_sym_colormap_traj, use_sym_colormap_error]
    alpha_func = [alpha_func_traj, alpha_func_error]
    c_bar_labels = [c_bar_labels_traj, c_bar_labels_error]
    vmin = [vmin_traj, vmin_error]
    vmax = [vmax_traj, vmax_error]
    return (
        traj,
        error,
        batch_names,
        cmap,
        use_sym_colormap,
        alpha_func,
        c_bar_labels,
        vmin,
        vmax,
    )

def _field_to_traj(field: Union[SpatialTensor["B C H ..."], SpatialArray["B C H ..."]]):
    if isinstance(field, torch.Tensor):
        field = field.cpu().detach()
        return field.unsqueeze(1)  # [B, 1, C, H, W, ...]
    elif isinstance(field, np.ndarray):
        return np.expand_dims(field, 1)  # [B, 1, C, H, W, ...]


def compare_error_traj(
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
    ticks_t: Tuple[Sequence[float], Sequence[str]] = None,
    ticks_x: Tuple[Sequence[float], Sequence[str]] = None,
    ticks_y: Tuple[Sequence[float], Sequence[str]] = None,
    show_ticks: Union[Literal["auto"], bool] = "auto",
    show_time_index: bool = True,
    animation: bool = True,
    fps=30,
    show_in_notebook: bool = True,
    animation_engine: Literal["jshtml", "html5"] = "html5",
    save_name: Optional[str] = None,
    show_3d_coordinates: bool = True,
    **kwargs,
) -> Optional[FuncAnimation]:
    """
    Compare two trajectories and plot the error between them.

    Args:
        traj_1 (Union[SpatialTensor["B T C H ..."], SpatialArray["B T C H ..."]]): The first trajectory to compare.
        traj_2 (Union[SpatialTensor["B T C H ..."], SpatialArray["B T C H ..."]]): The second trajectory to compare.
        error_func (Optional[Callable[[Union[SpatialTensor["B T C H ..."], SpatialArray["B T C H ..."]], Union[SpatialTensor["B T C H ..."], SpatialArray["B T C H ..."]]], Union[SpatialTensor["
        B T C H ..."], SpatialArray["B T C H ..."]]]], optional): The function to compute the error between the two trajectories. Defaults to None.
            If None, it will use the absolute difference.
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
        ticks_t (Tuple[Sequence[float], Sequence[str]], optional): Custom ticks for the time index. Defaults to None.
            If provided, it should be a tuple of two sequences:
            - The first sequence contains the tick positions.
            - The second sequence contains the tick labels.
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
        show_time_index (bool, optional): Whether to show the time index in the plot. Defaults to True.
        animation (bool, optional): Whether to create an animation of the plot. Defaults to True.
        fps (int, optional): The frames per second for the animation. Defaults to 30.
        show_in_notebook (bool, optional): Whether to show the plot in a Jupyter notebook. Defaults to True.
        animation_engine (Literal["jshtml", "html5"], optional): The engine to use for the animation. Defaults to "html5".
            - "jshtml": Uses JavaScript HTML for rendering the animation.
            - "html5": Uses HTML5 for rendering the animation.
        save_name (Optional[str], optional): The name of the file to save the plot. Defaults to None.
            If provided, the plot will be saved to the specified file.
            If not provided, the plot will not be saved.
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
    return concate_traj_plots(
        ploters=ChannelWisedPlotter(),
        trajs=[traj, error],
        channel_names=channel_names,
        batch_names=batch_names,
        hide_batch_channel_name_for_single_plot=False,
        vmin=vmin,
        vmax=vmax,
        subfig_size=subfig_size,
        space_x=space_x,
        space_y=space_y,
        rotate_cbar_for_single_batch=False,
        cbar_pad=cbar_pad,
        c_bar_labels=c_bar_labels,
        real_size_ratio=real_size_ratio,
        num_colorbar_value=num_colorbar_value,
        ctick_format=ctick_format,
        show_ticks=show_ticks,
        use_sym_colormap=use_sym_colormap,
        cmap=cmap,
        ticks_x=ticks_x,
        ticks_y=ticks_y,
        ticks_t=ticks_t,
        save_name=save_name,
        alpha_func=alpha_func,
        show_time_index=show_time_index,
        animation=animation,
        fps=fps,
        show_in_notebook=show_in_notebook,
        animation_engine=animation_engine,
        title=title,
        width_correction=width_correction,
        height_correction=height_correction,
        label_x=label_x,
        label_y=label_y,
        label_t=label_t,
        show_3d_coordinates=show_3d_coordinates,
        **kwargs,
    )


def compare_error_field(
    field_1: Union[SpatialTensor["1 C H ..."], SpatialArray["1 C H ..."]],
    field_2: Union[SpatialTensor["1 C H ..."], SpatialArray["1 C H ..."]],
    error_func: Optional[
        Callable[
            [
                Union[SpatialTensor["1 C H ..."], SpatialArray["1 C H ..."]],
                Union[SpatialTensor["1 C H ..."], SpatialArray["1 C H ..."]],
            ],
            Union[SpatialTensor["1 C H ..."], SpatialArray["1 C H ..."]],
        ]
    ] = None,
    name_field_1: str = "field 1",
    name_field_2: str = "field 2",
    error_name: str = "error",
    channel_names: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
    vmin_field: Optional[Union[float, Sequence[float]]] = None,
    vmax_field: Optional[Union[float, Sequence[float]]] = None,
    vmin_error: Optional[Union[float, Sequence[float]]] = None,
    vmax_error: Optional[Union[float, Sequence[float]]] = None,
    cmap_field: Union[str, Colormap] = "twilight",
    cmap_error: Union[str, Colormap] = "Reds",
    use_sym_colormap_field: bool = False,
    use_sym_colormap_error: bool = False,
    alpha_func_field: Union[
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
    c_bar_labels_field: Optional[Sequence[str]] = None,
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
    ticks_t: Tuple[Sequence[float], Sequence[str]] = None,
    ticks_x: Tuple[Sequence[float], Sequence[str]] = None,
    ticks_y: Tuple[Sequence[float], Sequence[str]] = None,
    show_ticks: Union[Literal["auto"], bool] = "auto",
    animation: bool = True,
    fps=30,
    show_in_notebook: bool = True,
    animation_engine: Literal["jshtml", "html5"] = "html5",
    save_name: Optional[str] = None,
    show_3d_coordinates: bool = True,
    **kwargs,
):
    if error_func is not None:

        def real_error_func(x, y):
            if isinstance(x, torch.Tensor):
                x = x.squeeze(1)
            if isinstance(y, torch.Tensor):
                y = y.squeeze(1)
            if isinstance(x, np.ndarray):
                x = np.squeeze(x, axis=1)
            if isinstance(y, np.ndarray):
                y = np.squeeze(y, axis=1)
            error = error_func(x, y)
            if isinstance(error, torch.Tensor):
                return error.unsqueeze(1)  # [B, 1, C, H, W, ...]
            elif isinstance(error, np.ndarray):
                return np.expand_dims(error, 1)

    else:
        real_error_func = None
    field_1 = _field_to_traj(field_1)
    field_2 = _field_to_traj(field_2)
    compare_error_traj(
        traj_1=field_1,
        traj_2=field_2,
        error_func=real_error_func,
        name_traj_1=name_field_1,
        name_traj_2=name_field_2,
        error_name=error_name,
        channel_names=channel_names,
        title=title,
        vmin_traj=vmin_field,
        vmax_traj=vmax_field,
        vmin_error=vmin_error,
        vmax_error=vmax_error,
        cmap_traj=cmap_field,
        cmap_error=cmap_error,
        use_sym_colormap_traj=use_sym_colormap_field,
        use_sym_colormap_error=use_sym_colormap_error,
        alpha_func_traj=alpha_func_field,
        alpha_func_error=alpha_func_error,
        num_colorbar_value=num_colorbar_value,
        c_bar_labels_traj=c_bar_labels_field,
        c_bar_labels_error=c_bar_labels_error,
        cbar_pad=cbar_pad,
        ctick_format=ctick_format,
        subfig_size=subfig_size,
        real_size_ratio=real_size_ratio,
        width_correction=width_correction,
        height_correction=height_correction,
        space_x=space_x,
        space_y=space_y,
        label_x=label_x,
        label_y=label_y,
        label_t=label_t,
        ticks_t=ticks_t,
        ticks_x=ticks_x,
        ticks_y=ticks_y,
        show_ticks=show_ticks,
        show_time_index=False,
        animation=animation,
        fps=fps,
        show_in_notebook=show_in_notebook,
        animation_engine=animation_engine,
        save_name=save_name,
        show_3d_coordinates=show_3d_coordinates,
        **kwargs,
    )

