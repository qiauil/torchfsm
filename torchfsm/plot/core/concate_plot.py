import numpy as np
import matplotlib.pyplot as plt
import torch
from ..._type import SpatialTensor, SpatialArray,ValueList
from ..render import AlphaFunction
from typing import Union, Optional, Sequence, Tuple, Literal
from matplotlib.colors import Colormap
from .plotter import ChannelWisedPlotter, BatchWisedPlotter

def concate_traj_plots(
    ploters: ValueList[Union[ChannelWisedPlotter, BatchWisedPlotter]],
    trajs: Sequence[Union[SpatialTensor["B T C H ..."], SpatialArray["B T C H ..."]]],
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
    label_t: ValueList[str] = "t",
    ticks_t: Tuple[Sequence[float], Sequence[str]] = None,
    ticks_x: ValueList[Tuple[Sequence[float], Sequence[str]]] = None,
    ticks_y: ValueList[Tuple[Sequence[float], Sequence[str]]] = None,
    show_ticks: Union[Literal["auto"], bool] = "auto",
    show_time_index: bool = True,
    animation: bool = True,
    fps=30,
    show_in_notebook: bool = True,
    animation_engine: Literal["jshtml", "html5"] = "html5",
    save_name: Optional[str] = None,
    **kwargs,
):
    """
    Plot multiple trajectories with different plotters. This function is a wrapper for the `plot` method of `ChannelWisedPlotter` and `BatchWisedPlotter`.

    Args:
        ploters (ValueList[Union[ChannelWisedPlotter, BatchWisedPlotter]]): A list of plotters to use for plotting the trajectories. Each plotter should be an instance of `ChannelWisedPlotter` or `BatchWisedPlotter`.
        trajs (Sequence[Union[SpatialTensor["B T C H ..."], SpatialArray["B T C H ..."]]]): A list of trajectories to plot. Each trajectory should be a `SpatialTensor` or `SpatialArray` with shape `(B, T, C, H, ...)`,
            where `B` is the batch size, `T` is the number of time steps, `C` is the number of channels, and `H` is the height (and possibly width or depth) of the spatial dimensions.
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
        rotate_cbar_for_single_batch (ValueList[bool], optional): Whether to rotate the colorbar for a single batch. Defaults to True.
        subfig_size (float, optional): The size of each subplot. Defaults to 2.5.
        real_size_ratio (bool, optional): Whether to use the real size ratio for the subfigures. Defaults to False.
        width_correction (float, optional): The correction factor for the width of the subfigures. Defaults to 1.0.
        height_correction (float, optional): The correction factor for the height of the subfigures. Defaults to 1.0.
        space_x (float, optional): The space between subfigures in the x direction. Defaults to 0.7.
        space_y (float, optional): The space between subfigures in the y direction. Defaults to 0.1.
        label_x (ValueList[str], optional): The label for the x axis. If a single value is provided, it will be used for all trajectories. If a list is provided, it should have the same length as `trajs`. Defaults to "x".
        label_y (ValueList[str], optional): The label for the y axis. If a single value is provided, it will be used for all trajectories. If a list is provided, it should have the same length as `trajs`. Defaults to "y".
        label_t (ValueList[str], optional): The label for the time axis. If a single value is provided, it will be used for all trajectories. If a list is provided, it should have the same length as `trajs`. Defaults to
            "t".
        ticks_t (Tuple[Sequence[float], Sequence[str]], optional): The ticks for the time axis. If provided, it should be a tuple of two sequences: the first sequence contains the tick positions, and the second sequence contains the tick labels. Defaults to None.
        ticks_x (ValueList[Tuple[Sequence[float], Sequence[str]]], optional): The ticks for the x axis. If a single value is provided, it will be used for all trajectories. If a list is provided, it should have the same length as `trajs`. Defaults to None.
        ticks_y (ValueList[Tuple[Sequence[float], Sequence[str]]], optional): The ticks for the y axis. If a single value is provided, it will be used for all trajectories. If a list is provided, it should have the same length as `trajs`. Defaults to None.
        show_ticks (Union[Literal["auto"], bool], optional): Whether to show ticks on the axes. If set to "auto", ticks will be shown for 1D trajectories with animation. Defaults to "auto".
        show_time_index (bool, optional): Whether to show the time index in the title. Defaults to True.
        animation (bool, optional): Whether to create an animation. Defaults to True.
        fps (int, optional): The frames per second for the animation. Defaults to 30.
        show_in_notebook (bool, optional): Whether to show the plot in a Jupyter notebook. Defaults to True.
        animation_engine (Literal["jshtml", "html5"], optional): The engine to use for the animation. Defaults to "html5".
        save_name (Optional[str], optional): The name of the file to save the plot. Defaults to None.
        show_plot (bool, optional): Whether to show the plot. Defaults to True.
        **kwargs: Additional keyword arguments to pass to the plotting functions.
    """
    if not isinstance(ploters, Sequence):
        ploters = [ploters] * len(trajs)
    ploter = ploters[-1]

    def check_value_list(value_list):
        n = len(trajs)
        if not isinstance(value_list, Sequence) or isinstance(value_list, str):
            value_list = [value_list] * n
        elif len(value_list) != n:
            raise ValueError(
                f"The length of the value list {value_list} should be equal to the number of trajectories {n}."
            )
        return value_list

    batch_names = check_value_list(batch_names)
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
    label_t = check_value_list(label_t)
    rotate_cbar_for_single_batch = check_value_list(rotate_cbar_for_single_batch)
    hide_batch_channel_name_for_single_plot = check_value_list(
        hide_batch_channel_name_for_single_plot
    )
    batch_size, n_frame, n_channel = trajs[0].shape[:3]
    for traj_i in trajs[1:]:
        if traj_i.shape[1:3] != (n_frame, n_channel):
            raise ValueError(
                "All trajectories must have the same n_frame and n_channel."
            )
    n_dim = len(trajs[0].shape) - 3
    ratios = [0.0]
    for i in range(len(trajs)):
        if n_dim == 1 or n_dim == 3:
            ratios.append(1.0 * trajs[i].shape[0])
        else:
            ratios.append(trajs[i].shape[-1] * trajs[i].shape[0])
    ratios.append(0.0)
    total_height = 0.0
    for i,traj_i in enumerate(trajs):
        space_x, space_y, cbar_pad = ploter.setup_pad_size(space_x, space_y, cbar_pad)
        cbar_location, cbar_mode, _ = ploter.setup_colorbar_location(
             batch_size, 
             universal_minmax[i], 
             rotate_cbar_for_single_batch[i]
        )
        total_width, height,_ = ploter.plot_size(
            subfig_size, 
            traj_i, 
            animation, 
            batch_size, 
            n_channel, 
            space_x, 
            space_y,
            width_correction,
            height_correction,
            cbar_mode,
            cbar_location,
            cbar_pad=cbar_pad
        )
        total_height += height
    # The pad size is hard coded for now, can be changed in the future if needed.
    pad_left= 0.5
    pad_bottom=0.5
    h_space = 3.0
    # here we consider the size of title and t_label
    # it is nicer if we can remove the space when title or t_label is None, just like what we do in the plotter
    # Ideally, total_height should be calculated as total_height += (h_space*(len(trajs)+1)+h_space*2)
    # But we find below setting works better in practice
    # The size setting in matplotlib is quite tricky...
    total_height += h_space*(len(trajs)-1)
    total_width += pad_left * 2
    pad_left = pad_left / total_width
    pad_bottom = pad_bottom / total_height
    h_space = h_space / total_height
    figure = plt.figure(
        figsize=(total_width, total_height),
        constrained_layout=False,
    )
    gs = figure.add_gridspec(
        len(trajs) + 2,
        1,
        height_ratios=ratios,
        left=pad_left,
        right=1.0 - pad_left,
        top=1 - pad_bottom,
        bottom=pad_bottom,
        hspace=h_space,
    )
    ax_t = figure.add_subplot(gs[0])
    ax_title = figure.add_subplot(gs[-1])
    returns = []
    for i in range(len(trajs)):
        returns.append(
            ploters[i].plot(
                rect_plot=gs[i + 1],
                fig=figure,
                traj=trajs[i],
                channel_names=(
                    channel_names if i == len(trajs) - 1 else [""] * n_channel
                ),
                batch_names=batch_names[i],
                hide_batch_channel_name_for_single_plot=hide_batch_channel_name_for_single_plot[i],
                vmin=vmin[i],
                vmax=vmax[i],
                universal_minmax=universal_minmax[i],
                subfig_size=subfig_size,
                space_x=space_x,
                space_y=space_y,
                cbar_pad=cbar_pad,
                c_bar_labels=c_bar_labels[i],
                num_colorbar_value=num_colorbar_value,
                ctick_format=ctick_format,
                rotate_cbar_for_single_batch=rotate_cbar_for_single_batch[i],
                show_ticks=show_ticks,
                cmap=cmap[i],
                use_sym_colormap=use_sym_colormap[i],
                label_x=label_x[i],
                label_y=label_y[i],
                label_t=label_t[i],
                ticks_x=ticks_x[i],
                ticks_y=ticks_y[i],
                ticks_t=ticks_t,
                save_name=None,
                alpha_func=alpha_func[i],
                animation=animation,
                animation_engine=animation_engine,
                fps=fps,
                show_in_notebook=False,
                show_plot=False,
                return_ani_func=True,
                show_time_index=False,
                real_size_ratio=real_size_ratio,
                **kwargs,
            )
        )
    if returns[0] is None:
        ploter.title_t(i, title, ticks_t, show_time_index, ax_t, ax_title, label_t[-1])
        if save_name is not None:
            plt.savefig(save_name, bbox_inches="tight")
        plt.show()
    else:
        def ani_func(i):
            for ani_func_i in returns:
                ani_func_i(i)
                ploter.title_t(
                    i, title, ticks_t, show_time_index, ax_t, ax_title, label_t[-1]
                )

        return ploter.build_animation(
            figure,
            ani_func,
            n_frame,
            fps,
            show_in_notebook=show_in_notebook,
            animation_engine=animation_engine,
        )


def concate_fields_plot(
    ploters: ValueList[Union[ChannelWisedPlotter, BatchWisedPlotter]],
    fields: Sequence[Union[SpatialTensor["B T C H ..."], SpatialArray["B T C H ..."]]],
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
    label_t: ValueList[str] = "t",
    ticks_t: Tuple[Sequence[float], Sequence[str]] = None,
    ticks_x: ValueList[Tuple[Sequence[float], Sequence[str]]] = None,
    ticks_y: ValueList[Tuple[Sequence[float], Sequence[str]]] = None,
    show_ticks: Union[Literal["auto"], bool] = "auto",
    show_time_index: bool = True,
    animation: bool = True,
    fps=30,
    show_in_notebook: bool = True,
    animation_engine: Literal["jshtml", "html5"] = "html5",
    save_name: Optional[str] = None,
    **kwargs,
):
    """
    Plot multiple static fields with different plotters. This function is a wrapper for plotting static fields.

    Args:
        ploters (ValueList[Union[ChannelWisedPlotter, BatchWisedPlotter]]): A list of plotters to use for plotting the fields.
        fields (Sequence[Union[SpatialTensor["B T C H ..."], SpatialArray["B T C H ..."]]]): A list of static fields to plot.
        channel_names (Optional[Sequence[str]], optional): The names of the channels. Defaults to None.
        batch_names (Optional[ValueList[Sequence[str]]], optional): The names of the batches. Defaults to None.
        hide_batch_channel_name_for_single_plot (ValueList[bool], optional): Whether to hide the batch and channel name for a single batch or channel. Defaults to True.
        title (Optional[str], optional): The title of the plot. Defaults to None.
        vmin (Optional[ValueList[Sequence[Union[float, Sequence[float]]]]], optional): The minimum value for the color scale. Defaults to None.
        vmax (Optional[ValueList[Sequence[Union[float, Sequence[float]]]], optional): The maximum value for the color scale. Defaults to None.
        cmap (ValueList[Union[str, Colormap]], optional): The colormap to use for the plot. Defaults to "twilight".
        use_sym_colormap (ValueList[bool], optional): Whether to use a symmetric colormap. Defaults to False.
        alpha_func (ValueList[Union[Literal["zigzag", "central_peak", "central_valley", "linear_increase", "linear_decrease", "luminance"], AlphaFunction]], optional): The alpha function to use for 3D rendering. Defaults to "zigzag".
        universal_minmax (ValueList[bool], optional): Whether to use a universal min-max for the color scale. Defaults to False.
        num_colorbar_value (int, optional): The number of values to show on the colorbar. Defaults to 4.
        c_bar_labels (Optional[ValueList[Sequence[str]]], optional): The labels for the colorbar. Defaults to None.
        cbar_pad (Optional[float], optional): The padding for the colorbar. Defaults to 0.1.
        ctick_format (Optional[str], optional): The format for the colorbar ticks. Defaults to "%.1f".
        rotate_cbar_for_single_batch (ValueList[bool], optional): Whether to rotate the colorbar for a single batch. Defaults to True.
        subfig_size (float, optional): The size of each subplot. Defaults to 2.5.
        real_size_ratio (bool, optional): Whether to use the real size ratio for the subfigures. Defaults to False.
        width_correction (float, optional): The correction factor for the width of the subfigures. Defaults to 1.0.
        height_correction (float, optional): The correction factor for the height of the subfigures. Defaults to 1.0.
        space_x (float, optional): The space between subfigures in the x direction. Defaults to 0.7.
        space_y (float, optional): The space between subfigures in the y direction. Defaults to 0.1.
        label_x (ValueList[str], optional): The label for the x axis. Defaults to "x".
        label_y (ValueList[str], optional): The label for the y axis. Defaults to "y".
        label_t (ValueList[str], optional): The label for the time axis. Defaults to "t".
        ticks_t (Tuple[Sequence[float], Sequence[str]], optional): The ticks for the time axis. Defaults to None.
        ticks_x (ValueList[Tuple[Sequence[float], Sequence[str]]], optional): The ticks for the x axis. Defaults to None.
        ticks_y (ValueList[Tuple[Sequence[float], Sequence[str]]], optional): The ticks for the y axis. Defaults to None.
        show_ticks (Union[Literal["auto"], bool], optional): Whether to show ticks on the axes. Defaults to "auto".
        show_time_index (bool, optional): Whether to show the time index. Defaults to True.
        animation (bool, optional): Whether to create an animation. Defaults to True.
        fps (int, optional): The frames per second for the animation. Defaults to 30.
        show_in_notebook (bool, optional): Whether to show the plot in a Jupyter notebook. Defaults to True.
        animation_engine (Literal["jshtml", "html5"], optional): The engine to use for the animation. Defaults to "html5".
        save_name (Optional[str], optional): The name of the file to save the plot. Defaults to None.
        **kwargs: Additional keyword arguments to pass to the plotting functions.

    Returns:
        Animation object or None.
    """
    trajs = []
    for field in fields:
        if isinstance(field, torch.Tensor):
            trajs.append(field.unsqueeze(1))
        elif isinstance(field, np.ndarray):
            trajs.append(np.expand_dims(field, axis=1))
        else:
            raise TypeError(
                "The input field should be a torch.Tensor or a numpy.ndarray."
            )
    return concate_traj_plots(
        ploters=ploters,
        trajs=trajs,
        channel_names=channel_names,
        batch_names=batch_names,
        hide_batch_channel_name_for_single_plot=hide_batch_channel_name_for_single_plot,
        title=title,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        use_sym_colormap=use_sym_colormap,
        alpha_func=alpha_func,
        universal_minmax=universal_minmax,
        num_colorbar_value=num_colorbar_value,
        c_bar_labels=c_bar_labels,
        cbar_pad=cbar_pad,
        ctick_format=ctick_format,
        rotate_cbar_for_single_batch=rotate_cbar_for_single_batch,
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
        show_time_index=show_time_index,
        animation=animation,
        fps=fps,
        show_in_notebook=show_in_notebook,
        animation_engine=animation_engine,
        save_name=save_name,
        **kwargs,
    )
