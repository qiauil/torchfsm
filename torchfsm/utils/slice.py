import torch
import numpy as np
from typing import Union, Optional,Sequence
from .._type import  SpatialArray, SpatialTensor

def traj_slices(
    traj: Union[SpatialTensor["B T C H ..."], SpatialArray["B T C H ..."]],
    slice_control: Sequence[Optional[Union[int, float]]],
) -> Sequence[Union[SpatialTensor["B T C H ..."], SpatialArray["B T C H ..."]]]:
    """
    Slice a trajectory along specified dimensions.

    Args:
        traj (Union[SpatialTensor["B T C H ..."], SpatialArray["B T C H ..."]]): The trajectory to slice.
        slice_control (Sequence[Optional[Union[int,float]]]): A sequence of slice values for each dimension.
            If a value is None, that dimension will not be sliced.
            If a value is negative, it will slice from the end of that dimension.
            If a value is positive, it will slice from the start of that dimension.

    Returns:
        Union[Sequence[SpatialTensor["B T C H ..."], SpatialArray["B T C H ..."]]]:
            A sequence of sliced trajectories. Each element corresponds to a slice along one dimension.
    """
    n_dim = len(traj.shape) - 3
    if n_dim != len(slice_control):
        raise ValueError(
            f"The number of slice control values {len(slice_control)} should be equal to the number of dimensions {n_dim} in the input trajectory."
        )
    if n_dim == 1:
        raise ValueError("Cannot slice 1D trajectory.")
    re = []
    if len(slice_control) != n_dim:
        raise ValueError(
            f"The number of slice control values {len(slice_control)} should be equal to the number of dimensions {n_dim} in the input trajectory."
        )
    for i, slice_value in enumerate(slice_control):
        if slice_value is None:
            continue
        if slice_value < 1:
            slice_value = int(traj.shape[i + 3] * slice_value)
        else:
            slice_value = int(slice_value)
        if i == 0:
            re.append(traj[:, :, :, slice_value, ...])
        elif i == 1:
            re.append(traj[:, :, :, :, slice_value, ...])
        elif i == 2:
            re.append(traj[:, :, :, :, :, slice_value])
    if len(re) == 0:
        raise ValueError("No slice value provided.")
    return re


def field_slices(
    field: Union[SpatialTensor["B C H ..."], SpatialArray["B C H ..."]],
    slice_control: Sequence[Optional[Union[int, float]]],
) -> Sequence[Union[SpatialTensor["B C H ..."], SpatialArray["B C H ..."]]]:
    """
    Slice a field along specified dimensions.
    Args:
        field (Union[SpatialTensor["B C H ..."], SpatialArray["B C H ..."]]): The field to slice.
        slice_control (Sequence[Optional[Union[int,float]]]): A sequence of slice values for each dimension.
            If a value is None, that dimension will not be sliced.
            If a value is negative, it will slice from the end of that dimension.
            If a value is positive, it will slice from the start of that dimension.
    Returns:
        Union[Sequence[SpatialTensor["B C H ..."], SpatialArray["B C H ..."]]]:
            A sequence of sliced fields. Each element corresponds to a slice along one dimension.
    """

    if isinstance(traj, torch.Tensor):
        traj = field.unsqueeze(1)
    if isinstance(traj, np.ndarray):
        traj = np.expand_dims(field, axis=1)
    return traj_slices(traj=traj, slice_control=slice_control)