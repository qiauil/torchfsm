from .._type import SpatialTensor
import torch
from typing import Union, Literal, Optional, Tuple


def normalize(
    u: SpatialTensor["B C H ..."],
    normalize_mode: Union[
        Literal["normal_distribution", "-1_1", "0_1"],
        Tuple[Union[float, Tuple[float, float]], Union[float, Tuple[float, float]]],
    ],
) -> SpatialTensor["B C H ..."]:
    """
    Normalize a spatial tensor according to the specified normalization mode.

    Args:
        u (SpatialTensor["B C H ..."]): The input spatial tensor to be normalized. Note that the normalization is performed independently for each batch.
        normalize_mode (Union[Literal["normal_distribution","-1_1","0_1"],
                              Tuple[Union[float, Tuple[float, float]],Union[float, Tuple[float, float]]]]):
            The normalization mode to apply. It can be one of the following:
            - "normal_distribution": Normalize to have zero mean and unit standard deviation.
            - "-1_1": Normalize to the range [-1, 1].
            - "0_1": Normalize to the range [0, 1].
            - "min_max": Normalize to the specified min and max values.
            - (mins, maxs): A tuple specifying the min and max values for normalization.
              mins and maxs can be either floats or tuples of floats specifying ranges.
    Returns:
        SpatialTensor["B C H ..."]: The normalized spatial tensor.
    """

    if normalize_mode not in ["normal_distribution", "-1_1", "0_1"]:
        if isinstance(normalize_mode, str):
            raise ValueError(
                f"normalize_mode must be one of 'normal_distribution','-1_1','0_1','min_max' or a tuple of (mins, maxs), but got {normalize_mode}"
            )
        if len(normalize_mode) != 2:
            raise ValueError(
                "When normalize_mode is a tuple, it must be of length 2 representing (mins, maxs) or ((min_min, min_max), (max_min, max_max))"
            )
        mins = normalize_mode[0]
        maxs = normalize_mode[1]
        normalize_mode = "min_max"
        if not isinstance(mins, float):
            if mins[0] >= mins[1]:
                raise ValueError(
                    f"mins[0] must be less than mins[1], but got mins={mins}"
                )
            mins = (
                torch.rand(
                    u.shape[0],
                    *[1] * (len(u.shape) - 1),
                    device=u.device,
                    dtype=u.dtype,
                )
                * (mins[1] - mins[0])
                + mins[0]
            )
        if not isinstance(maxs, float):
            if maxs[0] >= maxs[1]:
                raise ValueError(
                    f"maxs[0] must be less than maxs[1], but got maxs={maxs}"
                )
            maxs = (
                torch.rand(
                    u.shape[0],
                    *[1] * (len(u.shape) - 1),
                    device=u.device,
                    dtype=u.dtype,
                )
                * (maxs[1] - maxs[0])
                + maxs[0]
            )
        v_range = maxs - mins
        if (isinstance(v_range, float) and v_range <= 0) or (
            isinstance(v_range, torch.Tensor) and torch.any(v_range <= 0)
        ):
            raise ValueError("All elements in maxs must be greater than mins")
    if normalize_mode == "normal_distribution":
        u = u - u.mean(dim=[i for i in range(1, u.ndim)], keepdim=True)
        u = u / u.std(dim=[i for i in range(1, u.ndim)], keepdim=True)
        return u
    else:
        shape = [u.shape[0]] + [1] * (len(u.shape) - 1)
        max_v = torch.max(u.view(u.size(0), -1), dim=1).values.view(shape)
        min_v = torch.min(u.view(u.size(0), -1), dim=1).values.view(shape)
        u = (u - min_v) / (max_v - min_v)
        if normalize_mode == "-1_1":
            u = u * 2 - 1
        elif normalize_mode == "min_max":
            u = u * v_range + mins
        return u
