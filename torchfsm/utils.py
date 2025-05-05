import torch
import numpy as np
from typing import Union, Optional,Tuple
from ._type import ValueList, SpatialArray, SpatialTensor, FourierArray, FourierTensor


def default(value, default):
    """
    Return the default value if the value is None.

    Args:
        value: The value to check.
        default: The default value to return if value is None.

    Returns:
        The value if it is not None, otherwise the default value.
    """
    return value if value is not None else default

def format_device_dtype(
    device: Optional[Union[torch.device, str]] = None,
    dtype: Optional[torch.dtype] = None,
)-> Tuple[torch.device, torch.dtype]:
    """
    Format the device and dtype for PyTorch.

    Args:
        device (Optional[Union[torch.device, str]]): The device to use. If None, defaults to CPU.
        dtype (Optional[torch.dtype]): The data type to use. If None, defaults to float32.

    Returns:
        tuple[torch.device, torch.dtype]: The formatted device and dtype.
    """
    if device is None:
        device = torch.device("cpu")
    elif isinstance(device, str):
        device = torch.device(device)
        if device.index is None and device.type != "cpu":
            device = torch.device(device.type, 0)
    dtype = default(dtype, torch.float32)
    return device, dtype


def statistics_traj(traj: ValueList[Union[torch.Tensor, np.ndarray]])-> Tuple[float, float, float, float]:
    """
    Compute the mean, std, min, and max of a trajectory.

    Args:
        traj (ValueList[Union[torch.Tensor, np.ndarray]]): The trajectory to compute statistics for.
            The trajectory can be a list of tensors or numpy arrays, or a single tensor or numpy array.
    
    Returns:
        tuple: A tuple containing the mean, std, min, and max of the trajectory.
            The mean, std, min, and max are computed along the first dimension of the trajectory.
    """
    # [B, T, C, H, ...]
    if not isinstance(traj, list):
        traj = [traj]
    traj = [
        traj_i if isinstance(traj_i, torch.Tensor) else torch.from_numpy(traj_i)
        for traj_i in traj
    ]
    new_shape = tuple([-1] + list(traj[0].shape[2:]))
    traj_all = torch.cat([t.reshape(new_shape) for t in traj], dim=0)
    means = [traj_all[:, i].mean().item() for i in range(traj_all.shape[1])]
    stds = [traj_all[:, i].std().item() for i in range(traj_all.shape[1])]
    mins = [traj_all[:, i].min().item() for i in range(traj_all.shape[1])]
    maxs = [traj_all[:, i].max().item() for i in range(traj_all.shape[1])]
    return means, stds, mins, maxs


def random_clip_traj(
    traj: Union[
        SpatialTensor["B T C H ..."],
        SpatialArray["B T C H ..."],
        FourierTensor["B T C H ..."],
        FourierArray["B T C H ..."],
    ],
    length: int,
)-> Union[
        SpatialTensor["B length C H ..."],
        SpatialArray["B length C H ..."],
        FourierTensor["B length C H ..."],
        FourierArray["B length C H ..."],
    ]:
    
    """
    Randomly clip a trajectory to a specified length.

    Args:
        traj (Union[SpatialTensor, SpatialArray, FourierTensor, FourierArray]): The trajectory to clip.
            The trajectory can be a tensor or numpy array with shape [B, T, C, H, ...].
        length (int): The length to clip the trajectory to.
            The length should be less than the original length of the trajectory.
    
    Returns:
        Union[SpatialTensor, SpatialArray, FourierTensor, FourierArray]: The clipped trajectory.
            The clipped trajectory has shape [B, length, C, H, ...].
    """
    if isinstance(traj, np.ndarray):
        traj = torch.from_numpy(traj)
    new_traj = []
    ori_len_time = traj.shape[1]
    start = torch.randint(0, ori_len_time - length, (traj.shape[0],))
    end = start + length
    for i in range(traj.shape[0]):
        new_traj.append(traj[i, start[i] : end[i]])
    new_traj = torch.stack(new_traj, dim=0)
    if isinstance(traj, np.ndarray):
        new_traj = new_traj.numpy()
    return new_traj

def random_select_frames(
    traj: Union[
        SpatialTensor["B T C H ..."],
        SpatialArray["B T C H ..."],
        FourierTensor["B T C H ..."],
        FourierArray["B T C H ..."],
    ],
    n_frames: int,
)-> Union[
        SpatialTensor["B n_frames C H ..."],
        SpatialArray["B n_frames C H ..."],
        FourierTensor["B n_frames C H ..."],
        FourierArray["B n_frames C H ..."],
    ]:
    """
    Randomly select a specified number of frames from a trajectory.

    Args:
        traj (Union[SpatialTensor, SpatialArray, FourierTensor, FourierArray]): The trajectory to select frames from.
            The trajectory can be a tensor or numpy array with shape [B, T, C, H, ...].
        n_frames (int): The number of frames to select.
            The number of frames should be less than the original length of the trajectory.
    
    Returns:
        Union[SpatialTensor, SpatialArray, FourierTensor, FourierArray]: The selected frames.
            The selected frames have shape [B, n_frames, C, H, ...].
    """
    if isinstance(traj, np.ndarray):
        traj = torch.from_numpy(traj)
    ori_len_time = traj.shape[1]
    selected_frames = torch.randint(0, ori_len_time, (n_frames,))
    new_traj = traj[:, selected_frames]
    if isinstance(traj, np.ndarray):
        new_traj = new_traj.numpy()
    return new_traj

def uniformly_select_frames(
    traj: Union[
        SpatialTensor["B T C H ..."],
        SpatialArray["B T C H ..."],
        FourierTensor["B T C H ..."],
        FourierArray["B T C H ..."],
    ],
    n_frames: int,
)-> Union[
        SpatialTensor["B n_frames C H ..."],
        SpatialArray["B n_frames C H ..."],
        FourierTensor["B n_frames C H ..."],
        FourierArray["B n_frames C H ..."],
    ]:
    """
    Uniformly select a specified number of frames from a trajectory.

    Args:
        traj (Union[SpatialTensor, SpatialArray, FourierTensor, FourierArray]): The trajectory to select frames from.
            The trajectory can be a tensor or numpy array with shape [B, T, C, H, ...].
        n_frames (int): The number of frames to select.
            The number of frames should be less than the original length of the trajectory.
            
    Returns:
        Union[SpatialTensor, SpatialArray, FourierTensor, FourierArray]: The selected frames.
            The selected frames have shape [B, n_frames, C, H, ...].
    """
    if isinstance(traj, np.ndarray):
        traj = torch.from_numpy(traj)
    ori_len_time = traj.shape[1]
    selected_frames = torch.linspace(0, ori_len_time - 1, n_frames).long()
    new_traj = traj[:, selected_frames]
    if isinstance(traj, np.ndarray):
        new_traj = new_traj.numpy()
    return new_traj  