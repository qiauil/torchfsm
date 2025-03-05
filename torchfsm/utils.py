import torch
import numpy as np
from typing import Union, Optional
from ._type import ValueList, SpatialArray, SpatialTensor, FourierArray, FourierTensor


def default(value, default):
    return value if value is not None else default


def format_device_dtype(
    device: Optional[Union[torch.device, str]] = None,
    dtype: Optional[torch.dtype] = None,
):
    if device is None:
        device = torch.device("cpu")
    elif isinstance(device, str):
        device = torch.device(device)
        if device.index is None and device.type != "cpu":
            device = torch.device(device.type, 0)
    dtype = default(dtype, torch.float32)
    return device, dtype


def statistics_traj(traj: ValueList[Union[torch.Tensor, np.ndarray]]):
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
):
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
):
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
):
    if isinstance(traj, np.ndarray):
        traj = torch.from_numpy(traj)
    ori_len_time = traj.shape[1]
    selected_frames = torch.linspace(0, ori_len_time - 1, n_frames).long()
    new_traj = traj[:, selected_frames]
    if isinstance(traj, np.ndarray):
        new_traj = new_traj.numpy()
    return new_traj  