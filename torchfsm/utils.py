import torch
import numpy as np
from typing import Union,Optional
from typing import _SpecialForm,_type_check,List,Union
        
def default(value, default):
    return value if value is not None else default

def format_device_dtype(device:Optional[Union[torch.device,str]]=None,dtype:Optional[torch.dtype]=None):
    if device is None:
        device=torch.device("cpu")
    elif isinstance(device,str):
        device=torch.device(device)
        if device.index is None and device.type!="cpu":
            device=torch.device(device.type,0)
    dtype=default(dtype,torch.float32)
    return device,dtype

@_SpecialForm
def ValueList(self, parameters):
    arg = _type_check(parameters, f"{self} requires a single type.")
    return Union[arg, List[arg]]

def statistics_traj(traj:ValueList[Union[torch.Tensor,np.ndarray]]):
    # [B, T, C, H, ...]
    if not isinstance(traj,list):
        traj=[traj]
    traj=[traj_i if isinstance(traj_i,torch.Tensor) else torch.from_numpy(traj_i) for traj_i in traj]
    new_shape=tuple([-1]+list(traj[0].shape[2:]))
    traj_all=torch.cat([t.reshape(new_shape) for t in traj],dim=0)
    means=[traj_all[:,i].mean().item() for i in range(traj_all.shape[1])]
    stds=[traj_all[:,i].std().item() for i in range(traj_all.shape[1])]
    mins=[traj_all[:,i].min().item() for i in range(traj_all.shape[1])]
    maxs=[traj_all[:,i].max().item() for i in range(traj_all.shape[1])]
    return means,stds,mins,maxs