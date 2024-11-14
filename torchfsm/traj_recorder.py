from typing import Callable,Optional,Literal
from .utils import default
import torch
import numpy as np
import copy

class IntervalController():
    
    def __init__(self,interval:int=1,start:int=0,) -> None:
        self.start=start
        self.interval=interval
        
    def __call__(self, step:int) -> bool:
        return step>=self.start and (step-self.start)%self.interval==0

class _TrajRecorder():

    def __init__(self,
                 control_func:Optional[Callable[[int],bool]]=None,
                 include_initial_state:bool=True,
                 ):
        control_func=default(control_func,lambda step: True)
        if include_initial_state:
            self.control_func=control_func
        else:
            self.control_func=lambda step: False if step==0 else control_func(step)

    def record(self,step:int,frame:torch.tensor):
        if self.control_func(step):
            self._record(step,frame)
    
    def _record(self,step:int,frame:torch.tensor):
        raise NotImplementedError
    
    def _traj_ifft(self,trajectory:torch.tensor):
        fft_dim=tuple(-1*(i+1) for i in range(len(trajectory.shape)-3))
        return torch.fft.ifftn(trajectory,dim=fft_dim)
    
    @property
    def trajectory(self):
        raise NotImplementedError
    
class AutoRecorder(_TrajRecorder):
    
    def __init__(self,
                 control_func:Optional[Callable[[int],bool]]=None,
                 include_initial_state:bool=True,
                 ifft_traj:bool=True,
                 ):
        super().__init__(control_func,include_initial_state)
        self._trajectory=[]
        self.ifft_traj=ifft_traj
    
    def _record(self,step:int,frame:torch.tensor):
        if not isinstance(self._trajectory,torch.Tensor):
            self._trajectory.append(copy.deepcopy(frame))
        else:
            raise RuntimeError("The trajectory has been finalized.")
    
    @property
    def trajectory(self):
        self._trajectory=torch.stack(self._trajectory,dim=1)
        if self.ifft_traj:
            self._trajectory=self._traj_ifft(self._trajectory).real
        return self._trajectory
    
class CPURecorder(AutoRecorder):
    
    def __init__(self,
                 control_func:Optional[Callable[[int],bool]]=None,
                 include_initial_state:bool=True,
                 ifft_traj:bool=True,
                 ):
        super().__init__(control_func,include_initial_state,ifft_traj)
    
    def _record(self,step:int,frame:torch.tensor):
        if frame.is_cpu:
            self._trajectory.append(copy.deepcopy(frame))
        else:
            self._trajectory.append(frame.cpu())

class DiskRecorder(_TrajRecorder):
    
    def __init__(self,
                 control_func:Optional[Callable[[int],bool]]=None,
                 include_initial_state:bool=True,
                 cache_dir:Optional[str]=None,
                 cache_freq:int=1,
                 temp_cache_loc:Literal["auto","cpu"]="cpu",
                 save_format:Literal["numpy","torch"]="torch",
                 ifft_traj:bool=True,
                 ):
        super().__init__(control_func,include_initial_state)
        self.ifft_traj=ifft_traj
        self.cache_dir=default(cache_dir,"./saved_traj/")
        self.cache_freq=cache_freq
        self._trajectory=[]
        self.temp_cache_loc=temp_cache_loc
        if self.temp_cache_loc not in ["auto","cpu"]:
            raise ValueError("temp_cache_loc must be either 'auto' or 'cpu'.")
        self.save_format=save_format
        if self.save_format not in ["numpy","torch"]:
            raise ValueError("save_format must be either 'numpy' or 'torch'.")
    
    def _record(self,step:int,frame:torch.tensor):
        if len(self.traj<self.cache_freq):    
            if self.temp_cache_loc=="cpu" and not frame.is_cpu:
                self._trajectory.append(frame.cpu())
            else:
                self._trajectory.append(copy.deepcopy(frame))
        else:
            temp_cache=torch.stack(self._trajectory,dim=1)
            temp_cache=temp_cache.to("cpu") if not temp_cache.is_cpu else temp_cache
            if self.iff_traj:
                temp_cache=self._traj_ifft(temp_cache).real
            if self.save_format=="numpy":
                np.save(self.cache_dir+f"temp_cache_{step}",temp_cache.numpy()) 
            else:
                torch.save(temp_cache,self.cache_dir+f"temp_cache_{step}")     
            self._trajectory=[]      
    
    @property
    def trajectory(self):
        return None