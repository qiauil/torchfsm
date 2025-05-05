from typing import Callable,Optional,Literal
from .utils import default
import torch
import numpy as np
import copy

class IntervalController():
    """
    A class to control the recording of trajectories at specified intervals.
        This class can be used as an input for the `control_func` parameters of recorder objects.

    Args:
        interval (int): The interval at which to record the trajectory.
        start (int): The step at which to start recording the trajectory.
    """
    
    def __init__(self,interval:int=1,start:int=0,) -> None:
        self.start=start
        self.interval=interval
        
    def __call__(self, step:int) -> bool:
        return step>=self.start and (step-self.start)%self.interval==0

class _TrajRecorder():

    """
    A base class for trajectory recorders.
        A recorder is an object that helps to control the recording of trajectories during a simulation
            
    Args:
        control_func (Optional[Callable[[int],bool]]): A function that takes a step as input and returns a boolean indicating whether to record the trajectory at that step.
        include_initial_state (bool): If True, the initial state will be included in the trajectory.
    """

    def __init__(self,
                 control_func:Optional[Callable[[int],bool]]=None,
                 include_initial_state:bool=True,
                 ):
        control_func=default(control_func,lambda step: True)
        if include_initial_state:
            self.control_func=control_func
        else:
            self.control_func=lambda step: False if step==0 else control_func(step)
        self.return_in_fourier=False

    def record(self,step:int,frame:torch.tensor):
        """
        Record the trajectory at a given step.

        Args:
            step (int): The current step.
            frame (torch.tensor): The current frame to be recorded.
        """
        if self.control_func(step):
            self._record(step,frame)
    
    def _record(self,step:int,frame:torch.tensor):
        """
        Record the trajectory at a given step.
            This method should be implemented by subclasses.

        Args:
            step (int): The current step.
            frame (torch.tensor): The current frame to be recorded.
        """
        raise NotImplementedError
    
    def _traj_ifft(self,trajectory:torch.tensor):
        """
        Perform an inverse FFT on the trajectory.

        Args:
            trajectory (torch.tensor): The trajectory to be transformed.

        Returns:
            torch.tensor: The transformed trajectory.
        """
        fft_dim=tuple(-1*(i+1) for i in range(len(trajectory.shape)-3))
        return torch.fft.ifftn(trajectory,dim=fft_dim)

    @property
    def trajectory(self):
        """
        Get the recorded trajectory.
            This method should be implemented by subclasses.
        
        Args:
            return_in_fourier (bool): If True, return the trajectory in Fourier space. Default is False.
        
        Returns:
            torch.tensor: The recorded trajectory.
        """
        raise NotImplementedError
    
class AutoRecorder(_TrajRecorder):

    """
    A recorder that save the trajectory at the same devices as the simulation.

    Args:
        control_func (Optional[Callable[[int],bool]]): A function that takes a step as input and returns a boolean indicating whether to record the trajectory at that step.
        include_initial_state (bool): If True, the initial state will be included in the trajectory.
    """

    def __init__(self,
                 control_func:Optional[Callable[[int],bool]]=None,
                 include_initial_state:bool=True,
                 ):
        super().__init__(control_func,include_initial_state)
        self._trajectory=[]
    
    def _record(self,step:int,frame:torch.tensor):
        if not isinstance(self._trajectory,torch.Tensor):
            self._trajectory.append(copy.deepcopy(frame))
        else:
            raise RuntimeError("The trajectory has been finalized.")
    
    @property
    def trajectory(self):
        if len(self._trajectory)==0:
            return None
        if self.return_in_fourier:
            return torch.stack(self._trajectory,dim=1)
        else:
            return self._traj_ifft(torch.stack(self._trajectory,dim=1)).real
    
class CPURecorder(AutoRecorder):

    """
    A recorder that saves the trajectory on the CPU memory.
        This is useful for large trajectories that may not fit in GPU memory during simulation.

    Args:
        control_func (Optional[Callable[[int],bool]]): A function that takes a step as input and returns a boolean indicating whether to record the trajectory at that step.
        include_initial_state (bool): If True, the initial state will be included in the trajectory.
    """
    
    def __init__(self,
                 control_func:Optional[Callable[[int],bool]]=None,
                 include_initial_state:bool=True
                 ):
        super().__init__(control_func,include_initial_state)
    
    def _record(self,step:int,frame:torch.tensor):
        if frame.is_cpu:
            self._trajectory.append(copy.deepcopy(frame))
        else:
            self._trajectory.append(frame.cpu())

class DiskRecorder(_TrajRecorder):

    """
    A recorder that saves the trajectory on the disk.
        This is useful for large trajectories that may not fit in GPU memory during simulation.
        The trajectory is saved in a temporary cache and then written to disk at specified intervals.
    
    Args:
        control_func (Optional[Callable[[int],bool]]): A function that takes a step as input and returns a boolean indicating whether to record the trajectory at that step.
        include_initial_state (bool): If True, the initial state will be included in the trajectory.
        cache_dir (Optional[str]): The directory where the trajectory will be saved. Default is "./saved_traj/".
        cache_freq (int): The frequency at which to save the trajectory to disk. Default is 1.
        temp_cache_loc (Literal["auto","cpu"]): The location of the temporary cache. Default is "cpu".
        save_format (Literal["numpy","torch"]): The format in which to save the trajectory. Default is "torch".
    """
    
    def __init__(self,
                 control_func:Optional[Callable[[int],bool]]=None,
                 include_initial_state:bool=True,
                 cache_dir:Optional[str]=None,
                 cache_freq:int=1,
                 temp_cache_loc:Literal["auto","cpu"]="cpu",
                 save_format:Literal["numpy","torch"]="torch",
                 ):
        super().__init__(control_func,include_initial_state)
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
            if not self.return_in_fourier:
                temp_cache=self._traj_ifft(temp_cache).real
            if self.save_format=="numpy":
                np.save(self.cache_dir+f"temp_cache_{step}",temp_cache.numpy()) 
            else:
                torch.save(temp_cache,self.cache_dir+f"temp_cache_{step}")     
            self._trajectory=[]      
    
    @property
    def trajectory(self):
        return None