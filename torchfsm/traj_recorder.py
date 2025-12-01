from typing import Callable, Optional, Literal
from warnings import warn
from ._utils import default
import torch
import numpy as np


class IntervalController:
    """
    A class to control the recording of trajectories at specified intervals.
        This class can be used as an input for the `control_func` parameters of recorder objects.

    Args:
        interval (int): The interval at which to record the trajectory.
        start (int): The step at which to start recording the trajectory.
    """

    def __init__(
        self,
        interval: int = 1,
        start: int = 0,
    ) -> None:
        self.start = start
        self.interval = interval

    def __call__(self, step: int) -> bool:
        return step >= self.start and (step - self.start) % self.interval == 0


class _TrajRecorder:
    """
    A base class for trajectory recorders.
        A recorder is an object that helps to control the recording of trajectories during a simulation

    Args:
        control_func (Optional[Callable[[int],bool]]): A function that takes a step as input and returns a boolean indicating whether to record the trajectory at that step.
        include_initial_state (bool): If True, the initial state will be included in the trajectory.
    """

    def __init__(
        self,
        control_func: Optional[Callable[[int], bool]] = None,
        include_initial_state: bool = True,
    ):
        control_func = default(control_func, lambda step: True)
        if include_initial_state:
            self.control_func = lambda step: True if step == 0 else control_func(step)
        else:
            self.control_func = lambda step: False if step == 0 else control_func(step)
        self.return_in_fourier = False
        self._shutdown_flag = False
        
    def set_shutdown_flag(self):
        """
        Set the shutdown flag to True.
            This will prevent any further recording of trajectories.
        """
        self._shutdown_flag = True

    def record(self, step: int, frame: torch.tensor):
        """
        Record the trajectory at a given step.

        Args:
            step (int): The current step.
            frame (torch.tensor): The current frame to be recorded.
        """
        if self.control_func(step):
            self._record(step, frame)

    def _record(self, step: int, frame: torch.tensor):
        """
        Record the trajectory at a given step.
            This method should be implemented by subclasses.

        Args:
            step (int): The current step.
            frame (torch.tensor): The current frame to be recorded.
        """
        raise NotImplementedError

    def _traj_ifft(self, trajectory: torch.tensor):
        """
        Perform an inverse FFT on the trajectory.

        Args:
            trajectory (torch.tensor): The trajectory to be transformed.

        Returns:
            torch.tensor: The transformed trajectory.
        """
        fft_dim = tuple(-1 * (i + 1) for i in range(len(trajectory.shape) - 3))
        return torch.fft.ifftn(trajectory, dim=fft_dim)

    def _field_ifft(self, field: torch.tensor):
        """
        Perform an inverse FFT on the field.

        Args:
            field (torch.tensor): The field to be transformed.

        Returns:
            torch.tensor: The transformed field.
        """
        fft_dim = tuple(-1 * (i + 1) for i in range(len(field.shape) - 2))
        return torch.fft.ifftn(field, dim=fft_dim)

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

    def __init__(
        self,
        control_func: Optional[Callable[[int], bool]] = None,
        include_initial_state: bool = True,
    ):
        super().__init__(control_func, include_initial_state)
        self._trajectory = []

    def _record(self, step: int, frame: torch.tensor):
        if isinstance(self._trajectory, torch.Tensor) :
            self._trajectory = [] # reset trajectory if previous traj exists
        self._trajectory.append(frame.clone())

    @property
    def trajectory(self):
        if len(self._trajectory) == 0:
            return None
        self._trajectory = torch.stack(self._trajectory, dim=1)
        if self.return_in_fourier:
            return self._trajectory
        else:
            return self._traj_ifft(self._trajectory).real


class CPURecorder(AutoRecorder):
    """
    A recorder that saves the trajectory on the CPU memory.
        This is useful for large trajectories that may not fit in GPU memory during simulation.

    Args:
        control_func (Optional[Callable[[int],bool]]): A function that takes a step as input and returns a boolean indicating whether to record the trajectory at that step.
        include_initial_state (bool): If True, the initial state will be included in the trajectory.
        real_time_ifft (bool): If True, the trajectory will be transformed to real space in real time (if `return_in_fourier` is True). Default is True.
    """

    def __init__(
        self,
        control_func: Optional[Callable[[int], bool]] = None,
        include_initial_state: bool = True,
        real_time_ifft: bool = True,
    ):
        super().__init__(control_func, include_initial_state)
        self.real_time_ifft = real_time_ifft

    def _record(self, step: int, frame: torch.tensor):
        if isinstance(self._trajectory, torch.Tensor) :
            self._trajectory = [] # reset trajectory if previous traj exists
        if frame.is_cpu:
            if self.real_time_ifft and not self.return_in_fourier:
                self._trajectory.append(self._field_ifft(frame.clone()).real)
            else:
                self._trajectory.append(frame.clone())
        else:
            if self.real_time_ifft and not self.return_in_fourier:
                self._trajectory.append(self._field_ifft(frame.cpu()).real)
            else:
                self._trajectory.append(frame.cpu())

    @property
    def trajectory(self):
        if len(self._trajectory) == 0:
            return None
        self._trajectory = torch.stack(self._trajectory, dim=1)
        if self.return_in_fourier or self.real_time_ifft:
            return self._trajectory
        else:
            return self._traj_ifft(self._trajectory).real


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

    def __init__(
        self,
        control_func: Optional[Callable[[int], bool]] = None,
        include_initial_state: bool = True,
        cache_dir: Optional[str] = None,
        cache_freq: int = 1,
        temp_cache_loc: Literal["auto", "cpu"] = "cpu",
        save_format: Literal["numpy", "torch"] = "torch",
    ):
        super().__init__(control_func, include_initial_state)
        self.cache_dir = default(cache_dir, "./saved_traj/")
        self.cache_freq = cache_freq
        self._trajectory = []
        self.temp_cache_loc = temp_cache_loc
        if self.temp_cache_loc not in ["auto", "cpu"]:
            raise ValueError("temp_cache_loc must be either 'auto' or 'cpu'.")
        self.save_format = save_format
        if self.save_format not in ["numpy", "torch"]:
            raise ValueError("save_format must be either 'numpy' or 'torch'.")

    def _record(self, step: int, frame: torch.tensor):
        if len(self._trajectory)< self.cache_freq:
            if self.temp_cache_loc == "cpu" and not frame.is_cpu:
                self._trajectory.append(frame.cpu())
            else:
                self._trajectory.append(frame.clone())
        else:
            temp_cache = torch.stack(self._trajectory, dim=1)
            temp_cache = temp_cache.to("cpu") if not temp_cache.is_cpu else temp_cache
            if not self.return_in_fourier:
                temp_cache = self._traj_ifft(temp_cache).real
            if self.save_format == "numpy":
                np.save(self.cache_dir + f"temp_cache_{step}", temp_cache.numpy())
            else:
                torch.save(temp_cache, self.cache_dir + f"temp_cache_{step}")
            self._trajectory = []

    @property
    def trajectory(self):
        self._trajectory = [] # Clear the cache after saving to disk.
        return None


class RandomBatchWisedRecorder(_TrajRecorder):
    """
    A recorder that saves the trajectory at random intervals for each batch.

    Args:
        simulation_steps (int): The total number of simulation steps.
        recorder_interval (int): The interval at which to record the trajectory.
        n_recorded_frames (int): The number of frames to record for each batch. Default is 2.
    """

    def __init__(
        self,
        simulation_steps: int,
        recorder_interval: int,
        n_recorded_frames: int = 2,
    ):
        super().__init__(None, False)
        self.simulation_steps = simulation_steps
        self.recorder_interval = recorder_interval
        self.n_recorded_frames = n_recorded_frames
        self._trajectory = []
        self._batch_size = None
        self._recorded_frame_id = None

    def _initialize_recording(self, batch_size: int):
        self._batch_size = batch_size
        initial_id = np.random.randint(
            1,
            self.simulation_steps - self.recorder_interval * self.n_recorded_frames,
            size=self._batch_size,
        )
        self._recorded_frame_id = [
            initial_id + i * self.recorder_interval
            for i in range(self.n_recorded_frames)
        ]
        self._recorded_frame_id = np.stack(self._recorded_frame_id, axis=1)
        self._trajectory = [[] for _ in range(self._batch_size)]

    def record(self, step: int, frame: torch.tensor):
        """
        Record the trajectory at a given step.

        Args:
            step (int): The current step.
            frame (torch.tensor): The current frame to be recorded.
        """
        if self._batch_size is None:
            self._initialize_recording(
                batch_size=frame.shape[0],
            )
        for batch_i in range(self._batch_size):
            if step in self._recorded_frame_id[batch_i]:
                self._trajectory[batch_i].append(frame[batch_i].clone())

    @property
    def trajectory(self):
        if len(self._trajectory) == 0:
            return None
        trajs = []
        for batch_i in range(self._batch_size):
            trajs.append(torch.stack(self._trajectory[batch_i], dim=0))
        trajs = torch.stack(trajs, dim=0)
        self._trajectory = []
        self._batch_size = None # Reset for next recording
        if self.return_in_fourier:
            return trajs
        else:
            return self._traj_ifft(trajs).real
