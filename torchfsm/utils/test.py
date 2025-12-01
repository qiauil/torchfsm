from ..errors import NanSimulationError
from ..traj_recorder import _TrajRecorder
from ..operator import Operator
from ..mesh import MeshGrid, FourierMesh
from typing import Callable, Optional, Union, Sequence
import torch
import numpy as np

class _DtTestRecorder(_TrajRecorder):

    def __init__(
        self,
        control_func: Optional[Callable[[int], bool]] = None,
        include_initial_state: bool = True,
    ):
        super().__init__(control_func, include_initial_state)
        self.differences = []
        self.pre_traj= None

    def _record(self, step: int, frame: torch.tensor):
        current_frame = self._field_ifft(frame).real
        if self.pre_traj is not None:
            self.differences.append((current_frame - self.pre_traj).abs().mean().item())
        self.pre_traj = current_frame
        
    def teardown(self):
        self.differences = []
        self.pre_traj = None

    @property
    def trajectory(self):
        return self.pre_traj

def test_sim_dt(
    operator: Operator,
    u_0: torch.Tensor,
    max_sim_dt: float,
    min_sim_dt: float,
    mesh: Optional[
            Union[Sequence[tuple[float, float, int]], MeshGrid, FourierMesh]
        ] = None,
    stop_criteria: float = None,
    initial_step: int = 1,
    dt_decrement: float = 0.5,
    **kwargs
) -> tuple[dict, dict, dict]:
    """Test the simulation with varying time steps.
    
    Args:
        operator (Operator): The operator to be tested.
        u_0 (torch.Tensor): Initial condition tensor.
        max_sim_dt (float): Maximum simulation time step.
        min_sim_dt (float): Minimum simulation time step.
        mesh (Optional[Union[Sequence[tuple[float, float, int]], MeshGrid, FourierMesh]]): Mesh information or mesh object.
        stop_criteria (float, optional): Criteria to stop the simulation if reached.
        initial_step (int, optional): Initial step size for the simulation.
        dt_decrement (float, optional): Factor by which to reduce the time step on failure.
        **kwargs: Additional keyword arguments for the operator's integrate method.
    
    Returns:
        tuple[dict, dict, dict]: A tuple containing:
            - errors: Dictionary of errors for each time step.
            - mean_differences: Mean differences between frames for each time step.
            - std_differences: Standard deviations of differences for each time step.
    """
    
    
    current_dt = max_sim_dt
    recorder=_DtTestRecorder()
    current_frame = None
    previous_frame = None
    errors={}
    mean_differences={}
    std_differences={}
    while current_dt > min_sim_dt:
        print(f"Testing with dt={current_dt}")
        try:
            operator.integrate(
                u_0= u_0,
                trajectory_recorder=recorder,
                mesh=mesh,
                dt=current_dt,
                step=int(max_sim_dt // current_dt * initial_step),
                nan_check=True,
                progressive= True,
                **kwargs
            )
        except NanSimulationError:
            print(f"Simulation failed with dt={current_dt}, reducing dt.")
            current_dt *= dt_decrement
            recorder.teardown()
            continue
        mean_dif=np.mean(recorder.differences)
        std_dif=np.std(recorder.differences)
        mean_differences[current_dt] = mean_dif
        std_differences[current_dt] = std_dif
        current_frame = recorder.trajectory
        current_error = None
        if previous_frame is not None:
            current_error = (current_frame - previous_frame).abs().mean().item()
            errors[current_dt] = current_error
            if stop_criteria is not None:
                if current_error < stop_criteria:
                    print(f"Simulation converged with dt={current_dt}, error={current_error:.3e}")
                    break
        previous_frame = current_frame
        current_dt /= 2
        recorder.teardown()
        msg = ""
        if current_error is not None:
            msg += f"Error of the last frame: {current_error:.3e}; "
        msg += f"Difference between frames: {mean_dif:.3e}±{std_dif:.3e}"
        print(msg)
    print("Reach minimum dt:", current_dt)
    return errors, mean_differences, std_differences