import torch
from torch import Tensor
from ..mesh import FourierMesh
from ._base import LinearCoef,NonlinearOperator,CoreGenerator,NonlinearFunc

class _DivCore(NonlinearFunc):
    
    def __call__(self, u_fft: Tensor, f_mesh: FourierMesh, n_channel: int, u: Tensor | None) -> Tensor:
        return torch.sum(
            f_mesh.nabla_vector(1)*u_fft,
            dim=1,
            keepdim=True
        )
        
class _DivGenerator(CoreGenerator):
    
    def __call__(self, f_mesh: FourierMesh, n_channel: int) -> LinearCoef | NonlinearFunc:
        if f_mesh.n_dim != n_channel:
            raise ValueError(f"div operator only works for vector field with the same dimension as mesh")
        return _DivCore()
    
class Div(NonlinearOperator):
    r"""
    `Div` calculates the divergence of a vector field:
    $$
    \nabla \cdot \mathbf{u} = \sum_i \frac{\partial u_i}{\partial i}
    $$
    """
        
    def __init__(self) -> None:
        super().__init__(_DivGenerator())