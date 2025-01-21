import torch
from torch import Tensor
from ...mesh import FourierMesh
from .._base import LinearCoef,LinearOperator
from ..._type import FourierTensor

class _LaplacianCore(LinearCoef):
    
    def __call__(self, 
                 f_mesh: FourierMesh, 
                 n_channel: int) -> FourierTensor["B C H ..."]:
        return torch.cat([f_mesh.laplacian()]*n_channel,dim=1)
    
class Laplacian(LinearOperator):
    r"""
    `Laplacian` calculates the Laplacian of a vector field:
    $$
    \nabla \cdot (\nabla\mathbf{u}) = 
    \left[\begin{matrix}
    \sum_i \frac{\partial^2 u_x}{\partial i^2 } \\
    \sum_i \frac{\partial^2 u_y}{\partial i^2 } \\
    \cdots \\
    \sum_i \frac{\partial^2 u_i}{\partial i^2 } \\
    \end{matrix}
    \right]
    $$
    """
    
    def __init__(self) -> None:
        super().__init__( _LaplacianCore())