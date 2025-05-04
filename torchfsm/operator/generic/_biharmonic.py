import torch
from torch import Tensor
from ...mesh import FourierMesh
from .._base import LinearCoef, LinearOperator
from ..._type import FourierTensor


class _BiharmonicCore(LinearCoef):
    r"""
    Implementation of the Biharmonic operator.
    """

    def __call__(
        self, f_mesh: FourierMesh, n_channel: int
    ) -> FourierTensor["B C H ..."]:
        return torch.cat([f_mesh.laplacian() * f_mesh.laplacian()] * n_channel, dim=1)


class Biharmonic(LinearOperator):
    r"""
    `Biharmonic` calculates the Biharmonic of a vector field. 
        It is defined as $\nabla^4\mathbf{u}=\left[\begin{matrix}(\sum_{i=0}^I\frac{\partial^2}{\partial i^2 })(\sum_{j=0}^I\frac{\partial^2}{\partial j^2 })u_x \\ (\sum_{i=0}^I\frac{\partial^2}{\partial i^2 })(\sum_{j=0}^I\frac{\partial^2}{\partial j^2 })u_y \\ \cdots \\ (\sum_{i=0}^I\frac{\partial^2}{\partial i^2 })(\sum_{j=0}^I\frac{\partial^2}{\partial j^2 })u_i \\ \end{matrix} \right]$
        Note that this class is an operator wrapper. The real implementation of the source term is in the `_BiharmonicCore` class.    
        
    """

    def __init__(self) -> None:
        super().__init__(_BiharmonicCore())
