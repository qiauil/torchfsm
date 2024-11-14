from torch import Tensor
from ..mesh import FourierMesh
from ._base import LinearCoef,LinearOperator,CoreGenerator
class _GradCore(LinearCoef):
    
    def __call__(self, f_mesh: FourierMesh, n_channel: int) -> Tensor:
        return f_mesh.nabla_vector(1)
    
class _GradGenerator(CoreGenerator):
    
    def __call__(self, f_mesh: FourierMesh, n_channel: int) -> LinearCoef:
        if n_channel!=1:
            raise ValueError("The Grad operator only supports scalar field.")
        return _GradCore()
        
class Grad(LinearOperator):
    
    r"""
    `Grad` calculates the spatial gradient of a scalar field:
    $$
    \nabla p = \left[\begin{matrix}
    \frac{\partial p}{\partial x} \\
    \frac{\partial p}{\partial y} \\
    \cdots \\
    \frac{\partial p}{\partial i} \\
    \end{matrix}
    \right]
    $$
    """
    
    def __init__(self) -> None:
        super().__init__(_GradGenerator())