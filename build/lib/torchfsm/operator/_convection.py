from torch import Tensor
from ..mesh import FourierMesh
from ._base import LinearCoef,NonlinearOperator,CoreGenerator,NonlinearFunc
from functools import lru_cache


class _ConvectionCore(NonlinearFunc):
    
    def __call__(self, u_fft: Tensor, f_mesh: FourierMesh, n_channel: int, u: Tensor | None) -> Tensor:
        return f_mesh.fft(self.spatial_value(u_fft,f_mesh,n_channel,u))
    
    def spatial_value(self, u_fft: Tensor, f_mesh: FourierMesh, n_channel: int, u: Tensor | None):
        if u is None:
            u=f_mesh.ifft(u_fft).real
        nabla_u=f_mesh.nabla_vector(1).unsqueeze(2)*u_fft.unsqueeze(1)
        return (u.unsqueeze(2)*f_mesh.ifft(nabla_u).real).sum(1)
        # another way to calculate convection:
        # return sum([u[:,i:i+1,...]*f_mesh.ifft(f_mesh.grad(i,1)*u_fft).real for i in range(n_channel)])
        
class _ConvectionGenerator(CoreGenerator):
    
    def __call__(self, f_mesh: FourierMesh, n_channel: int) -> LinearCoef | NonlinearFunc:
        if f_mesh.n_dim != n_channel:
            raise ValueError(f"convection operator only works for vector field with the same dimension as mesh")
        return _ConvectionCore()
    
class Convection(NonlinearOperator):
    
    """
    `Convection` calculates the convection of a vector field on itself if the vector field is divergence free, i.e., $\nabla \cdot \mathbf{u} =0$.
    $$
    \mathbf{u} \cdot \nabla  \mathbf{u}
    =
    \left[\begin{matrix}
    \sum_{i=0}^I u_i\frac{\partial u_x }{\partial i} \\
    \sum_{i=0}^I u_i\frac{\partial u_y }{\partial i} \\
    \cdots\\
    \sum_{i=0}^I u_i\frac{\partial u_I }{\partial i} \\
    \end{matrix}
    \right]
    $$
    """
        
    def __init__(self) -> None:
        super().__init__(_ConvectionGenerator())