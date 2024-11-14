from .._base import CoreGenerator,NonlinearFunc, NonlinearOperator, LinearCoef,LinearOperator
from ...mesh import FourierMesh
import torch
from torch import Tensor
from typing import Optional

class _VorticityConvectionCore(NonlinearFunc):

    def __call__(self, 
                 u_fft:torch.Tensor,
                 f_mesh:FourierMesh,
                 n_channel:int,
                 u:Optional[torch.Tensor]) -> torch.Tensor:
        return f_mesh.fft(self.spatial_value(u_fft,f_mesh,n_channel,u))
    
    def spatial_value(self, 
                 u_fft:torch.Tensor,
                 f_mesh:FourierMesh,
                 n_channel:int,
                 u:Optional[torch.Tensor]) -> torch.Tensor:
        psi= -u_fft * f_mesh.invert_laplacian()
        ux=f_mesh.ifft(f_mesh.grad(1,1)*psi).real
        uy=f_mesh.ifft(-f_mesh.grad(0,1)*psi).real
        grad_x_w=f_mesh.ifft(f_mesh.grad(0,1)*u_fft).real
        grad_y_w=f_mesh.ifft(f_mesh.grad(1,1)*u_fft).real
        return ux*grad_x_w+uy*grad_y_w
        
class _VorticityConvectionGenerator(CoreGenerator):
    
    def __call__(self, f_mesh: FourierMesh, n_channel: int) -> NonlinearFunc:
        if f_mesh.n_dim != 2 or n_channel != 1:
            raise ValueError("Only vorticity in 2Dmesh is supported")
        return _VorticityConvectionCore()
    
class VorticityConvection(NonlinearOperator):
        
    def __init__(self) -> None:
        super().__init__(_VorticityConvectionGenerator())

class _Vorticity2VelocityCore(LinearCoef):

    def __call__(self, f_mesh, n_channel):
        return -1*f_mesh.invert_laplacian()*torch.cat([f_mesh.grad(1,1).repeat([1,1,f_mesh.mesh_info[0][-1],1]),
                                                       -f_mesh.grad(0,1).repeat([1,1,1,f_mesh.mesh_info[1][-1]])],dim=1)
    
class _Vorticity2VelocityGenerator(CoreGenerator):

    def __call__(self, f_mesh: FourierMesh, n_channel: int) -> NonlinearFunc:
        if f_mesh.n_dim != 2 or n_channel != 1:
            raise ValueError("Only vorticity in 2Dmesh is supported")
        return _Vorticity2VelocityCore()
    
class Vorticity2Velocity(LinearOperator):
    
    def __init__(self):
        super().__init__(_Vorticity2VelocityGenerator())