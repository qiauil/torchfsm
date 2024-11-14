from torchfsm.operator._base import LinearCoef,LinearOperator,CoreGenerator,NonlinearFunc
from torchfsm.mesh import FourierMesh
import torch
from torch import Tensor

class _VorticityConvectionCore(NonlinearFunc):
    
    def __call__(self, u_fft: Tensor, 
                 f_mesh: FourierMesh, 
                 n_channel: int, 
                 u: Tensor | None,
                 return_in_fourier=True) -> Tensor:
        psi= -u_fft * f_mesh.invert_laplacian()
        ux=f_mesh.ifft(f_mesh.grad(1,1)*psi).real
        uy=f_mesh.ifft(-f_mesh.grad(0,1)*psi).real
        grad_x_w=f_mesh.ifft(f_mesh.grad(0,1)*u_fft).real
        grad_y_w=f_mesh.ifft(f_mesh.grad(1,1)*u_fft).real
        if return_in_fourier:
            return f_mesh.fft(
                ux*grad_x_w+uy*grad_y_w
            )
        else:
            return ux*grad_x_w+uy*grad_y_w
        
class _VorticityConvectionGenerator(CoreGenerator):
    
    def __call__(self, f_mesh: FourierMesh, n_channel: int) -> LinearCoef | NonlinearFunc:
        if f_mesh.n_dim != 2 or n_channel != 1:
            raise ValueError("Only vorticity in 2Dmesh is supported")
        return _VorticityConvectionCore()
    
class VorticityConvection(LinearOperator):
        
        def __init__(self) -> None:
            super().__init__(_VorticityConvectionGenerator())