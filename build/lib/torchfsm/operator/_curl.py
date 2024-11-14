import torch
from torch import Tensor
from ..mesh import FourierMesh
from ._base import LinearCoef,NonlinearOperator,CoreGenerator,NonlinearFunc

class _Curl2DCore(NonlinearFunc):
    
    def __call__(self, u_fft: Tensor, f_mesh: FourierMesh, n_channel: int, u: Tensor | None) -> Tensor:
        return f_mesh.grad(0,1)*u_fft[:,1:2,...]-f_mesh.grad(1,1)*u_fft[:,0:1,...]
         
class _Curl3DCore(NonlinearFunc):
    def __call__(self, u_fft: Tensor, f_mesh: FourierMesh, n_channel: int, u: Tensor | None) -> Tensor:
         return torch.cat(
                [
                    f_mesh.grad(1,1)*u_fft[:,2:3,...]-f_mesh.grad(2,1)*u_fft[:,1:2,...],
                    f_mesh.grad(2,1)*u_fft[:,0:1,...]-f_mesh.grad(0,1)*u_fft[:,2:3,...],
                    f_mesh.grad(0,1)*u_fft[:,1:2,...]-f_mesh.grad(1,1)*u_fft[:,0:1,...]
                ],
                dim=1
            )
        
class _CurlGenerator(CoreGenerator):
    
    def __call__(self, f_mesh: FourierMesh, n_channel: int) -> LinearCoef | NonlinearFunc:
        if f_mesh.n_dim != n_channel:
            raise ValueError(f"div operator only works for vector field with the same dimension as mesh")
        if n_channel>3 or n_channel<2:
            raise ValueError(f"div operator only works for 2D or 3D vector field")
        if n_channel==2:
            return _Curl2DCore()
        else:
            return _Curl3DCore()
    
class Curl(NonlinearOperator):
        
        def __init__(self) -> None:
            super().__init__(_CurlGenerator())
