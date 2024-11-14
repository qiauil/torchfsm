
from torchfsm.operator._base import Operator,NonlinearFunc
from torchfsm.mesh import FourierMesh
import torch
from torch import Tensor
from typing import Callable,Optional


class _ImplicitSourceCore(NonlinearFunc):
    
    def __init__(self, source_func: Optional[Callable]=None) -> None:
        super().__init__()
        self.source_func = source_func
        
    def __call__(self, u_fft: Tensor, 
                 f_mesh: FourierMesh, 
                 n_channel: int, 
                 u: Tensor | None) -> Tensor:
        if self.source_func is None:
            return u_fft
        else:
            if u is None:
                u = f_mesh.ifft(u_fft).real
            return f_mesh.fft(self.source_func(u))
    
class ImplicitSource(Operator):
        
    def __init__(self, source_func: Optional[Callable]=None) -> None:
        super().__init__()
        self.add_generator(lambda f_mesh, n_channel: _ImplicitSourceCore(source_func))
            
class _ExplicitSourceCore(NonlinearFunc):
        
    def __init__(self, source:torch.Tensor) -> None:
        super().__init__()
        fft_dim=[i+2 for i in range(source.dim()-2)]
        self.source=torch.fft.fftn(source, dim=fft_dim)
    
    def __call__(self, 
                 u_fft: Tensor, 
                f_mesh: FourierMesh, 
                n_channel: int, 
                u: Tensor | None) -> Tensor:
        return self.source
    
class ExplicitSource(Operator):
            
    def __init__(self, source:torch.Tensor) -> None:
        super().__init__()
        self.add_generator(lambda f_mesh, n_channel: _ExplicitSourceCore(source))
