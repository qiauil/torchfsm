
from ._base import LinearOperator,NonlinearFunc
from ..mesh import FourierMesh
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
    
class ImplicitSource(LinearOperator):
        
    def __init__(self, source_func: Optional[Callable]=None) -> None:
        super().__init__(_ImplicitSourceCore(source_func))
