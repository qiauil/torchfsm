
from .operator import Laplacian,ImplicitSource,ExplicitSource,Curl
from .mesh import FourierMesh,MeshGrid,mesh_shape
from .integrator import ETDRKIntegrator
import torch
from typing import Union,Sequence,Optional

def diffused_noise(
    mesh: Union[Sequence[tuple[float, float, int]],MeshGrid,FourierMesh],
    diffusion_coef:float=1.0,
    zero_centered:bool=True,
    unit_variance:bool=True,
    unit_magnitude:bool=True,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    n_batch:int=1,
    n_channel:int=1,
    ):
    if device is None and (isinstance(mesh,FourierMesh) or isinstance(mesh,MeshGrid)):
        device=mesh.device
    if dtype is None and (isinstance(mesh,FourierMesh) or isinstance(mesh,MeshGrid)):
        dtype=mesh.dtype
    u_0=torch.randn(*mesh_shape(mesh,n_batch=n_batch,n_channel=n_channel),device=device,dtype=dtype)
    diffusion=diffusion_coef*Laplacian()
    u_0=diffusion.integrate(u_0,
                    dt=1,
                    step=1,
                    mesh=mesh)
    if zero_centered:
        u_0=u_0-u_0.mean()
    if unit_variance:
        u_0=u_0/u_0.std()
    if unit_magnitude:
        u_0=u_0/u_0.abs().max()
    return u_0

def kolm_force(x:torch.Tensor,
               drag_coef:float=-0.1,
               k:float=4.0,
               length_scale:float=1.0,
               ):
    return drag_coef*ImplicitSource()-ExplicitSource(k*torch.cos(k*length_scale*x))