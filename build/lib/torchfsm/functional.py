from .operator import  Convection, Laplacian, Div, Curl, OperatorLike, Vorticity2Velocity
from .mesh import FourierMesh,MeshGrid
from typing import Optional,Union,Sequence
import torch

def _get_fft_with_mesh(value:Optional[torch.Tensor]=None,
                       value_fft:Optional[torch.Tensor]=None,
                          mesh:Union[Sequence[tuple[float, float, int]],MeshGrid,FourierMesh]=None,):
    if value_fft is None:
        f_mesh=mesh if isinstance(mesh,FourierMesh) else FourierMesh(mesh,device=value.device,dtype=value.dtype)
        value_fft=f_mesh.fft(value)
    else:
        f_mesh=mesh if isinstance(mesh,FourierMesh) else FourierMesh(mesh,device=value_fft.device,dtype=value_fft.dtype)
    return value_fft,f_mesh

def vorticity2velocity(vorticity:Optional[torch.Tensor]=None,
                          vorticity_fft:Optional[torch.Tensor]=None,
                          mesh:Optional[Union[Sequence[tuple[float, float, int]],MeshGrid,FourierMesh]]=None,):
    return Vorticity2Velocity()(u=vorticity,u_fft=vorticity_fft,mesh=mesh)

def velocity2pressure(velocity:Optional[torch.Tensor]=None,
                      velocity_fft:Optional[torch.Tensor]=None,
                      mesh:Optional[Union[Sequence[tuple[float, float, int]],MeshGrid,FourierMesh]]=None,
                      external_force:Optional[OperatorLike]=None,):
    velocity_fft,f_mesh=_get_fft_with_mesh(value=velocity,value_fft=velocity_fft,mesh=mesh)
    convection=Convection()(u_fft=velocity_fft,mesh=f_mesh,return_in_fourier=True)
    if external_force is not None:
        convection-=external_force(u_fft=velocity_fft,mesh=f_mesh,return_in_fourier=True)
    convection=Div()(u_fft=convection,mesh=f_mesh,return_in_fourier=True)
    return Laplacian().solve(b_fft=-1*convection,mesh=f_mesh,n_channel=1)

def vorticity2pressure(vorticity:Optional[torch.Tensor]=None,
                      vorticity_fft:Optional[torch.Tensor]=None,
                       mesh:Optional[Union[Sequence[tuple[float, float, int]],MeshGrid,FourierMesh]]=None,
                      external_force:Optional[OperatorLike]=None):
    vorticity_fft,f_mesh=_get_fft_with_mesh(value=vorticity,value_fft=vorticity_fft,mesh=mesh)
    velocity_fft=Vorticity2Velocity()(u_fft=vorticity_fft,mesh=f_mesh,return_in_fourier=True)
    convection=Convection()(u_fft=velocity_fft,mesh=f_mesh,return_in_fourier=True)
    if external_force is not None:
        convection-=external_force(u_fft=vorticity_fft,mesh=f_mesh,return_in_fourier=True)
    convection=Div()(u_fft=convection,mesh=f_mesh,return_in_fourier=True)
    return Laplacian().solve(b_fft=-1*convection,mesh=f_mesh,n_channel=1)