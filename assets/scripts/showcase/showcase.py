from torchfsm import *
import torch

def Burgers(nu:float) -> Operator:
    return nu*Laplacian()-Convection()
burgers=Burgers(0.01)
mesh=MeshGrid([(0,2*torch.pi,128)],device="cuda:0")
burgers.integrate(
    u_0=torch.sin(mesh.bc_mesh_grid()),
  	mesh=mesh,
    dt=0.01,step=200
)