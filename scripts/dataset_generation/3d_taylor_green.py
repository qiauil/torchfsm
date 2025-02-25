#usr/bin/python3
# -*- coding: UTF-8 -*-
from torchfsm.operator import Laplacian,Operator,NSPressureConvection
from torchfsm.mesh import MeshGrid
from torchfsm.traj_recorder import CPURecorder,IntervalController
from torchfsm.integrator import RKIntegrator
import numpy as np
import torch,h5py,os

RESOLUTION=128
REYNOLDS_NUMBER={'train':[2000,4000,6000],'test':[1000,3000,5000]}
# Set to True to use Euler integrator
# Euler integrator requires less memory but smaller time step
USE_EULER_INTEGRATOR=False 
DT_FUNC=lambda Re:0.005*1000/Re
# Simulation time, unit: second
SIMULATION_TIME=20
# The time to start recording, unit: second
# The flow may not be well developed at the beginning
START_RECORD_TIME=10
# Save Interval, unit: second
SAVE_INTERVAL=0.1
SAVE_DTYPE=np.float16
DEVICE='cuda:0'
SAVE_DIR="./"

def save_as_pbdl(
    data:np.ndarray,
    out_dir:str,
    out_name:str,
    metadata:dict,
    ):
    print(f"Shape simulation data: {data.shape}")

    const = np.arange(data.shape[0])
    const = const[:, np.newaxis]

    with h5py.File(os.path.join(out_dir, out_name + ".hdf5"), "w") as h5py_file:
        group = h5py_file.create_group("sims", track_order=True)
        for key in metadata:
            if key != "Constants Sim":
                group.attrs[key] = metadata[key]
        for s in range(data.shape[0]):
            dataset = h5py_file.create_dataset("sims/sim%d" % (s), data=data[s])
            constant=metadata["Constants Sim"][s]
            for key,item in constant.items():
                dataset.attrs[key]=item
        h5py_file.close()

meta_data={
        "PDE": "Navier-Stokes: Taylor-Green Vortex",
        "Dimension": 3,
        "Fields Scheme": "VVV",
        "Fields": ["Velocity X", "Velocity Y", "Velocity Z"],
        "Domain Extent": 2*np.pi,
        "Resolution": [RESOLUTION]*3,
        "Time Steps": 0.0, # Different traj may have different time steps, please the values in constants
        "Dt": 0.0, # Different traj may have different time steps, please the values in constants
        "Boundary Conditions Order": ["x negative", "x positive", "y negative", "y positive" "z negative", "z positive"],
        "Boundary Conditions": ["periodic", "periodic", "periodic", "periodic", "periodic", "periodic"],
        "Constants": ["Re", "Dt", "Time Steps"],
    }

# Define the Navier-Stokes operator
def NavierStokes(nu:float)->Operator:
    return NSPressureConvection()+nu*Laplacian()
# Define the Mesh
mesh=MeshGrid([(0,2*np.pi,RESOLUTION)]*3,
               device=DEVICE)
x,y,z=mesh.bc_mesh_grid()
# Define the initial condition from
# https://doi.org/10.1016/j.cpc.2016.02.005
# It is also possible to change the initial condition, but could be tricky
u=torch.cat(
    [torch.sin(x)*torch.cos(y)*torch.cos(z),
     -torch.cos(x)*torch.sin(y)*torch.cos(z),
     torch.zeros_like(x)],
    dim=1)
# simulate the flow
for key,item in REYNOLDS_NUMBER.items():
    trajs=[]
    constants=[]
    for Re in item:
        print(f"Simulating Re={Re} in {key}")
        ns=NavierStokes(1./Re)
        if USE_EULER_INTEGRATOR:
            ns.set_integrator(RKIntegrator.Euler)
        dt=DT_FUNC(Re)
        step=int(SIMULATION_TIME/dt)
        save_interval=int(SAVE_INTERVAL/dt)
        traj=ns.integrate(
            u,
            dt=dt,
            step=step,
            mesh=mesh,
            trajectory_recorder=CPURecorder(control_func=IntervalController(interval=save_interval,
                                                                            start=int(START_RECORD_TIME/dt))),
            progressive=True
        )
        if torch.isnan(traj).any():
            raise ValueError("Nan value detected in the simulation")
        trajs.append(SAVE_DTYPE(traj.numpy()))
        constants.append({'Re':Re, "Dt":dt*save_interval, "Time Steps": step})
    trajs=np.concatenate(trajs,axis=0)
    meta_data["Constants Sim"]=constants
    save_as_pbdl(
        data=trajs,
        out_dir=SAVE_DIR,
        out_name=f'ns_taylor_green_3d_{key}',
        metadata=meta_data
    )    