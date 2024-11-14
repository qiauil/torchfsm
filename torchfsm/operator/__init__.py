import torch
from typing import Union,Sequence
from ..utils import default
from ..mesh import FourierMesh,MeshGrid

from ._base import Operator, OperatorLike, ExplicitSource, LinearCoef, NonlinearFunc, LinearOperator, NonlinearOperator, CoreGenerator, check_value_with_mesh
from .generic import *
from .dedicated import *

def run_operators(u:torch.Tensor,
                  operators:Sequence[Operator],
                  mesh: Union[Sequence[tuple[float, float, int]],MeshGrid,FourierMesh]) -> torch.Tensor:
    if not isinstance(mesh,FourierMesh):
        mesh=FourierMesh(mesh)
    u_fft=mesh.fft(u)
    def _run_operator(operator:Operator):
        return operator(u_fft=u_fft,mesh=mesh)
    return map(_run_operator,operators)