from typing import Union,Sequence
from ..mesh import FourierMesh,MeshGrid

from ._base import Operator, OperatorLike, ExplicitSource, LinearCoef, NonlinearFunc, LinearOperator, NonlinearOperator, CoreGenerator, check_value_with_mesh
from .generic import *
from .dedicated import *

from .._type import SpatialTensor

def run_operators(u:SpatialTensor["B C H ..."],
                  operators:Sequence[Operator],
                  mesh: Union[Sequence[tuple[float, float, int]],MeshGrid,FourierMesh]) -> SpatialTensor["B C H ..."]:
    r"""
    Run a sequence of operators on the input tensor.
    
    Args:
        u (SpatialTensor): Input tensor of shape (B, C, H, ...).
        operators (Sequence[Operator]): Sequence of operators to be applied.
        mesh (Union[Sequence[tuple[float, float, int]],MeshGrid,FourierMesh]): Mesh information or mesh object.

    Returns:
        SpatialTensor: Resulting tensor after applying the operators.
    """
    if not isinstance(mesh,FourierMesh):
        mesh=FourierMesh(mesh)
    u_fft=mesh.fft(u)
    def _run_operator(operator:Operator):
        return operator(u_fft=u_fft,mesh=mesh)
    return map(_run_operator,operators)