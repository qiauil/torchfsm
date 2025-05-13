from torch import Tensor
from ...mesh import FourierMesh
from .._base import LinearCoef, NonlinearOperator, CoreGenerator, NonlinearFunc
from ..._type import FourierTensor, SpatialTensor
from typing import Optional, Union


class _ConservativeConvectionCore(NonlinearFunc):

    r"""
    Implementation of the Conservative Convection operator.
    """

    def __init__(self):
        super().__init__()

    def __call__(
        self,
        u_fft: FourierTensor["B C H ..."],
        f_mesh: FourierMesh,
        u: Optional[SpatialTensor["B C H ..."]] = None,
    ) -> FourierTensor["B C H ..."]:
        if u is None:
            u = f_mesh.ifft(u_fft).real
        uu = u.unsqueeze(2) * u.unsqueeze(1)
        uu_fft = f_mesh.fft(uu)
        return (f_mesh.nabla_vector(1).unsqueeze(2) * uu_fft).sum(1)


class _ConservativeConvectionGenerator(CoreGenerator):

    r"""
    Generator of the Conservative Convection operator. It ensures that the operator is only applied to vector fields with the same dimension as the mesh.
    """

    def __call__(
        self, f_mesh: FourierMesh, n_channel: int
    ) -> Union[LinearCoef, NonlinearFunc]:
        if f_mesh.n_dim != n_channel:
            raise ValueError(
                f"div operator only works for vector field with the same dimension as mesh"
            )
        return _ConservativeConvectionCore()


class ConservativeConvection(NonlinearOperator):
    r"""
    `ConservativeConvection` calculates the convection of a vector field on itself. 
        It is defined as $\nabla \cdot \mathbf{u}\mathbf{u}=\left[\begin{matrix}\sum_{i=0}^I \frac{\partial u_i u_x }{\partial i} \\\sum_{i=0}^I \frac{\partial u_i u_y }{\partial i} \\\cdots\\\sum_{i=0}^I \frac{\partial u_i u_I }{\partial i} \\\end{matrix}\right]$.
        This operator only works for vector fields with the same dimension as the mesh.
        Note that this class is an operator wrapper. The real implementation of the source term is in the `_ConservativeConvectionCore` class.
    """

    def __init__(self) -> None:
        super().__init__(_ConservativeConvectionGenerator())
