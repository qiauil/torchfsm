from ._div import _DivCore
from ...mesh import FourierMesh
from .._base import NonlinearOperator, NonlinearFunc
from ..._type import FourierTensor, SpatialTensor
from typing import Optional


class _AdvectionCore(NonlinearFunc):
    r"""
    Implementation of the Advection operator.
    """

    def __init__(self,
                 velocity:SpatialTensor["B C H ..."]):
        super().__init__(False)
        self.velocity = velocity
        self.div=_DivCore()

    def __call__(
        self,
        u_fft: FourierTensor["B C H ..."],
        f_mesh: FourierMesh,
        u: Optional[SpatialTensor["B C H ..."]] = None,
    ) -> FourierTensor["B C H ..."]:
        assert self.velocity.shape[1] == f_mesh.n_dim, \
            f"advection operator only works for scalar field with velocity dimension {f_mesh.n_dim}, "\
            f"but got velocity with dimension {self.velocity.shape[1]}"
        if u is None:
            u = f_mesh.ifft(u_fft).real
        return self.div(f_mesh.fft(self.velocity*u),f_mesh)

class Advection(NonlinearOperator):
    r"""
    `Advection` calculates the advection of a scalar field by a constant velocity field.
        If your velocity field is constant in space, please consider using `LinearAdvection` operator to allow you use larger simulation dt.
        It is defined as $\nabla \cdot (\phi\mathbf{u}) = \sum_{i=0}^I \frac{\phi\partial u_i}{\partial i}$
        where $\mathbf{u}$ is the velocity field.
        Note that this class is an operator wrapper. The actual implementation of the operator is in the `_AdvectionCore` class.

    Args:
        velocity (SpatialTensor["B C H ..."]): The velocity field used for advection. Please not that your velocity should be smooth enough to avoid aliasing error in Fourier space.
    """

    def __init__(self,
                 velocity:SpatialTensor["B C H ..."]
                 ) -> None:
        super().__init__(_AdvectionCore(velocity))