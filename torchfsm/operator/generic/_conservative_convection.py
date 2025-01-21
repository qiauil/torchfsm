from torch import Tensor
from ...mesh import FourierMesh
from .._base import LinearCoef, NonlinearOperator, CoreGenerator, NonlinearFunc
from ..._type import FourierTensor, SpatialTensor


class _ConservativeConvectionCore(NonlinearFunc):

    def __init__(self):
        super().__init__()

    def __call__(
        self,
        u_fft: FourierTensor["B C H ..."],
        f_mesh: FourierMesh,
        n_channel: int,
        u: SpatialTensor["B C H ..."] | None,
    ) -> FourierTensor["B C H ..."]:
        if u is None:
            u = f_mesh.ifft(u_fft).real
        uu = u.unsqueeze(2) * u.unsqueeze(1)
        uu_fft = f_mesh.fft(uu)
        return (f_mesh.nabla_vector(1).unsqueeze(2) * uu_fft).sum(1)


class _ConservativeConvectionGenerator(CoreGenerator):

    def __call__(
        self, f_mesh: FourierMesh, n_channel: int
    ) -> LinearCoef | NonlinearFunc:
        if f_mesh.n_dim != n_channel:
            raise ValueError(
                f"div operator only works for vector field with the same dimension as mesh"
            )
        return _ConservativeConvectionCore()


class ConservativeConvection(NonlinearOperator):
    """
    `ConservativeConvection` calculates the convection of a vector field on itself.
    $$
    \nabla \cdot \mathbf{u}\mathbf{u}
    =
    \left[\begin{matrix}
    \sum_{i=0}^I \frac{\partial u_i u_x }{\partial i} \\
    \sum_{i=0}^I \frac{\partial u_i u_y }{\partial i} \\
    \cdots\\
    \sum_{i=0}^I \frac{\partial u_i u_I }{\partial i} \\
    \end{matrix}
    \right]
    $$
    """

    def __init__(self) -> None:
        super().__init__(_ConservativeConvectionGenerator())
