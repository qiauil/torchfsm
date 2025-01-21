import torch
from torch import Tensor
from ...mesh import FourierMesh
from .._base import LinearCoef, NonlinearOperator, CoreGenerator, NonlinearFunc
from ..._type import FourierTensor, SpatialTensor


class _DivCore(NonlinearFunc):
    def __init__(self):
        super().__init__(False)

    def __call__(
        self,
        u_fft: FourierTensor["B C H ..."],
        f_mesh: FourierMesh,
        n_channel: int,
        u: SpatialTensor["B C H ..."] | None,
    ) -> FourierTensor["B C H ..."]:
        return torch.sum(f_mesh.nabla_vector(1) * u_fft, dim=1, keepdim=True)


class _DivGenerator(CoreGenerator):

    def __call__(
        self, f_mesh: FourierMesh, n_channel: int
    ) -> LinearCoef | NonlinearFunc:
        if f_mesh.n_dim != n_channel:
            raise ValueError(
                f"div operator only works for vector field with the same dimension as mesh"
            )
        return _DivCore()


class Div(NonlinearOperator):
    r"""
    `Div` calculates the divergence of a vector field:
    $$
    \nabla \cdot \mathbf{u} = \sum_i \frac{\partial u_i}{\partial i}
    $$
    """

    def __init__(self) -> None:
        super().__init__(_DivGenerator())
