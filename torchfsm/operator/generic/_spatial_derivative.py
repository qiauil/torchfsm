from torch import Tensor
from ...mesh import FourierMesh
from .._base import LinearCoef, LinearOperator, CoreGenerator
from ..._type import FourierTensor


class _SpatialDerivativeCore(LinearCoef):

    def __init__(self, dim_index, order) -> None:
        super().__init__()
        self.dim_index = dim_index
        self.order = order

    def __call__(
        self, f_mesh: FourierMesh, n_channel: int
    ) -> FourierTensor["B C H ..."]:
        return f_mesh.grad(self.dim_index, self.order)


class _SpatialDerivativeGenerator(CoreGenerator):

    def __init__(self, dim_index, order) -> None:
        super().__init__()
        self.dim_index = dim_index
        self.order = order

    def __call__(self, f_mesh: FourierMesh, n_channel: int) -> LinearCoef:
        if n_channel != 1:
            raise ValueError(
                "The SpatialDerivative operator only supports scalar field."
            )
        return _SpatialDerivativeCore(self.dim_index, self.order)


class SpatialDerivative(LinearOperator):
    r"""
    `SpatialDeritivate` calculates the spatial derivative of a scalar field w.r.t to a spatial dimension:
    $$
    \frac{\partial ^n}{\partial i} p
    $$
    where $i = x, y, z, \cdots$ and $n=1, 2, 3$

    Args:
        dim_index (int): The index of the spatial dimension.
        order (int): The order of the derivative.
    """

    def __init__(self, dim_index: int, order: int) -> None:
        super().__init__(_SpatialDerivativeGenerator(dim_index, order))
