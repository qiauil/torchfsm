from ...mesh import FourierMesh
from .._base import LinearCoef, LinearOperator, CoreGenerator
from ..._type import FourierTensor


class _GradCore(LinearCoef):

    r"""
    Implementation of the Grad operator.
    """

    def __call__(
        self, f_mesh: FourierMesh, n_channel: int
    ) -> FourierTensor["B C H ..."]:
        return f_mesh.nabla_vector(1)


class _GradGenerator(CoreGenerator):
    r"""
    Generator of the Grad operator.
        It ensures that grad only works for scalar field.
    """

    def __call__(self, f_mesh: FourierMesh, n_channel: int) -> LinearCoef:
        if n_channel != 1:
            raise ValueError("The Grad operator only supports scalar field.")
        return _GradCore()


class Grad(LinearOperator):
    r"""
    `Grad` calculates the spatial gradient of a scalar field.
        It is defined as $\nabla p = \left[\begin{matrix}\frac{\partial p}{\partial x} \\\frac{\partial p}{\partial y} \\\cdots \\\frac{\partial p}{\partial i} \\\end{matrix}\right]$
        Note that this class is an operator wrapper. The actual implementation of the operator is in the `_GradCore` class.
    """

    def __init__(self) -> None:
        super().__init__(_GradGenerator())
