from typing import Sequence, Union
from ...mesh import FourierMesh
from .._base import LinearCoef, LinearOperator
from ..._type import FourierTensor


class _LinearAdvectionCore(LinearCoef):
    r"""
    Implementation of the LinearAdvection operator.
    """

    def __init__(self, velocity: Union[float, Sequence[float]] = 1.0) -> None:
        super().__init__()
        self.velocity = velocity

    def __call__(
        self, f_mesh: FourierMesh, n_channel: int
    ) -> FourierTensor["B C H ..."]:
        if isinstance(self.velocity, float):
            velocity = [self.velocity] * f_mesh.n_dim
        else:
            assert len(self.velocity) == f_mesh.n_dim, (
                "Length of velocity list must match the number of dimensions of the mesh."
                + " If you have same operator for a different mesh before, please reset the velocity using `set_advection_velocity` method."
            )
            velocity = self.velocity
        if n_channel != 1:
            raise ValueError("The Advection operator only supports scalar field.")
        return sum(
            [
                v * f_mesh.nabla_vector(1)[:, i : i + 1, ...]
                for i, v in enumerate(velocity)
            ]
        )

    def set_advection_velocity(self, velocity: Union[float, Sequence[float]]) -> None:
        """Set the advection velocity.

        Args:
            velocity (Union[float, Sequence[float]]): The constant velocity field.
                If a float is provided, it is assumed that the velocity is the same in all dimensions.
                If a sequence is provided, its length must match the number of dimensions of the mesh.
        """
        self.velocity = velocity


class LinearAdvection(LinearOperator):
    r"""
    `LinearAdvection` calculates the advection of a scalar field by a constant velocity field.
        It is defined as $ \mathbf{u} \cdot \nabla \phi = \sum_{i=0}^I u_i \frac{\partial \phi}{\partial i} $
        where $\mathbf{u} = [u_x, u_y, \cdots, u_I]$ is the constant velocity field.
        If your velocity is not constant in space, please consider using `Advection` operator instead.
        Note that this class is an operator wrapper. The actual implementation of the operator is in the `_AdvectionCore` class.

    Args:
        velocity (Union[float, Sequence[float]]): The constant velocity field.
            If a float is provided, it is assumed that the velocity is the same in all dimensions.
            If a sequence is provided, its length must match the number of dimensions of the mesh.
            Default is 1.0.
    """

    def __init__(self, velocity: Union[float, Sequence[float]] = 1.0) -> None:
        super().__init__(_LinearAdvectionCore(velocity))

    def set_advection_velocity(self, velocity: Union[float, Sequence[float]]) -> None:
        """Set the advection velocity.

        Args:
            velocity (Union[float, Sequence[float]]): The constant velocity field.
                If a float is provided, it is assumed that the velocity is the same in all dimensions.
                If a sequence is provided, its length must match the number of dimensions of the mesh.
        """
        self._core.set_advection_velocity(velocity)
