import torch
from torch import Tensor
from ...mesh import FourierMesh
from .._base import LinearCoef, NonlinearOperator, CoreGenerator, NonlinearFunc
from ..._type import FourierTensor, SpatialTensor
from typing import Optional, Union


class _Curl2DCore(NonlinearFunc):

    r"""
    Implementation of the Curl operator for 2D vector fields.
    """

    def __init__(self):
        super().__init__(False)

    def __call__(
        self,
        u_fft: FourierTensor["B C H ..."],
        f_mesh: FourierMesh,
        u: Optional[SpatialTensor["B C H ..."]] = None,
    ) -> FourierTensor["B C H ..."]:
        return (
            f_mesh.grad(0, 1) * u_fft[:, 1:2, ...]
            - f_mesh.grad(1, 1) * u_fft[:, 0:1, ...]
        )


class _Curl3DCore(NonlinearFunc):

    r"""
    Implementation of the Curl operator for 3D vector fields.
    """

    def __init__(self):
        super().__init__(False)

    def __call__(
        self,
        u_fft: FourierTensor["B C H ..."],
        f_mesh: FourierMesh,
        u: Optional[SpatialTensor["B C H ..."]] = None,
    ) -> FourierTensor["B C H ..."]:
        return torch.cat(
            [
                f_mesh.grad(1, 1) * u_fft[:, 2:3, ...]
                - f_mesh.grad(2, 1) * u_fft[:, 1:2, ...],
                f_mesh.grad(2, 1) * u_fft[:, 0:1, ...]
                - f_mesh.grad(0, 1) * u_fft[:, 2:3, ...],
                f_mesh.grad(0, 1) * u_fft[:, 1:2, ...]
                - f_mesh.grad(1, 1) * u_fft[:, 0:1, ...],
            ],
            dim=1,
        )


class _CurlGenerator(CoreGenerator):
    r"""
    Generator of the Curl operator. 
        It ensure that curl only works for 2D or 3D vector field with the same dimension as the mesh.
    """

    def __call__(
        self, f_mesh: FourierMesh, n_channel: int
    ) -> Union[LinearCoef, NonlinearFunc]:
        if f_mesh.n_dim != n_channel:
            raise ValueError(
                f"div operator only works for vector field with the same dimension as mesh"
            )
        if n_channel > 3 or n_channel < 2:
            raise ValueError(f"div operator only works for 2D or 3D vector field")
        if n_channel == 2:
            return _Curl2DCore()
        else:
            return _Curl3DCore()


class Curl(NonlinearOperator):

    r"""
    Curl operator for 2D and 3D vector fields. 
        It is defined as: $\nabla \times \mathbf{u} = \frac{\partial u_y}{\partial x}-\frac{\partial u_x}{\partial y}$
        for 2D vector field $\mathbf{u} = (u_x, u_y)$ and
        $\nabla \times \mathbf{u} = \left[\begin{matrix} \frac{\partial u_z}{\partial y}-\frac{\partial u_y}{\partial z} \\ \frac{\partial u_x}{\partial z}-\frac{\partial u_z}{\partial x} \\ \frac{\partial u_y}{\partial x}-\frac{\partial u_x}{\partial y} \end{matrix} \right]$
        for 3D vector field $\mathbf{u} = (u_x, u_y, u_z)$.
        This operator only works for vector fields with the same dimension as the mesh.
        Note that this class is an operator wrapper. The real implementation of the source term is in the `_Curl2DCore` and `_Curl2DCore` class.

    """

    def __init__(self) -> None:
        super().__init__(_CurlGenerator())
