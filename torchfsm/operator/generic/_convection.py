from torch import Tensor
from ...mesh import FourierMesh
from .._base import LinearCoef, NonlinearOperator, CoreGenerator, NonlinearFunc
from functools import lru_cache
from ..._type import FourierTensor, SpatialTensor


class _ConvectionCore(NonlinearFunc):

    r"""
    Implementation of the Convection operator.
    """

    def __init__(self):
        super().__init__()

    def __call__(
        self,
        u_fft: FourierTensor["B C H ..."],
        f_mesh: FourierMesh,
        u: SpatialTensor["B C H ..."] | None,
    ) -> FourierTensor["B C H ..."]:
        return f_mesh.fft(self.spatial_value(u_fft, f_mesh, u))

    def spatial_value(
        self,
        u_fft: FourierTensor["B C H ..."],
        f_mesh: FourierMesh,
        u: FourierTensor["B C H ..."] | None,
    ) -> SpatialTensor["B C H ..."]:
        r"""
        Return the result of the nonlinear function in spatial domain.

        Args:
            u_fft (FourierTensor): Fourier-transformed input tensor.
            f_mesh (FourierMesh): Fourier mesh object.
            u (Optional[SpatialTensor]): Corresponding tensor of u_fft in spatial domain. This option aims to avoid repeating the inverse FFT operation in operators.
        
        Returns:
            SpatialTensor: Result of the nonlinear function in spatial domain.
        """
        if u is None:
            u = f_mesh.ifft(u_fft).real
        nabla_u = f_mesh.nabla_vector(1).unsqueeze(2) * u_fft.unsqueeze(1)
        return (u.unsqueeze(2) * f_mesh.ifft(nabla_u).real).sum(1)
        # another way to calculate convection:
        # return sum([u[:,i:i+1,...]*f_mesh.ifft(f_mesh.grad(i,1)*u_fft).real for i in range(n_channel)])


class _ConvectionGenerator(CoreGenerator):

    r"""
    Generator of the Convection operator. It ensures that the operator is only applied to vector fields with the same dimension as the mesh.
    """

    def __call__(
        self, f_mesh: FourierMesh, n_channel: int
    ) -> LinearCoef | NonlinearFunc:
        if f_mesh.n_dim != n_channel:
            raise ValueError(
                f"convection operator only works for vector field with the same dimension as mesh"
            )
        return _ConvectionCore()


class Convection(NonlinearOperator):
    r"""
    `Convection` calculates the convection of a vector field on itself if the vector field is divergence free, i.e., $\nabla \cdot \mathbf{u} =0$.
        It is defined as $\mathbf{u} \cdot \nabla  \mathbf{u}=\left[\begin{matrix}\sum_{i=0}^I u_i\frac{\partial u_x }{\partial i} \\\sum_{i=0}^I u_i\frac{\partial u_y }{\partial i} \\\cdots\\\sum_{i=0}^I u_i\frac{\partial u_I }{\partial i} \\\end{matrix}\right]$
        This operator only works for vector fields with the same dimension as the mesh.
        Note that this class is an operator wrapper. The real implementation of the source term is in the `_ConvectionCore` class.
    """

    def __init__(self) -> None:
        super().__init__(_ConvectionGenerator())
