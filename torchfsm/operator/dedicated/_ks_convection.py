from .._base import CoreGenerator, NonlinearFunc, NonlinearOperator
from ...mesh import FourierMesh
from ..._type import FourierTensor, SpatialTensor
import torch
from typing import Optional


class _KSConvectionCore(NonlinearFunc):

    r"""
    Implementation of the KSConvection operator.
    """

    def __init__(self, remove_mean: bool) -> None:
        super().__init__()
        self.remove_mean = remove_mean

    def __call__(
        self,
        u_fft: FourierTensor["B C H ..."],
        f_mesh: FourierMesh,
        n_channel: int,
        u: Optional[SpatialTensor["B C H ..."]]=None,
    ) -> FourierTensor["B C H ..."]:
        return f_mesh.fft(self.spatial_value(u_fft, f_mesh, u))

    def spatial_value(
        self,
        u_fft: FourierTensor["B C H ..."],
        f_mesh: FourierMesh,
        u: Optional[SpatialTensor["B C H ..."]]=None,
    ) -> SpatialTensor["B C H ..."]:
        grad_u = f_mesh.ifft(f_mesh.nabla_vector(1) * u_fft).real
        if self.remove_mean:
            re = 0.5 * torch.sum(grad_u**2, dim=1, keepdim=True)
            return re - re.mean()
        else:
            return 0.5 * torch.sum(grad_u**2, dim=1, keepdim=True)


class _KSConvectionGenerator(CoreGenerator):
    r"""
    Generator of the KSConvection operator. It ensures that the operator is only applied to scalar fields.
    """

    def __init__(self, remove_mean: bool) -> None:
        self.remove_mean = remove_mean

    def __call__(self, f_mesh: FourierMesh, n_channel: int) -> NonlinearFunc:
        if n_channel != 1:
            raise NotImplementedError("KSConvection only supports scalar field")
        return _KSConvectionCore(self.remove_mean)


class KSConvection(NonlinearOperator):

    r"""
    The Kuramoto-Sivashinsky convection operator for a scalar field.
        It is defined as: $\frac{1}{2}|\nabla \phi|^2=\frac{1}{2}\sum_{i=0}^{I}(\frac{\partial \phi}{\partial i})^2$
        Note that this class is an operator wrapper. The real implementation of the source term is in the `_KSConvectionCore` class.
   
    Args:
        remove_mean (bool): Whether to remove the mean of the result. Default is True. Set to True will improve the stability of the simulation.
    """

    def __init__(self, remove_mean: bool = True) -> None:
        super().__init__(_KSConvectionGenerator(remove_mean))
