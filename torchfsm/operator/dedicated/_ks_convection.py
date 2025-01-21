from .._base import CoreGenerator, NonlinearFunc, NonlinearOperator
from ...mesh import FourierMesh
from ..._type import FourierTensor, SpatialTensor
import torch
from typing import Optional


class _KSConvectionCore(NonlinearFunc):

    def __init__(self, remove_mean: bool) -> None:
        super().__init__()
        self.remove_mean = remove_mean

    def __call__(
        self,
        u_fft: FourierTensor["B C H ..."],
        f_mesh: FourierMesh,
        n_channel: int,
        u: Optional[SpatialTensor["B C H ..."]],
    ) -> FourierTensor["B C H ..."]:
        return f_mesh.fft(self.spatial_value(u_fft, f_mesh, n_channel, u))

    def spatial_value(
        self,
        u_fft: FourierTensor["B C H ..."],
        f_mesh: FourierMesh,
        n_channel: int,
        u: Optional[SpatialTensor["B C H ..."]],
    ) -> SpatialTensor["B C H ..."]:
        grad_u = f_mesh.ifft(f_mesh.nabla_vector(1) * u_fft).real
        if self.remove_mean:
            re = 0.5 * torch.sum(grad_u**2, dim=1, keepdim=True)
            return re - re.mean()
        else:
            return 0.5 * torch.sum(grad_u**2, dim=1, keepdim=True)


class _KSConvectionGenerator(CoreGenerator):
    def __init__(self, remove_mean: bool) -> None:
        self.remove_mean = remove_mean

    def __call__(self, f_mesh: FourierMesh, n_channel: int) -> NonlinearFunc:
        if n_channel != 1:
            return NotImplementedError("KSConvection only supports scalar field")
        return _KSConvectionCore(self.remove_mean)


class KSConvection(NonlinearOperator):

    def __init__(self, remove_mean: bool = True) -> None:
        super().__init__(_KSConvectionGenerator(remove_mean))
