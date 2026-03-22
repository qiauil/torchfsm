from ..._base import (
    NonlinearFunc,
    NonlinearOperator,
    OperatorLike,
)
from ....mesh import FourierMesh
import torch
from typing import Optional
from ...._type import FourierTensor, SpatialTensor
from ...generic._convection import _ConvectionCore
from ...._type import FourierTensor, SpatialTensor
from ..._base import ExplicitSource

# Velocity Convection
class _NSPressureConvectionCore(NonlinearFunc):
    r"""
    Implementation of the Navier-Stokes pressure convection operator.
    """

    def __init__(self, external_force: Optional[OperatorLike] = None) -> None:
        super().__init__(dealiasing_swtich=external_force is None)
        self.external_force = external_force
        self._convection = _ConvectionCore()

    def _get_force(self,u_fft, mesh):
        if isinstance(self.external_force, ExplicitSource):
            # If the external force is an explicit source, we directly get the force value.
            # This provide better performance since directly calling the operator requires fft for the velocity field.
            return self.external_force.operator_cores[0](u_fft=None,f_mesh=mesh)
        else:
            return self.external_force(
                u_fft=u_fft, mesh=mesh, return_in_fourier=True
            )

    def __call__(
        self,
        u_fft: FourierTensor["B C H ..."],
        f_mesh: FourierMesh,
        u: Optional[SpatialTensor["B C H ..."]] = None,
    ) -> torch.Tensor:
        if self.external_force is not None:  # u_fft is original version
            force = self._get_force(
                u_fft=u_fft, mesh=f_mesh
            )
            u_fft *= f_mesh.low_pass_filter()
            u = f_mesh.ifft(u_fft).real
        else:  # u_fft is dealiased version
            if u is None:
                u = f_mesh.ifft(u_fft).real
        convection = self._convection(u_fft, f_mesh,u)
        if self.external_force is not None:
            convection -= force
        p = f_mesh.invert_laplacian() * torch.sum(
            f_mesh.nabla_vector(1) * convection, dim=1, keepdim=True
        )  # - p = nabla.(u.nabla_u)/laplacian
        if self.external_force is not None:
            return f_mesh.nabla_vector(1) * p - convection + force # -nabla(p) - nabla.(u.nabla_u) + f
        return f_mesh.nabla_vector(1) * p - convection  # -nabla(p) - nabla.(u.nabla_u)


class NSPressureConvection(NonlinearOperator):
    r"""
    Operator for Navier-Stokes pressure convection.
        It is defined as $-\nabla (\nabla^{-2} \nabla \cdot (\left(\mathbf{u}\cdot\nabla\right)\mathbf{u}-f))-\left(\mathbf{u}\cdot\nabla\right)\mathbf{u} + \mathbf{f}$.
        Note that this class is an operator wrapper. The real implementation of the source term is in the `_NSPressureConvectionCore` class.
    
    Args:
        external_force: Optional[OperatorLike], optional, default=None
    """

    def __init__(self, external_force: Optional[OperatorLike] = None) -> None:
        super().__init__(_NSPressureConvectionCore(external_force))
