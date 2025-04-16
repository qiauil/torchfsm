from .._base import (
    CoreGenerator,
    NonlinearFunc,
    NonlinearOperator,
    LinearCoef,
    LinearOperator,
    OperatorLike,
)
from ...mesh import FourierMesh
import torch
from typing import Optional
from ..._type import FourierTensor, SpatialTensor
from ..generic._convection import _ConvectionCore
from ..._type import FourierTensor, SpatialTensor


# Vorticity Convection
class _VorticityConvectionCore(NonlinearFunc):

    def __init__(self) -> None:
        super().__init__()

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
        psi = -u_fft * f_mesh.invert_laplacian()
        ux = f_mesh.ifft(f_mesh.grad(1, 1) * psi).real
        uy = f_mesh.ifft(-f_mesh.grad(0, 1) * psi).real
        grad_x_w = f_mesh.ifft(f_mesh.grad(0, 1) * u_fft).real
        grad_y_w = f_mesh.ifft(f_mesh.grad(1, 1) * u_fft).real
        return ux * grad_x_w + uy * grad_y_w


class _VorticityConvectionGenerator(CoreGenerator):

    def __call__(self, f_mesh: FourierMesh, n_channel: int) -> NonlinearFunc:
        if f_mesh.n_dim != 2 or n_channel != 1:
            raise ValueError("Only vorticity in 2Dmesh is supported")
        return _VorticityConvectionCore()


class VorticityConvection(NonlinearOperator):

    def __init__(self) -> None:
        super().__init__(_VorticityConvectionGenerator())


# Vorticity To Velocity
class _Vorticity2VelocityCore(LinearCoef):

    def __call__(self, f_mesh, n_channel) -> FourierTensor["B C H ..."]:
        return (
            -1
            * f_mesh.invert_laplacian()
            * torch.cat(
                [
                    f_mesh.grad(1, 1).repeat([1, 1, f_mesh.mesh_info[0][-1], 1]),
                    -f_mesh.grad(0, 1).repeat([1, 1, 1, f_mesh.mesh_info[1][-1]]),
                ],
                dim=1,
            )
        )


class _Vorticity2VelocityGenerator(CoreGenerator):

    def __call__(self, f_mesh: FourierMesh, n_channel: int) -> NonlinearFunc:
        if f_mesh.n_dim != 2 or n_channel != 1:
            raise ValueError("Only vorticity in 2Dmesh is supported")
        return _Vorticity2VelocityCore()


class Vorticity2Velocity(LinearOperator):

    def __init__(self):
        super().__init__(_Vorticity2VelocityGenerator())


# Vorticity To Pressure
class _Vorticity2PressureCore(NonlinearFunc):
    def __init__(self, external_force: Optional[OperatorLike] = None) -> None:
        # if exeternal_force is not None, the external force may need the original u_fft, thus we do not dealiasing at the beginning
        super().__init__(dealiasing_swtich=external_force is None)
        self.external_force = external_force
        self._vorticity2velocity = _Vorticity2VelocityCore()
        self._convection = _ConvectionCore()

    def __call__(self, u_fft, f_mesh, n_channel, u) -> FourierTensor["B C H ..."]:
        velocity_fft = u_fft * self._vorticity2velocity(f_mesh, n_channel)
        if self.external_force is not None:
            velocity_fft *= f_mesh.low_pass_filter()
        convection = self._convection(
            velocity_fft, f_mesh, n_channel, f_mesh.ifft(velocity_fft).real
        )
        if self.external_force is not None:
            convection -= self.external_force(
                u_fft=u_fft, mesh=f_mesh, return_in_fourier=True
            )
        p = torch.sum(
            f_mesh.nabla_vector(1) * convection, dim=1, keepdim=True
        )  # nabla.(u.nabla_u)
        return -1 * p * f_mesh.invert_laplacian()  #  p = - nabla.(u.nabla_u)/laplacian


class _Vorticity2PressureGenerator(CoreGenerator):

    def __init__(self, external_force: Optional[OperatorLike] = None) -> None:
        self.external_force = external_force

    def __call__(self, f_mesh: FourierMesh, n_channel: int) -> NonlinearFunc:
        if f_mesh.n_dim != 2 or n_channel != 1:
            raise ValueError("Only vorticity in 2Dmesh is supported")
        return _Vorticity2PressureCore(self.external_force)


class Vorticity2Pressure(NonlinearOperator):

    def __init__(self, external_force: Optional[OperatorLike] = None) -> None:
        super().__init__(_Vorticity2PressureGenerator(external_force))


# Velocity To Pressure
class _Velocity2PressureCore(NonlinearFunc):

    def __init__(self, external_force: Optional[OperatorLike] = None) -> None:
        # if exeternal_force is not None, the external force may need the original u_fft, thus we do not dealiasing at the beginning
        super().__init__(dealiasing_swtich=external_force is None)
        self.external_force = external_force
        self._convection = _ConvectionCore()

    def __call__(
        self,
        u_fft: FourierTensor["B C H ..."],
        f_mesh: FourierMesh,
        n_channel: int,
        u: SpatialTensor["B C H ..."] | None,
    ) -> FourierTensor["B C H ..."]:
        if self.external_force is not None:  # u_fft is original version
            force = self.external_force(
                u_fft=u_fft, mesh=f_mesh, return_in_fourier=True
            )
            u_fft *= f_mesh.low_pass_filter()
            u = f_mesh.ifft(u_fft).real
        else:  # u_fft is dealiased version
            if u is None:
                u = f_mesh.ifft(u_fft).real
        convection = self._convection(u_fft, f_mesh, n_channel, u)
        if self.external_force is not None:
            convection -= force
        p = torch.sum(
            f_mesh.nabla_vector(1) * convection, dim=1, keepdim=True
        )  # nabla.(u.nabla_u)
        return -1 * p * f_mesh.invert_laplacian()  #  p = - nabla.(u.nabla_u)/laplacian


class Velocity2Pressure(NonlinearOperator):

    def __init__(self, external_force: Optional[OperatorLike] = None) -> None:
        super().__init__(_Velocity2PressureCore(external_force))


# Velocity Convection
class _NSPressureConvectionCore(NonlinearFunc):

    def __init__(self, external_force: Optional[OperatorLike] = None) -> None:
        super().__init__(dealiasing_swtich=external_force is None)
        self.external_force = external_force
        self._convection = _ConvectionCore()

    def __call__(
        self,
        u_fft: FourierTensor["B C H ..."],
        f_mesh: FourierMesh,
        n_channel: int,
        u: SpatialTensor["B C H ..."] | None,
    ) -> torch.Tensor:
        if self.external_force is not None:  # u_fft is original version
            force = self.external_force(
                u_fft=u_fft, mesh=f_mesh, return_in_fourier=True
            )
            u_fft *= f_mesh.low_pass_filter()
            u = f_mesh.ifft(u_fft).real
        else:  # u_fft is dealiased version
            if u is None:
                u = f_mesh.ifft(u_fft).real
        convection = self._convection(u_fft, f_mesh, n_channel, u)
        if self.external_force is not None:
            convection -= force
        p = f_mesh.invert_laplacian() * torch.sum(
            f_mesh.nabla_vector(1) * convection, dim=1, keepdim=True
        )  # - p = nabla.(u.nabla_u)/laplacian
        if self.external_force is not None:
            return f_mesh.nabla_vector(1) * p - convection + force # -nabla(p) - nabla.(u.nabla_u) + f
        return f_mesh.nabla_vector(1) * p - convection  # -nabla(p) - nabla.(u.nabla_u)


class NSPressureConvection(NonlinearOperator):

    def __init__(self, external_force: Optional[OperatorLike] = None) -> None:
        super().__init__(_NSPressureConvectionCore(external_force))
