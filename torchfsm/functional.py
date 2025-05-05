from .operator import *
from .mesh import FourierMesh, MeshGrid
from typing import Optional, Union, Sequence
import torch
from ._type import SpatialTensor, FourierTensor
from .operator._base import OperatorLike


def biharmonic(
    u: Optional[SpatialTensor["B C H ..."]] = None,
    u_fft: Optional[FourierTensor["B C H ..."]] = None,
    mesh: Optional[
        Union[Sequence[tuple[float, float, int]], MeshGrid, FourierMesh]
    ] = None,
) -> SpatialTensor["B C H ..."]:
    r"""
    Function form for the biharmonic operator.
        For details, see `torchfsm.operator.Biharmonic`.

    Args:
        u (Optional[SpatialTensor["B C H ..."]]): The input field.
        u_fft (Optional[FourierTensor["B C H ..."]]): The Fourier-transformed input field.
        mesh (Optional[Union[Sequence[tuple[float, float, int]], MeshGrid, FourierMesh]]): The mesh for the field.

    Returns:
        SpatialTensor["B C H ..."]: The result of applying the biharmonic operator to the input field.   
    """
    return Biharmonic()(u=u, u_fft=u_fft, mesh=mesh)


def conservative_convection(
    u: Optional[SpatialTensor["B C H ..."]] = None,
    u_fft: Optional[FourierTensor["B C H ..."]] = None,
    mesh: Optional[
        Union[Sequence[tuple[float, float, int]], MeshGrid, FourierMesh]
    ] = None,
) -> SpatialTensor["B C H ..."]:
    r"""
    Function form for the conservative convection operator.
        For details, see `torchfsm.operator.ConservativeConvection`.

    Args:
        u (Optional[SpatialTensor["B C H ..."]]): The input field.
        u_fft (Optional[FourierTensor["B C H ..."]]): The Fourier-transformed input field.
        mesh (Optional[Union[Sequence[tuple[float, float, int]], MeshGrid, FourierMesh]]): The mesh for the field.

    Returns:
        SpatialTensor["B C H ..."]: The result of applying the conservative convection operator to the input field.
    """
    return ConservativeConvection()(u=u, u_fft=u_fft, mesh=mesh)


def convection(
    u: Optional[SpatialTensor["B C H ..."]] = None,
    u_fft: Optional[FourierTensor["B C H ..."]] = None,
    mesh: Optional[
        Union[Sequence[tuple[float, float, int]], MeshGrid, FourierMesh]
    ] = None,
) -> SpatialTensor["B C H ..."]:
    r"""
    Function form for the convection operator.
        For details, see `torchfsm.operator.Convection`.
    
    Args:
        u (Optional[SpatialTensor["B C H ..."]]): The input field.
        u_fft (Optional[FourierTensor["B C H ..."]]): The Fourier-transformed input field.
        mesh (Optional[Union[Sequence[tuple[float, float, int]], MeshGrid, FourierMesh]]): The mesh for the field.
    
    Returns:
        SpatialTensor["B C H ..."]: The result of applying the convection operator to the input field.
    """

    return Convection()(u=u, u_fft=u_fft, mesh=mesh)


def curl(
    u: Optional[SpatialTensor["B C H ..."]] = None,
    u_fft: Optional[FourierTensor["B C H ..."]] = None,
    mesh: Optional[
        Union[Sequence[tuple[float, float, int]], MeshGrid, FourierMesh]
    ] = None,
) -> SpatialTensor["B C H ..."]:
    r"""
    Function form for the curl operator.
        For details, see `torchfsm.operator.Curl`.

    Args:
        u (Optional[SpatialTensor["B C H ..."]]): The input field.
        u_fft (Optional[FourierTensor["B C H ..."]]): The Fourier-transformed input field.
        mesh (Optional[Union[Sequence[tuple[float, float, int]], MeshGrid, FourierMesh]]): The mesh for the field.

    Returns:
        SpatialTensor["B C H ..."]: The result of applying the curl operator to the input field.
    """

    return Curl()(u=u, u_fft=u_fft, mesh=mesh)


def div(
    u: Optional[SpatialTensor["B C H ..."]] = None,
    u_fft: Optional[FourierTensor["B C H ..."]] = None,
    mesh: Optional[
        Union[Sequence[tuple[float, float, int]], MeshGrid, FourierMesh]
    ] = None,
) -> SpatialTensor["B C H ..."]:
    r"""
    Function form for the divergence operator.
        For details, see `torchfsm.operator.Div`.
    
    Args:
        u (Optional[SpatialTensor["B C H ..."]]): The input field.
        u_fft (Optional[FourierTensor["B C H ..."]]): The Fourier-transformed input field.
        mesh (Optional[Union[Sequence[tuple[float, float, int]], MeshGrid, FourierMesh]]): The mesh for the field.
    
    Returns:
        SpatialTensor["B C H ..."]: The result of applying the divergence operator to the input field.
    """

    return Div()(u=u, u_fft=u_fft, mesh=mesh)


def grad(
    u: Optional[SpatialTensor["B C H ..."]] = None,
    u_fft: Optional[FourierTensor["B C H ..."]] = None,
    mesh: Optional[
        Union[Sequence[tuple[float, float, int]], MeshGrid, FourierMesh]
    ] = None,
) -> SpatialTensor["B C H ..."]:
    r"""
    Function form for the gradient operator.
        For details, see `torchfsm.operator.Grad`.

    Args:
        u (Optional[SpatialTensor["B C H ..."]]): The input field.
        u_fft (Optional[FourierTensor["B C H ..."]]): The Fourier-transformed input field.
        mesh (Optional[Union[Sequence[tuple[float, float, int]], MeshGrid, FourierMesh]]): The mesh for the field.
    
    Returns:
        SpatialTensor["B C H ..."]: The result of applying the gradient operator to the input field.
    """

    return Grad()(u=u, u_fft=u_fft, mesh=mesh)


def laplacian(
    u: Optional[SpatialTensor["B C H ..."]] = None,
    u_fft: Optional[FourierTensor["B C H ..."]] = None,
    mesh: Optional[
        Union[Sequence[tuple[float, float, int]], MeshGrid, FourierMesh]
    ] = None,
) -> SpatialTensor["B C H ..."]:
    r"""
    Function form for the Laplacian operator.
        For details, see `torchfsm.operator.Laplacian`.

    Args:
        u (Optional[SpatialTensor["B C H ..."]]): The input field.
        u_fft (Optional[FourierTensor["B C H ..."]]): The Fourier-transformed input field.
        mesh (Optional[Union[Sequence[tuple[float, float, int]], MeshGrid, FourierMesh]]): The mesh for the field.
    
    Returns:
        SpatialTensor["B C H ..."]: The result of applying the Laplacian operator to the input field.
    """
    return Laplacian()(u=u, u_fft=u_fft, mesh=mesh)


def spatial_derivative(
    dim_index: int,
    order: int,
    u: Optional[SpatialTensor["B C H ..."]] = None,
    u_fft: Optional[FourierTensor["B C H ..."]] = None,
    mesh: Optional[
        Union[Sequence[tuple[float, float, int]], MeshGrid, FourierMesh]
    ] = None,
) -> SpatialTensor["B C H ..."]:
    r"""
    Function form for the spatial derivative operator.
        For details, see `torchfsm.operator.SpatialDerivative`.

    Args:
        dim_index (int): The index of the dimension to take the derivative along.
        order (int): The order of the derivative.
        u (Optional[SpatialTensor["B C H ..."]]): The input field.
        u_fft (Optional[FourierTensor["B C H ..."]]): The Fourier-transformed input field.
        mesh (Optional[Union[Sequence[tuple[float, float, int]], MeshGrid, FourierMesh]]): The mesh for the field.
    
    Returns:
        SpatialTensor["B C H ..."]: The result of applying the spatial derivative operator to the input field.
    """
    return SpatialDerivative(dim_index, order)(u=u, u_fft=u_fft, mesh=mesh)


def ks_convection(
    u: Optional[SpatialTensor["B C H ..."]] = None,
    u_fft: Optional[FourierTensor["B C H ..."]] = None,
    mesh: Optional[
        Union[Sequence[tuple[float, float, int]], MeshGrid, FourierMesh]
    ] = None,
    remove_mean: bool = True,
) -> SpatialTensor["B C H ..."]:
    r"""
    Function form for the Kuramoto-Sivashinsky convection operator.
        For details, see `torchfsm.operator.KSConvection`.

    Args:
        u (Optional[SpatialTensor["B C H ..."]]): The input field.
        u_fft (Optional[FourierTensor["B C H ..."]]): The Fourier-transformed input field.
        mesh (Optional[Union[Sequence[tuple[float, float, int]], MeshGrid, FourierMesh]]): The mesh for the field.
        remove_mean (bool): If True, the mean of the input field will be removed before applying the operator.

    Returns:
        SpatialTensor["B C H ..."]: The result of applying the Kuramoto-Sivashinsky convection operator to the input field.
    """
    return KSConvection(remove_mean)(u=u, u_fft=u_fft, mesh=mesh)

def vorticity_convection(
    u: Optional[SpatialTensor["B C H ..."]] = None,
    u_fft: Optional[FourierTensor["B C H ..."]] = None,
    mesh: Optional[
        Union[Sequence[tuple[float, float, int]], MeshGrid, FourierMesh]
    ] = None,
) -> SpatialTensor["B C H ..."]:

    r"""
    Function form for the vorticity convection operator.
        For details, see `torchfsm.operator.VorticityConvection`.

    Args:
        u (Optional[SpatialTensor["B C H ..."]]): The input field.
        u_fft (Optional[FourierTensor["B C H ..."]]): The Fourier-transformed input field.
        mesh (Optional[Union[Sequence[tuple[float, float, int]], MeshGrid, FourierMesh]]): The mesh for the field.
    
    Returns:
        SpatialTensor["B C H ..."]: The result of applying the vorticity convection operator to the input field.
    """
    return VorticityConvection()(u=u, u_fft=u_fft, mesh=mesh)

def _get_fft_with_mesh(
    value: Optional[torch.Tensor] = None,
    value_fft: Optional[torch.Tensor] = None,
    mesh: Union[Sequence[tuple[float, float, int]], MeshGrid, FourierMesh] = None,
) -> tuple[torch.Tensor, FourierMesh]:
    r"""
    Helper function to get the Fourier transform of a value and the corresponding mesh.

    Args:
        value (Optional[torch.Tensor]): The input value.
        value_fft (Optional[torch.Tensor]): The Fourier-transformed value.
        mesh (Union[Sequence[tuple[float, float, int]], MeshGrid, FourierMesh]): The mesh for the field.

    Returns:
        tuple: A tuple containing the Fourier-transformed value and the mesh.
    """
    if value_fft is None:
        f_mesh = (
            mesh
            if isinstance(mesh, FourierMesh)
            else FourierMesh(mesh, device=value.device, dtype=value.dtype)
        )
        value_fft = f_mesh.fft(value)
    else:
        f_mesh = (
            mesh
            if isinstance(mesh, FourierMesh)
            else FourierMesh(mesh, device=value_fft.device, dtype=value_fft.dtype)
        )
    return value_fft, f_mesh


def vorticity2velocity(
    vorticity: Optional[torch.Tensor] = None,
    vorticity_fft: Optional[torch.Tensor] = None,
    mesh: Optional[
        Union[Sequence[tuple[float, float, int]], MeshGrid, FourierMesh]
    ] = None,
) -> SpatialTensor["B C H ..."]:
    r"""
    Function form for the vorticity to velocity operator.
        For details, see `torchfsm.operator.Vorticity2Velocity`.

    Args:
        vorticity (Optional[torch.Tensor]): The input vorticity field.
        vorticity_fft (Optional[torch.Tensor]): The Fourier-transformed vorticity field.
        mesh (Optional[Union[Sequence[tuple[float, float, int]], MeshGrid, FourierMesh]]): The mesh for the field.
    
    Returns:
        SpatialTensor["B C H ..."]: The result of applying the vorticity to velocity operator to the input field.
    """

    return Vorticity2Velocity()(u=vorticity, u_fft=vorticity_fft, mesh=mesh)


def velocity2pressure(
    velocity: Optional[torch.Tensor] = None,
    velocity_fft: Optional[torch.Tensor] = None,
    mesh: Optional[
        Union[Sequence[tuple[float, float, int]], MeshGrid, FourierMesh]
    ] = None,
    external_force: Optional[OperatorLike] = None,
) -> SpatialTensor["B C H ..."]:
    r"""
    Function form for the velocity to pressure operator.
        For details, see `torchfsm.operator.Velocity2Pressure`.

    Args:
        velocity (Optional[torch.Tensor]): The input velocity field.
        velocity_fft (Optional[torch.Tensor]): The Fourier-transformed velocity field.
        mesh (Optional[Union[Sequence[tuple[float, float, int]], MeshGrid, FourierMesh]]): The mesh for the field.
        external_force (Optional[OperatorLike]): An optional external force to be applied.
    
    Returns:
        SpatialTensor["B C H ..."]: The result of applying the velocity to pressure operator to the input field.
    """

    velocity_fft, f_mesh = _get_fft_with_mesh(
        value=velocity, value_fft=velocity_fft, mesh=mesh
    )
    convection = Convection()(u_fft=velocity_fft, mesh=f_mesh, return_in_fourier=True)
    if external_force is not None:
        convection -= external_force(
            u_fft=velocity_fft, mesh=f_mesh, return_in_fourier=True
        )
    convection = Div()(u_fft=convection, mesh=f_mesh, return_in_fourier=True)
    return Laplacian().solve(b_fft=-1 * convection, mesh=f_mesh, n_channel=1)


def vorticity2pressure(
    vorticity: Optional[torch.Tensor] = None,
    vorticity_fft: Optional[torch.Tensor] = None,
    mesh: Optional[
        Union[Sequence[tuple[float, float, int]], MeshGrid, FourierMesh]
    ] = None,
    external_force: Optional[OperatorLike] = None,
) -> SpatialTensor["B C H ..."]:
    r"""
    Function form for the vorticity to pressure operator.
        For details, see `torchfsm.operator.Vorticity2Pressure`.

    Args:
        vorticity (Optional[torch.Tensor]): The input vorticity field.
        vorticity_fft (Optional[torch.Tensor]): The Fourier-transformed vorticity field.
        mesh (Optional[Union[Sequence[tuple[float, float, int]], MeshGrid, FourierMesh]]): The mesh for the field.
        external_force (Optional[OperatorLike]): An optional external force to be applied.

    Returns:
        SpatialTensor["B C H ..."]: The result of applying the vorticity to pressure operator to the input field.
    """

    vorticity_fft, f_mesh = _get_fft_with_mesh(
        value=vorticity, value_fft=vorticity_fft, mesh=mesh
    )
    velocity_fft = Vorticity2Velocity()(
        u_fft=vorticity_fft, mesh=f_mesh, return_in_fourier=True
    )
    convection = Convection()(u_fft=velocity_fft, mesh=f_mesh, return_in_fourier=True)
    if external_force is not None:
        convection -= external_force(
            u_fft=vorticity_fft, mesh=f_mesh, return_in_fourier=True
        )
    convection = Div()(u_fft=convection, mesh=f_mesh, return_in_fourier=True)
    return Laplacian().solve(b_fft=-1 * convection, mesh=f_mesh, n_channel=1)