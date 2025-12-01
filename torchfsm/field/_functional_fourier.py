from ..mesh import FourierMesh, MeshGrid
from .._type import SpatialTensor
from ._normalize import normalize
from ._truncated_fourier import _get_mesh_device_and_dtype
from typing import Union, Sequence, Optional, Literal, Callable, Tuple
import torch


def functional_fourier_series(
    mesh: Union[Sequence[tuple[float, float, int]], MeshGrid, FourierMesh],
    mag_func: Callable[[FourierMesh], SpatialTensor["B C H ..."]],
    phi_func: Callable[[FourierMesh], SpatialTensor["B C H ..."]] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    normalize_mode: Optional[
        Union[
            Literal["normal_distribution", "-1_1", "0_1"],
            Tuple[Union[float, Tuple[float, float]], Union[float, Tuple[float, float]]],
        ]
    ] = None,
) -> SpatialTensor["B C H ..."]:
    r"""
    Generate a truncated Fourier series noise field with custom magnitude and phase functions on a given mesh.

    Args:
        mesh (Union[Sequence[tuple[float, float, int]], MeshGrid, FourierMesh]): The mesh on which to generate the noise.
        mag_func (Callable[[FourierMesh],SpatialTensor["B C H ..."]]): A function that takes a FourierMesh and returns the magnitude tensor.
            The magnitude function defines the amplitude of each frequency component in the Fourier domain.
            It is your responsibility to generate a magnitude function that has suitable batch and channel dimensions for your application.
        phi_func (Callable[[FourierMesh],SpatialTensor["B C H ..."]], optional): A function that takes a FourierMesh and returns the phase tensor.
            The phase function defines the phase shift of each frequency component in the Fourier domain.
            If None, a random phase between 0 and 2*pi with same shape as the magnitude will be generated.
        device (Optional[torch.device]): The device on which to create the tensor.
        dtype (Optional[torch.dtype]): The data type of the tensor.
        normalize_mode (Optional[Union[Literal["normal_distribution", "-1_1", "0_1"],Tuple[Union[float, Tuple[float, float]], Union[float, Tuple[float, float]]],]]): 
            The normalization mode for the generated noise. See `torchfsm.field.normalize` for details.
            If None, no normalization is applied. Default is None.

    Returns:
        SpatialTensor["B C H ..."]: The generated noise field.
    """
    mesh, device, dtype = _get_mesh_device_and_dtype(mesh, device, dtype)
    mag = mag_func(mesh)
    if phi_func is None:
        phi = mesh.fft(torch.randn(mag.shape)).angle()
    else:
        phi = phi_func(mesh)
    if phi.shape != mag.shape:
        raise ValueError(f"phi shape {phi.shape} does not match mag shape {mag.shape}")
    fourier_noise = mesh.ifft(mag * torch.exp(1j * phi)).real
    if normalize_mode is not None:
        fourier_noise = normalize(fourier_noise, normalize_mode=normalize_mode)
    return fourier_noise
