from ..mesh import FourierMesh, MeshGrid, mesh_shape
from .._type import SpatialTensor
from ._normalize import normalize
import torch,random
from typing import Union, Sequence, Optional,Literal

def _get_mesh_device_and_dtype(
    mesh: Union[Sequence[tuple[float, float, int]], MeshGrid, FourierMesh],
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None
) -> tuple[torch.device, torch.dtype]:
    if device is None and (isinstance(mesh, FourierMesh) or isinstance(mesh, MeshGrid)):
        device = mesh.device
    if dtype is None and (isinstance(mesh, FourierMesh) or isinstance(mesh, MeshGrid)):
        dtype = mesh.dtype
    if not isinstance(mesh, FourierMesh):
        mesh = FourierMesh(mesh, device=device, dtype=dtype)
    return mesh, device, dtype

def truncated_fourier_series_custom_filter(
    mesh: Union[Sequence[tuple[float, float, int]], MeshGrid, FourierMesh],
    low_pass_filter: SpatialTensor["B C H ..."],
    amplitude_range: tuple[int, int] = (-1.0, 1.0),
    angle_range: tuple[int, int] = (0.0, 2.0 * torch.pi),
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    batch_size: int = 1,
    n_channel: int = 1,
    normalize_mode:Optional[Literal["normal_distribution","-1_1","0_1"]]=None,

) -> SpatialTensor["B C H ..."]:
    r"""
    Generate a truncated Fourier series noise field on a given mesh.
    
    Args:
        mesh (Union[Sequence[tuple[float, float, int]], MeshGrid, FourierMesh]): The mesh on which to generate the noise.
        low_pass_filter (SpatialTensor["B C H ..."]): The custom low-pass filter to apply to the Fourier noise.
        amplitude_range (tuple[int, int]): The range of amplitudes for the noise.
        angle_range (tuple[int, int]): The range of angles for the noise.
        device (Optional[torch.device]): The device on which to create the tensor.
        dtype (Optional[torch.dtype]): The data type of the tensor.
        batch_size (int): The number of batches.
        n_channel (int): The number of channels.
        normalize_mode (Optional[Literal["normal_distribution","-1_1","0_1"]]): The normalization mode for the generated noise.
            If None, no normalization is applied. Default is None.
    
    Returns:
        SpatialTensor["B C H ..."]: The generated noise field.
    """
    mesh, device, dtype = _get_mesh_device_and_dtype(mesh, device, dtype)
    
    magnitude=torch.rand(
        *mesh_shape(mesh, batch_size=batch_size, n_channel=n_channel),
        device=device,
        dtype=dtype
    )* (amplitude_range[1] - amplitude_range[0]) + amplitude_range[0]   
    angle=torch.rand(
        *mesh_shape(mesh, batch_size=batch_size, n_channel=n_channel),
        device=device,
        dtype=dtype
    )* (angle_range[1] - angle_range[0]) + angle_range[0]
    fourier_noise = magnitude * torch.exp(1j * angle)
    fourier_noise = fourier_noise * low_pass_filter
    fourier_noise = mesh.ifft(fourier_noise).real
    if normalize_mode is not None:
        fourier_noise = normalize(fourier_noise, normalize_mode=normalize_mode)
    return fourier_noise


def truncated_fourier_series(
    mesh: Union[Sequence[tuple[float, float, int]], MeshGrid, FourierMesh],
    freq_threshold: int = 5,
    amplitude_range: tuple[int, int] = (-1.0, 1.0),
    angle_range: tuple[int, int] = (0.0, 2.0 * torch.pi),
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    batch_size: int = 1,
    n_channel: int = 1,
    normalize_mode:Optional[Literal["normal_distribution","-1_1","0_1"]]=None,
    normalized_freq: bool = True
) -> SpatialTensor["B C H ..."]:
    r"""
    Generate a truncated Fourier series noise field on a given mesh.
    
    Args:
        mesh (Union[Sequence[tuple[float, float, int]], MeshGrid, FourierMesh]): The mesh on which to generate the noise.
        freq_threshold (int): The frequency threshold for truncation.
        amplitude_range (tuple[int, int]): The range of amplitudes for the noise.
        angle_range (tuple[int, int]): The range of angles for the noise.
        device (Optional[torch.device]): The device on which to create the tensor.
        dtype (Optional[torch.dtype]): The data type of the tensor.
        batch_size (int): The number of batches.
        n_channel (int): The number of channels.
        normalize_mode (Optional[Literal["normal_distribution","-1_1","0_1"]]): The normalization mode for the generated noise.
            If None, no normalization is applied. Default is None.
        normalized_freq (bool): If True, wheather to set the frequency threshold as a normalized value.
            If the domain length is 1, setting this to True or False will not make a difference. 
    
    Returns:
        SpatialTensor["B C H ..."]: The generated noise field.
    """
    mesh, device, dtype = _get_mesh_device_and_dtype(mesh, device, dtype)

    if normalized_freq:
        filter = mesh.normalized_low_pass_filter(freq_threshold)
        #mesh.normalized_low_pass_filter.cache_clear()
    else:
        filter =  mesh.abs_low_pass_filter(freq_threshold)
        #mesh.abs_low_pass_filter.cache_clear()
    
    return truncated_fourier_series_custom_filter(
        mesh=mesh,
        low_pass_filter=filter,
        amplitude_range=amplitude_range,
        angle_range=angle_range,
        device=device,
        dtype=dtype,
        batch_size=batch_size,  
        n_channel=n_channel,
        normalize_mode=normalize_mode,
    )

def random_truncated_fourier_series(
    mesh: Union[Sequence[tuple[float, float, int]], MeshGrid, FourierMesh],
    min_freq: int = 2,
    max_freq: int = 5,
    amplitude_range: tuple[int, int] = (-1.0, 1.0),
    angle_range: tuple[int, int] = (0.0, 2.0 * torch.pi),
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    batch_size: int = 1,
    n_channel: int = 1,
    normalize_mode:Optional[Literal["normal_distribution","-1_1","0_1"]]=None,
    normalized_freq: bool = True
) -> SpatialTensor["B C H ..."]:
    r"""
    Generate a truncated Fourier series noise field on a given mesh.
    
    Args:
        mesh (Union[Sequence[tuple[float, float, int]], MeshGrid, FourierMesh]): The mesh on which to generate the noise.
        freq_threshold (int): The frequency threshold for truncation.
        amplitude_range (tuple[int, int]): The range of amplitudes for the noise.
        angle_range (tuple[int, int]): The range of angles for the noise.
        device (Optional[torch.device]): The device on which to create the tensor.
        dtype (Optional[torch.dtype]): The data type of the tensor.
        batch_size (int): The number of batches.
        n_channel (int): The number of channels.
        normalize_mode (Optional[Literal["normal_distribution","-1_1","0_1"]]): The normalization mode for the generated noise.
            If None, no normalization is applied. Default is None.
        normalized_freq (bool): If True, wheather to set the frequency threshold as a normalized value.
            If the domain length is 1, setting this to True or False will not make a difference. 
    
    Returns:
        SpatialTensor["B C H ..."]: The generated noise field.
    """
    mesh, device, dtype = _get_mesh_device_and_dtype(mesh, device, dtype)

    if normalized_freq:
        filter = torch.cat([
            mesh.normalized_low_pass_filter(random.randint(min_freq, max_freq + 1))
            for _ in range(batch_size)
        ], dim=0)
        #mesh.normalized_low_pass_filter.cache_clear()
    else:
        filter = torch.cat([
            mesh.abs_low_pass_filter(random.randint(min_freq, max_freq + 1))
            for _ in range(batch_size)
        ], dim=0)
        #mesh.abs_low_pass_filter.cache_clear()
    
    return truncated_fourier_series_custom_filter(
        mesh=mesh,
        low_pass_filter=filter,
        amplitude_range=amplitude_range,
        angle_range=angle_range,
        device=device,
        dtype=dtype,
        batch_size=batch_size,  
        n_channel=n_channel,
        normalize_mode=normalize_mode,
    )