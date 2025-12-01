from ..mesh import FourierMesh, MeshGrid, mesh_shape
from .._type import SpatialTensor
from ._normalize import normalize
import torch, random
from typing import Union, Sequence, Optional, Literal, Tuple


def _get_mesh_device_and_dtype(
    mesh: Union[Sequence[tuple[float, float, int]], MeshGrid, FourierMesh],
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
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
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    batch_size: int = 1,
    n_channel: int = 1,
    normalize_mode: Optional[
        Union[
            Literal["normal_distribution", "-1_1", "0_1"],
            Tuple[Union[float, Tuple[float, float]], Union[float, Tuple[float, float]]],
        ]
    ] = None,
    noise_type: Literal["normal", "uniform"] = "normal",
) -> SpatialTensor["B C H ..."]:
    r"""
    Generate a truncated Fourier series noise field on a given mesh.

    Args:
        mesh (Union[Sequence[tuple[float, float, int]], MeshGrid, FourierMesh]): The mesh on which to generate the noise.
        low_pass_filter (SpatialTensor["B C H ..."]): The custom low-pass filter to apply to the Fourier noise.
        device (Optional[torch.device]): The device on which to create the tensor.
        dtype (Optional[torch.dtype]): The data type of the tensor.
        batch_size (int): The number of batches.
        n_channel (int): The number of channels.
        normalize_mode (Optional[Union[Literal["normal_distribution", "-1_1", "0_1"],Tuple[Union[float, Tuple[float, float]], Union[float, Tuple[float, float]]],]]):
            The normalization mode for the generated noise. See `torchfsm.field.normalize` for details.
            If None, no normalization is applied. Default is None.
        noise_type (Literal["normal","uniform"]): The type of noise to generate in the Fourier domain.

    Returns:
        SpatialTensor["B C H ..."]: The generated noise field.
    """
    mesh, device, dtype = _get_mesh_device_and_dtype(mesh, device, dtype)
    if noise_type == "uniform":
        fourier_noise = torch.rand(
            *mesh_shape(mesh, batch_size=batch_size, n_channel=n_channel),
            device=device,
            dtype=dtype,
        )
    elif noise_type == "normal":
        fourier_noise = torch.randn(
            *mesh_shape(mesh, batch_size=batch_size, n_channel=n_channel),
            device=device,
            dtype=dtype,
        )
    else:
        raise ValueError(
            f"Unsupported noise_type: {noise_type}. Supported types are 'normal' and 'uniform'."
        )
    fourier_noise = mesh.fft(fourier_noise)
    fourier_noise = fourier_noise * low_pass_filter
    fourier_noise = mesh.ifft(fourier_noise).real
    if normalize_mode is not None:
        fourier_noise = normalize(fourier_noise, normalize_mode=normalize_mode)
    return fourier_noise


def truncated_fourier_series(
    mesh: Union[Sequence[tuple[float, float, int]], MeshGrid, FourierMesh],
    freq_threshold: int = 5,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    batch_size: int = 1,
    n_channel: int = 1,
    normalize_mode: Optional[
        Union[
            Literal["normal_distribution", "-1_1", "0_1"],
            Tuple[Union[float, Tuple[float, float]], Union[float, Tuple[float, float]]],
        ]
    ] = None,
    normalized_freq: bool = True,
    noise_type: Literal["normal", "uniform"] = "normal",
) -> SpatialTensor["B C H ..."]:
    r"""
    Generate a truncated Fourier series noise field on a given mesh.

    Args:
        mesh (Union[Sequence[tuple[float, float, int]], MeshGrid, FourierMesh]): The mesh on which to generate the noise.
        freq_threshold (int): The frequency threshold for truncation.
        device (Optional[torch.device]): The device on which to create the tensor.
        dtype (Optional[torch.dtype]): The data type of the tensor.
        batch_size (int): The number of batches.
        n_channel (int): The number of channels.
        normalize_mode (Optional[Union[Literal["normal_distribution", "-1_1", "0_1"],Tuple[Union[float, Tuple[float, float]], Union[float, Tuple[float, float]]],]]):
            The normalization mode for the generated noise. See `torchfsm.field.normalize` for details.
            If None, no normalization is applied. Default is None.
        normalized_freq (bool): If True, wheather to set the frequency threshold as a normalized value.
            If the domain length is 1, setting this to True or False will not make a difference.
        noise_type (Literal["normal","uniform"]): The type of noise to generate in the Fourier domain.

    Returns:
        SpatialTensor["B C H ..."]: The generated noise field.
    """
    mesh, device, dtype = _get_mesh_device_and_dtype(mesh, device, dtype)

    if normalized_freq:
        filter = mesh.normalized_low_pass_filter(freq_threshold)
        # mesh.normalized_low_pass_filter.cache_clear()
    else:
        filter = mesh.abs_low_pass_filter(freq_threshold)
        # mesh.abs_low_pass_filter.cache_clear()

    return truncated_fourier_series_custom_filter(
        mesh=mesh,
        low_pass_filter=filter,
        device=device,
        dtype=dtype,
        batch_size=batch_size,
        n_channel=n_channel,
        normalize_mode=normalize_mode,
        noise_type=noise_type,
    )


def random_truncated_fourier_series(
    mesh: Union[Sequence[tuple[float, float, int]], MeshGrid, FourierMesh],
    min_freq: int = 2,
    max_freq: int = 5,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    batch_size: int = 1,
    n_channel: int = 1,
    normalize_mode: Optional[
        Union[
            Literal["normal_distribution", "-1_1", "0_1"],
            Tuple[Union[float, Tuple[float, float]], Union[float, Tuple[float, float]]],
        ]
    ] = None,
    normalized_freq: bool = True,
    noise_type: Literal["normal", "uniform"] = "normal",
) -> SpatialTensor["B C H ..."]:
    r"""
    Generate a truncated Fourier series noise field on a given mesh.

    Args:
        mesh (Union[Sequence[tuple[float, float, int]], MeshGrid, FourierMesh]): The mesh on which to generate the noise.
        min_freq (int): The minimum frequency threshold for truncation.
        max_freq (int): The maximum frequency threshold for truncation.
        device (Optional[torch.device]): The device on which to create the tensor.
        dtype (Optional[torch.dtype]): The data type of the tensor.
        batch_size (int): The number of batches.
        n_channel (int): The number of channels.
        normalize_mode (Optional[Union[Literal["normal_distribution", "-1_1", "0_1"],Tuple[Union[float, Tuple[float, float]], Union[float, Tuple[float, float]]],]]):
            The normalization mode for the generated noise. See `torchfsm.field.normalize` for details.
            If None, no normalization is applied. Default is None.
        normalized_freq (bool): If True, wheather to set the frequency threshold as a normalized value.
            If the domain length is 1, setting this to True or False will not make a difference.
        noise_type (Literal["normal","uniform"]): The type of noise to generate in the Fourier domain.

    Returns:
        SpatialTensor["B C H ..."]: The generated noise field.
    """
    mesh, device, dtype = _get_mesh_device_and_dtype(mesh, device, dtype)

    if normalized_freq:
        filter = torch.cat(
            [
                mesh.normalized_low_pass_filter(random.randint(min_freq, max_freq + 1))
                for _ in range(batch_size)
            ],
            dim=0,
        )
        # mesh.normalized_low_pass_filter.cache_clear()
    else:
        filter = torch.cat(
            [
                mesh.abs_low_pass_filter(random.randint(min_freq, max_freq + 1))
                for _ in range(batch_size)
            ],
            dim=0,
        )
        # mesh.abs_low_pass_filter.cache_clear()

    return truncated_fourier_series_custom_filter(
        mesh=mesh,
        low_pass_filter=filter,
        device=device,
        dtype=dtype,
        batch_size=batch_size,
        n_channel=n_channel,
        normalize_mode=normalize_mode,
        noise_type=noise_type,
    )
