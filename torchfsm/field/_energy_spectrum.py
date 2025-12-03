from ..mesh import FourierMesh, MeshGrid
from .._type import SpatialTensor
from typing import Union, Callable, Sequence, Optional, Literal, Tuple
from ._normalize import normalize
import torch


def random_hermitian_field(magnitude: SpatialTensor["B C H ..."]):
    """
    Generate a random Hermitian field in Fourier space with a given magnitude.

    Args:
        magnitude (SpatialTensor["B C H ..."]): The desired magnitude of the Fourier field

    Returns:
        SpatialTensor["B C H ..."]: A random Hermitian field in Fourier space with the specified magnitude.
    """

    real_field = torch.randn_like(magnitude)
    dims = tuple(2 + i for i in range(magnitude.ndim - 2))
    fft_field = torch.fft.fftn(real_field, dim=dims)
    current_magnitude = torch.abs(fft_field)
    current_magnitude = torch.where(current_magnitude > 1e-12, current_magnitude, 1.0)
    return magnitude * (fft_field / current_magnitude)


def functional_energy_spectrum(
    mesh: Union[Sequence[tuple[float, float, int]], MeshGrid, FourierMesh],
    spectrum_func: Callable[[torch.Tensor], torch.Tensor],
    batch_size: int=1,
    n_channel: int=1,
    normalize_mode: Optional[
        Union[
            Literal["normal_distribution", "-1_1", "0_1"],
            Tuple[Union[float, Tuple[float, float]], Union[float, Tuple[float, float]]],
        ]
    ] = None,
) -> SpatialTensor["B C H ..."]:
    """
    Generate an field $\mathbf{u}$ based on a given energy spectrum function $E(k)$ which statisfies$\frac{1}{2}\oiint_{A(K)}\hat{\mathbf{u}}(\mathbf{k})\hat{\mathbf{u}}^*(\mathbf{k})dA(k)=E(k)$
    where $\hat{\mathbf{u}}$ is the Fourier transform of $\mathbf{u}$ and $\mathbf{u}^*$ is the corresponding complex conjugate.
    How it works:
    If $\hat{\mathbf{u}}$ is independent of the direction of $\mathbf{k}$, i.e., $\hat{\mathbf{u}}(\mathbf{k})=\hat{\mathbf{u}}(k)$, then the above equation can be simplified as$\frac{4\pi k^2}{2}|\hat{\mathbf{u}}(k)|^2=E(k)$.
    Therefore, we can derive that $|\hat{\mathbf{u}}(k)|=\sqrt{\frac{E(k)}{2\pi k^2}}$.
    You can use `torchfsm.utils.collect_energy_spectrum` to verify the energy spectrum of the generated field.

    Args:
        mesh (Union[Sequence[tuple[float, float, int]], MeshGrid, FourierMesh]): The mesh or grid on which to generate the initial field.
        spectrum_func (Callable[[torch.Tensor], torch.Tensor]): A function that takes a tensor of wave numbers (SpatialTensor["B C H ..."]) and returns the corresponding energy spectrum values, e.g., lambda k: 0.327*k**(-5/3).
        batch_size (int): The number of batches. Default is 1.
        n_channel (int): The number of channels. Note that if multiple channels are used, each channel will be treated as a component of the vector field ans the energy is equally distributed among all channels. Default is 1.
        normalize_mode (Optional[Union[Literal["normal_distribution", "-1_1", "0_1"],Tuple[Union[float, Tuple[float, float]], Union[float, Tuple[float, float]]],]]): 
            The normalization mode for the generated noise. See `torchfsm.field.normalize` for details.
            Note that the normalization will change the energy spectrum of the final generated field.
            If None, no normalization is applied. Default is None.

    Returns:
        SpatialTensor["B C H ..."]: The generated initial field with shape (batch_size, n_channel, H, W, D, ...). The device and dtype are inherited from the input mesh.

    """
    f_mesh = FourierMesh(mesh) if not isinstance(mesh, FourierMesh) else mesh
    k_vec = f_mesh.bf_vector * (2 * torch.pi)
    norm_k = torch.norm(k_vec, dim=1, keepdim=True)
    norm_k = torch.repeat_interleave(norm_k, batch_size, dim=0)
    norm_k = torch.repeat_interleave(norm_k, n_channel, dim=1)
    spectral_magnitude = torch.nan_to_num(
        spectrum_func(norm_k) / n_channel / (2 * torch.pi * norm_k**2),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    spectral_magnitude = random_hermitian_field(torch.sqrt(spectral_magnitude))
    u_0 = f_mesh.ifft(spectral_magnitude).real.view(
        batch_size, n_channel, *k_vec.shape[2:]
    )
    if normalize_mode is not None:
        u_0 = normalize(u_0, normalize_mode=normalize_mode)
    return u_0

def random_power_law_energy_spectrum(
    mesh: Union[Sequence[tuple[float, float, int]], MeshGrid, FourierMesh],
    min_power: float=-5.0,
    max_power: float=-2.0,
    batch_size: int=1,
    n_channel: int=1,
    normalize_mode: Optional[
        Union[
            Literal["normal_distribution", "-1_1", "0_1"],
            Tuple[Union[float, Tuple[float, float]], Union[float, Tuple[float, float]]],
        ]
    ] = None,
) -> SpatialTensor["B C H ..."]:
    """
    Generate a random field with a power-law energy spectrum on a given mesh, i.e., $E(k)=k^p$
    
    Args:
        mesh (Union[Sequence[tuple[float, float, int]], MeshGrid, FourierMesh]): The mesh or grid on which to generate the initial field.
        min_power (float): The minimum power-law exponent.
        max_power (float): The maximum power-law exponent.
        batch_size (int): The number of batches. Default is 1.
        n_channel (int): The number of channels. Default is 1.
        normalize_mode (Optional[Union[Literal["normal_distribution", "-1_1", "0_1"],Tuple[Union[float, Tuple[float, float]], Union[float, Tuple[float, float]]],]]): 
            The normalization mode for the generated noise. See `torchfsm.field.normalize` for details.
            Note that the normalization will change the energy spectrum of the final generated field.
            If None, no normalization is applied. Default is None.
    
    Returns:
        SpatialTensor["B C H ..."]: The generated initial field with shape (batch_size, n_channel, H, W, D, ...). The device and dtype are inherited from the input mesh.
    
    """
    mesh=FourierMesh(mesh) if not isinstance(mesh,FourierMesh) else mesh
    powers=torch.rand(batch_size,1,*tuple([1,]*mesh.n_dim),device=mesh.device)*(max_power-min_power)+min_power
    return functional_energy_spectrum(
        mesh=mesh,
        spectrum_func=lambda k: torch.pow(k,powers),
        batch_size=batch_size,
        n_channel=n_channel,
        normalize_mode=normalize_mode,
    )