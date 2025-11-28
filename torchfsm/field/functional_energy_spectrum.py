from ..mesh import FourierMesh, MeshGrid
from .._type import SpatialTensor
from typing import Union, Callable, Sequence, Optional, Literal
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
    n_batch: int,
    n_channels: int,
    spectrum_func: Callable[[torch.Tensor], torch.Tensor],
    normalize_mode: Optional[Literal["normal_distribution", "-1_1", "0_1"]] = None,
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
        n_batch (int): The number of batches.
        n_channels (int): The number of channels. Note that if multiple channels are used, each channel will be treated as a component of the vector field ans the energy is equally distributed among all channels.
        spectrum_func (Callable[[torch.Tensor], torch.Tensor]): A function that takes a tensor of wave numbers and returns the corresponding energy spectrum values, e.g., lambda k: 0.327*k**(-5/3).
        normalize_mode (Optional[Literal["normal_distribution","-1_1","0_1"]]): The normalization mode for the generated noise.
            If None, no normalization is applied. Default is None.
            Note that normalization will **change** the energy spectrum of the generated field.

    Returns:
        SpatialTensor["B C H ..."]: The generated initial field with shape (n_batch, n_channels, H, W, D, ...).

    """
    if not isinstance(mesh, FourierMesh):
        f_mesh = FourierMesh(mesh, device="cpu")
    else:
        f_mesh = mesh
    k_vec = f_mesh.bf_vector * (2 * torch.pi)
    norm_k = torch.norm(k_vec, dim=1, keepdim=True)
    norm_k = torch.repeat_interleave(norm_k, n_batch * n_channels, dim=0)
    spectral_magnitude = torch.nan_to_num(
        spectrum_func(norm_k) / n_channels / (2 * torch.pi * norm_k**2),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    spectral_magnitude = random_hermitian_field(torch.sqrt(spectral_magnitude))
    u_0 = f_mesh.ifft(spectral_magnitude).real.view(
        n_batch, n_channels, *k_vec.shape[2:]
    )
    if normalize_mode is not None:
        u_0 = normalize(u_0, normalize_mode=normalize_mode)
    return u_0
