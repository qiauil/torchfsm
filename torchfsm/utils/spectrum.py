from .._type import SpatialTensor
from ..mesh import FourierMesh, MeshGrid
from typing import Union, Sequence, Optional
from collections import defaultdict
from tqdm.auto import tqdm
import torch


def collect_energy_spectrum(
    u: SpatialTensor["1 C H ..."],
    mesh: Optional[
        Union[Sequence[tuple[float, float, int]], MeshGrid, FourierMesh]
    ] = None,
    progressive: bool = False,
    exact=False,
    n_bins: Optional[int] = None,
):
    """
    Collect the energy spectrum from a spatial tensor with batch size 1.

    Args:
        u (SpatialTensor["1 C H ..."]): The input spatial tensor with batch size 1.
        mesh (Optional[Union[Sequence[tuple[float, float, int]], MeshGrid, FourierMesh]]): The mesh grid or Fourier mesh to use for FFT. If None, a default mesh is created based on the shape of u. Default is None.
        progressive (bool): Whether to show a progress bar during computation.  Default is False.
        exact (bool): Whether to compute the exact energy spectrum without binning. Default is False.
        n_bins (Optional[int]): Number of bins to smooth the spectrum if exact is False. If None, it defaults to min(50, u.shape[-1] // 2). Default is None.

    Returns:
        Tuple[List[float], List[float]]: Two lists containing the wave numbers and their corresponding energy spectrum values.
    """
    if u.shape[0] != 1:
        raise ValueError("Batch size of u must be 1 for collecting energy spectrum.")
    if mesh is None:
        mesh = [(0, 1, size_i) for size_i in u.shape[2:]]
    if not isinstance(mesh, FourierMesh):
        f_mesh = FourierMesh(mesh, device=u.device)
    else:
        f_mesh = mesh
    u_fft = f_mesh.fft(u)
    k_vec_norm = torch.norm(f_mesh.bf_vector * 2 * torch.pi, dim=1, keepdim=True)
    energy_fft = (
        0.5
        * torch.sum(torch.abs(u_fft) ** 2, dim=1, keepdim=True)
        * (4 * torch.pi * k_vec_norm**2)
    )
    if not exact:
        k_max = torch.max(k_vec_norm)
        if n_bins is None:
            n_bins = min(50, u.shape[-1] // 2)
        k_bins = torch.linspace(0, k_max, n_bins + 1)
        k_centers = (k_bins[1:] + k_bins[:-1]) / 2
        radial = torch.zeros(len(k_centers), device=k_vec_norm.device)
        for i in range(len(k_centers)):
            mask = (k_vec_norm >= k_bins[i]) & (k_vec_norm < k_bins[i + 1])
            if torch.any(mask):
                radial[i] = torch.mean(energy_fft[mask])
        return k_centers.cpu().numpy().tolist(), radial.cpu().numpy().tolist()
    else:
        re = defaultdict(list)
        if progressive:
            iterator = tqdm(
                zip(k_vec_norm.view(-1), energy_fft.view(-1)), total=k_vec_norm.numel()
            )
        else:
            iterator = zip(k_vec_norm.view(-1), energy_fft.view(-1))
        for k, e in iterator:
            re[k.item()].append(e.item())
        for k, e in re.items():
            if k == 0:
                re[k] = 0.0  # Avoid division by zero for k=0
            else:
                re[k] = sum(e) / len(e)
        sorted_k = sorted(re.keys())
        sorted_e = [re[k] for k in sorted_k]
        sorted_k = sorted_k[1:]
        sorted_e = sorted_e[1:]
    return sorted_k, sorted_e
