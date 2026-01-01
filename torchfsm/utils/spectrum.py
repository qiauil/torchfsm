from .._type import SpatialTensor
from ..mesh import FourierMesh, MeshGrid
from typing import Union, Sequence,Optional
from collections import defaultdict
from tqdm.auto import tqdm
import torch
import numpy as np

def _smooth_spectrum(k_radial,E,n_bins):
    if not isinstance(k_radial,np.ndarray):
        k_radial=np.array(k_radial)
    if not isinstance(E,np.ndarray):
        E=np.array(E)
    k_max = np.max(k_radial)
    k_bins = np.linspace(0, k_max, n_bins + 1)
    k_centers = (k_bins[1:] + k_bins[:-1]) / 2
    gt_radial = np.zeros(len(k_centers))
    for i in range(len(k_centers)):
        mask = (k_radial >= k_bins[i]) & (k_radial < k_bins[i+1])
        if np.any(mask):
            gt_radial[i] = np.mean(E[mask])
    return k_centers.tolist(), gt_radial.tolist()


def collect_energy_spectrum(
    u: SpatialTensor["1 C H ..."],
    mesh: Union[Sequence[tuple[float, float, int]], MeshGrid, FourierMesh],
    progressive: bool = False,
    n_bins: Optional[int] = None,
):
    """
    Collect the energy spectrum from a spatial tensor with batch size 1.

    Args:
        u (SpatialTensor["1 C H ..."]): The input spatial tensor with batch size 1.
        mesh (Union[Sequence[tuple[float, float, int]], MeshGrid, FourierMesh]): The mesh grid or Fourier mesh corresponding to the spatial tensor.
        progressive (bool): Whether to show a progress bar during computation.  Default is False.
        n_bins (Optional[int]): Number of bins to smooth the spectrum. If None, no smoothing is applied. Default is None.

    Returns:
        Tuple[List[float], List[float]]: Two lists containing the wave numbers and their corresponding energy spectrum values.
    """
    if u.shape[0] != 1:
        raise ValueError("Batch size of u must be 1 for collecting energy spectrum.")
    if not isinstance(mesh, FourierMesh):
        f_mesh = FourierMesh(mesh, device=u.device)
    else:
        f_mesh = mesh
    u_fft = f_mesh.fft(u)
    energy_fft = 0.5 * torch.sum(torch.abs(u_fft) ** 2, dim=1, keepdim=True)
    k_vec_norm = torch.norm(f_mesh.bf_vector * 2 * torch.pi, dim=1, keepdim=True)
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
            re[k] = sum(e) / len(e) * (4 * torch.pi * k**2)
    sorted_k = sorted(re.keys())
    sorted_e = [re[k] for k in sorted_k]
    sorted_k=sorted_k[1:]
    sorted_e=sorted_e[1:]
    if n_bins is not None:
        if n_bins < len(sorted_k):
            raise ValueError("n_bins must be greater than or equal to the number of unique wave numbers.")
        sorted_k, sorted_e = _smooth_spectrum(sorted_k, sorted_e, n_bins)
    return sorted_k, sorted_e
