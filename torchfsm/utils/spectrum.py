from ..mesh import FourierMesh, MeshGrid
from .._type import SpatialTensor
from typing import Union, Sequence
from collections import defaultdict
from tqdm.auto import tqdm
import torch


def collect_energy_spectrum(
    u: SpatialTensor["1 C H ..."],
    mesh: Union[Sequence[tuple[float, float, int]], MeshGrid, FourierMesh],
    progressive: bool = False,
):
    """
    Collect the energy spectrum from a spatial tensor with batch size 1.

    Args:
        u (SpatialTensor["1 C H ..."]): The input spatial tensor with batch size 1.
        mesh (Union[Sequence[tuple[float, float, int]], MeshGrid, FourierMesh]): The mesh grid or Fourier mesh corresponding to the spatial tensor.
        progressive (bool): Whether to show a progress bar during computation.  Default is False.

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
    return sorted_k[1:], sorted_e[1:]
