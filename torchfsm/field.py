from .operator import Laplacian, ImplicitSource, ExplicitSource, Operator
from .mesh import FourierMesh, MeshGrid, mesh_shape
import torch
from typing import Union, Sequence, Optional
from ._type import SpatialTensor

def diffused_noise(
    mesh: Union[Sequence[tuple[float, float, int]], MeshGrid, FourierMesh],
    diffusion_coef: float = 1.0,
    zero_centered: bool = True,
    unit_variance: bool = True,
    unit_magnitude: bool = True,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    n_batch: int = 1,
    n_channel: int = 1,
) -> SpatialTensor["B C H W ..."]:
    if device is None and (isinstance(mesh, FourierMesh) or isinstance(mesh, MeshGrid)):
        device = mesh.device
    if dtype is None and (isinstance(mesh, FourierMesh) or isinstance(mesh, MeshGrid)):
        dtype = mesh.dtype
    u_0 = torch.randn(
        *mesh_shape(mesh, n_batch=n_batch, n_channel=n_channel),
        device=device,
        dtype=dtype
    )
    diffusion = diffusion_coef * Laplacian()
    u_0 = diffusion.integrate(u_0, dt=1, step=1, mesh=mesh)
    if zero_centered:
        u_0 = u_0 - u_0.mean()
    if unit_variance:
        u_0 = u_0 / u_0.std()
    if unit_magnitude:
        u_0 = u_0 / u_0.abs().max()
    return u_0


def kolm_force(
    x: torch.Tensor,
    drag_coef: float = -0.1,
    k: float = 4.0,
    length_scale: float = 1.0,
) -> Operator:
    return drag_coef * ImplicitSource() - ExplicitSource(
        k * torch.cos(k * length_scale * x)
    )


def wave_1d(
    x: SpatialTensor["B C H W ..."],
    min_k: int = 1,
    max_k: int = 5,
    min_amplitude: float = 0.5,
    max_amplitude: float = 1.0,
    n_polynomial: int = 5,
    zero_mean: bool = False,
    mean_shift_coef=0.3,
    batched: bool = False,
) -> SpatialTensor["B C H W ..."]:
    x_new = x / x.max() * torch.pi * 2
    y = torch.zeros_like(x)
    if not batched:
        x_new = x_new.unsqueeze(0)
        y = y.unsqueeze(0)
    batch = x_new.shape[0]
    shape = [batch, n_polynomial] + [1] * (x_new.dim() - 2)
    k = torch.randint(min_k, max_k + 1, shape, device=x.device, dtype=x.dtype)
    amplitude = (
        torch.rand(*shape, device=x.device, dtype=x.dtype)
        * (max_amplitude - min_amplitude)
        + min_amplitude
    )
    shift = torch.rand(*shape, device=x.device, dtype=x.dtype) * torch.pi * 2
    for i in range(n_polynomial):
        y += amplitude[:, i : i + 1, ...] * torch.sin(
            k[:, i : i + 1, ...] * (x_new + shift[:, i : i + 1, ...])
        )
    if not zero_mean:
        value_shift = torch.rand(
            [batch] + [1] * (x_new.dim() - 1), device=x.device, dtype=x.dtype
        )
        value_shift = (value_shift - 0.5) * 2 * (
            max_amplitude - min_amplitude
        ) * mean_shift_coef + min_amplitude
        y += value_shift
    if not batched:
        y = y.squeeze(0)
    return y
