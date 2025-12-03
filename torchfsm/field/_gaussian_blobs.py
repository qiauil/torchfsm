from ..mesh import FourierMesh, MeshGrid, mesh_shape
from .._type import SpatialTensor
from torch import Tensor
import torch
from typing import Union, Sequence, Optional,Literal, Tuple

def _mahalanobis_distance(
    locations: Tensor, 
    mean: torch.Tensor,
    variance: torch.Tensor,
):
    r"""
    Calculate the Mahalanobis distance at given locations.

    Args:
        locations (Tensor): The locations where the PDF is evaluated Shape should be (C, X, Y, Z, ...) where C is number of dimensions and X, Y, Z are additional dimensions.
        position (torch.Tensor): The mean position of the Gaussian distribution. The shape should be (C,).
        variance (torch.Tensor): The variance of the Gaussian distribution. The shape should be (C,).

    Returns:
        Tensor: The Mahalanobis distance at the specified locations, shape (X, Y, Z, ...).
    """
    k=locations.shape[0]
    if mean.ndim != 1 or variance.ndim != 1:
        raise ValueError("mean and variance must be 1-dimensional tensors.")
    if mean.shape[0] != k or variance.shape[0] != k:
        raise ValueError(f"mean and variance must have {k} elements, but got {mean.shape[0]} and {variance.shape[0]} respectively.")
    diff = locations - mean.view(-1, *((1,) * (locations.ndim - 1)))
    inv_covariance = torch.linalg.inv(torch.diag_embed(variance))
    return torch.einsum("i...,ij,j...->...", diff, inv_covariance, diff)

_batched_mahalanobis_distance = torch.vmap(_mahalanobis_distance)

#TODO: Add support to generate blob given a certain location and variance

def random_gaussian_blobs(
    mesh: Union[Sequence[tuple[float, float, int]], MeshGrid, FourierMesh],
    position_range: Tuple[float, float] = (0.4, 0.6),
    variance_range: Tuple[float, float] = (0.005, 0.01),
    batch_size: int = 1,
    n_channel: int = 1,
    device: Optional[torch.device] = None
) -> SpatialTensor["B C H ..."]:
    r"""
    Generate random Gaussian blobs on the specified mesh.
    
    Args:
        mesh (Union[Sequence[tuple[float, float, int]], MeshGrid, FourierMesh]):
            The mesh on which to generate the Gaussian blob.
        position_range (Tuple[float, float]): The range of positions for the Gaussian blob.
            Default is (0.4, 0.6).
        variance_range (Tuple[float, float]): The range of variances for the Gaussian blob.
            Default is (0.005, 0.01).
        batch_size (int): The number of batches. Default is 1.
        n_channel (int): The number of channels. Default is 1.
        device (Optional[torch.device]): The device on which to create the tensor. Default is None. 
        
    Returns:
        SpatialTensor["B C H ..."]: The generated Gaussian blob field.
    """
    if not isinstance(mesh, MeshGrid):
        mesh = MeshGrid(mesh.mesh_info if isinstance(mesh, FourierMesh) else mesh,device=device)
    if device is not None:
        if mesh.device != device:
            raise ValueError(f"Mesh device {mesh.device} does not match the specified device {device}.")
    mesh_grid= mesh.bc_mesh_grid()
    mesh_grid = [mesh_grid] if isinstance(mesh_grid, Tensor) else mesh_grid
    locations=torch.cat(mesh_grid,dim=1).squeeze(0)
    n_dim = locations.ndim-1 
    position = torch.empty(n_dim).uniform_(position_range[0],position_range[1]).to(mesh.device)
    variance = torch.empty(n_dim).uniform_(variance_range[0], variance_range[1]).to(mesh.device)
    locations = torch.stack([locations]*batch_size*n_channel, dim=0)
    position = position.unsqueeze(0).repeat(batch_size*n_channel, 1)
    variance = variance.unsqueeze(0).repeat(batch_size*n_channel, 1)
    blob=torch.exp(-0.5*_batched_mahalanobis_distance(locations, position, variance))
    return blob.view(batch_size, n_channel, *blob.shape[1:])
