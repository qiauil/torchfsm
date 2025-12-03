import torch
from typing import Union, Sequence, Optional, Annotated, Tuple
#from functools import lru_cache
from ._utils import format_device_dtype, default
import numpy as np
from ._type import SpatialTensor, FourierTensor, ValueList


class MeshGrid:
    """
    An interable class that reprents a mesh grid.
    This class is particularly useful for generating the initial condition.
    The length of the class is the number of mesh dimensions.
    The attribute x, y, z are the mesh grid for the first three dimension.
    You can also access the mesh grid for other dimension by indexing the object.
    E.g., `mesh_grid[0]` is the mesh grid for the first dimension, equivalent to x.
    There is no limit for the number of dimension.
    Assume that the number of points in each dimension is $n_1, n_2, n_3, \cdots, n_k$, the mesh grid will be of shape $(n_1,n_2,n_3,...,n_k)$.
    While for the attribute x, y, z, the shape will be $(n_1)$, $(n_2)$, $(n_3)$ respectively.

    Args:
        mesh_info (Sequence[tuple[float,float,int]]): sequence of tuple (start,end,n_points) for each dimension
        device: device of the mesh.
        dtype: data type of the mesh.

    Methods:
        mesh_grid: Generate the mesh grid for all dimensions.
        bc_mesh_grid: Generate the mesh grid with batch size and channel size.

    """

    def __init__(
        self, mesh_info: Sequence[tuple[float, float, int]], device=None, dtype=None
    ) -> None:
        for dim_i in mesh_info:
            if len(dim_i) != 3:
                raise ValueError(
                    "each dimension should be a tuple of (start,end,n_points)"
                )
        self.mesh_info = mesh_info
        self.meshs = [[] for _ in range(len(mesh_info))]
        self._dim_names = ["x", "y", "z"]
        self.device, self.dtype = format_device_dtype(device, dtype)
        self.n_dim = len(mesh_info)

    def __len__(self):
        return len(self.mesh_info)

    def __getitem__(self, idx: int) -> SpatialTensor["H ..."]:
        if len(self.meshs) <= idx:
            if idx > 2:
                raise ValueError(f"mesh dim with id{idx} is not defined")
            else:
                raise ValueError(f"{self._dim_names[idx]} dim is not defined")
        if len(self.meshs[idx]) == 0:
            self.meshs[idx] = (
                (self.mesh_info[idx][1] - self.mesh_info[idx][0])
                * torch.arange(
                    0, self.mesh_info[idx][2], device=self.device, dtype=self.dtype
                )
                / self.mesh_info[idx][2]
            )
        return self.meshs[idx]

    @property
    def x(self) -> SpatialTensor["H"]:
        """
        Mesh grid for the first dimension
        """
        return self[0]

    @property
    def y(self) -> SpatialTensor["H"]:
        """
        Mesh grid for the second dimension
        """
        return self[1]

    @property
    def z(self) -> SpatialTensor["H"]:
        """
        Mesh grid for the third dimension
        """
        return self[2]

    def mesh_grid(
        self, numpy=False
    ) -> ValueList[
        Union[SpatialTensor["H ..."], Annotated[np.ndarray, "Spatial, H ..."]]
    ]:
        """
        Generate the mesh grid for all dimensions.
        The shape of the mesh grid will be (n1,n2,n3,...,nk).

        Args:
            numpy (bool): whether to return the mesh grid as numpy array

        Returns:
            torch.Tensor: mesh grid for all dimensions
        """
        if numpy:
            mesh_grid = np.meshgrid(*[self[i] for i in range(len(self))], indexing="ij")
        else:
            mesh_grid = torch.meshgrid(
                *[self[i] for i in range(len(self))], indexing="ij"
            )
        if len(mesh_grid) == 1:
            return mesh_grid[0]
        return mesh_grid

    def bc_mesh_grid(
        self, batch_size: int = 1, n_channel: int = 1, numpy=False
    ) -> ValueList[
        Union[SpatialTensor["B C H ..."], Annotated[np.ndarray, "Spatial, B C H ..."]]
    ]:
        """
        Generate the mesh grid with batch size and channel size.
        The shape of the mesh grid will be (batch_size,n_channel,n1,n2,n3,...,nk).

        Args:
            batch_size (int): batch size
            n_channel (int): channel size
            numpy (bool): whether to return the mesh grid as numpy array

        Returns:
            torch.Tensor: mesh grid with batch size and channel size
        """
        bc_mesh_grid = []
        mesh_grid = self.mesh_grid()
        if not isinstance(mesh_grid, Sequence):
            mesh_grid = [mesh_grid]
        for i in range(len(mesh_grid)):
            bc_mesh_grid.append(
                mesh_grid[i]
                .reshape(1, 1, *mesh_grid[i].shape)
                .repeat(batch_size, n_channel, *[1] * len(mesh_grid[i].shape))
            )
        if numpy:
            mesh_grid = [i.cpu().numpy() for i in bc_mesh_grid]
        if len(bc_mesh_grid) == 1:
            return bc_mesh_grid[0]
        return bc_mesh_grid

    def to(self, device=None, dtype=None):
        self.__init__(self.mesh_info, device=device, dtype=dtype)


class FFTFrequency:
    """
    FFT frequency for each dimension.
    The length of the class is determined by the number of dimension.
    The attribute f_x, f_y, f_z are the fft frequency for the first three dimension.
    You can also access the fft frequency for other dimension by indexing the object.
    E.g., fft_frequency[0] is the fft frequency for the first dimension, equivalent to f_x.
    There is no limit for the number of dimension.

    Args:
        mesh_info (Sequence[tuple[float,float,int]]): sequence of tuple (start,end,n_points) for each dimension
        device: device for the fft frequency
        dtype: data type for the fft frequency

    """

    def __init__(
        self,
        mesh_info: Sequence[tuple[float, float, int]] = [],
        device=None,
        dtype=None,
    ) -> None:
        self._dim_names = ["x", "y", "z"]
        self.mesh_info = mesh_info
        self.fs = [[] for _ in range(len(mesh_info))]
        self.device, self.dtype = format_device_dtype(device, dtype)

    def __len__(self):
        return len(self.mesh_info)

    def __getitem__(self, idx: int):
        if len(self.fs) <= idx:
            if idx > 2:
                raise ValueError(f"fft frequency with id{idx} is not defined")
            else:
                raise ValueError(f"{self._dim_names[idx]} fft frequency is not defined")
        if len(self.fs[idx]) == 0:
            self.fs[idx] = torch.fft.fftfreq(
                self.mesh_info[idx][2],
                (self.mesh_info[idx][1] - self.mesh_info[idx][0])
                / self.mesh_info[idx][2],
                device=self.device,
                dtype=self.dtype,
            )
        return self.fs[idx]

    @property
    def f_x(self) -> torch.Tensor:
        """
        FFT frequency for the first dimension
        """
        return self[0]

    @property
    def f_y(self) -> torch.Tensor:
        """
        FFT frequency for the second dimension
        """
        return self[1]

    @property
    def f_z(self) -> torch.Tensor:
        """
        FFT frequency for the third dimension
        """
        return self[2]

    def to(self, device=None, dtype=None):
        self.__init__(self.mesh_info, device=device, dtype=dtype)


class BroadcastedFFTFrequency:
    """
    Broadcasted fft frequency for each dimension.
    The fft frequency is broadcasted to the shape of the value field.
    For example, if the value field is of shape (batch_size,1,nx,ny,nz),
    the fft frequency for the first, second and third dimension will be broadcasted to (1,1,nx,1,1), (1,1,1,ny,1), (1,1,1,1,nz) respectively.
    The length of the class is determined by the number of dimension.
    The attribute bf_x, bf_y, bf_z are the broadcasted fft frequency for the first three dimension.
    You can also access the broadcasted fft frequency for other dimension by indexing the object.
    E.g., broadcasted_fft_frequency[0] is the broadcasted fft frequency for the first dimension, equivalent to bf_x.

    Args:
        fft_frequency (FFTFrequency): FFTFrequency object
    """

    def __init__(self, fft_frequency: FFTFrequency) -> None:
        self.fft_frequency = fft_frequency
        self.bdks = [[] for _ in range(len(self.fft_frequency))]
        self._dim_names = self.fft_frequency._dim_names
        self.mesh_shape = tuple((i[-1] for i in self.fft_frequency.mesh_info))
        self._bf_vector = None

    def __len__(self):
        return len(self.fft_frequency)

    def __getitem__(self, idx: int) -> torch.Tensor:
        if len(self) <= idx:
            if idx > 2:
                raise ValueError(f"fft frequency with id{idx} is not defined")
            else:
                raise ValueError(f"{self._dim_names[idx]} fft frequency is not defined")
        if len(self.bdks[idx]) == 0:
            shapes = [1] * (len(self.fft_frequency) + 2)
            shapes[idx + 2] = self.fft_frequency[idx].shape[0]
            self.bdks[idx] = self.fft_frequency[idx].reshape(*shapes)
        return self.bdks[idx]

    @property
    def bf_vector(self):
        if self._bf_vector is None:
            shapes = (1, 1) + self.mesh_shape
            bks = []
            for i in range(len(self)):
                new_shape = list(shapes)
                new_shape[i + 2] = 1
                bks.append(self[i].repeat(*new_shape))
            self._bf_vector = torch.cat(bks, dim=1)
        return self._bf_vector

    @property
    def bf_vector_norm(self) -> torch.Tensor:
        """
        Broadcasted fft frequency vector norm
        """
        return torch.sqrt(torch.sum(self.bf_vector ** 2, dim=1, keepdim=True))

    @property
    def bf_x(self) -> torch.Tensor:
        """
        Broadcasted fft frequency for the first dimension
        """
        return self[0]

    @property
    def bf_y(self) -> torch.Tensor:
        """
        Broadcasted fft frequency for the second dimension
        """
        return self[1]

    @property
    def bf_z(self) -> torch.Tensor:
        """
        Broadcasted fft frequency for the third dimension
        """
        return self[2]

    def to(self, device=None, dtype=None):
        self.fft_frequency.to(device=device, dtype=dtype)
        self.__init__(self.fft_frequency)


class FourierMesh:
    """
    A class contains the fft frequency information and basic deritivate operators for a mesh system.
    This class is used inside of an Operator class.

    Args:
        mesh (Union[Sequence[tuple[float, float, int]],MeshGrid]): mesh information for the Fourier spectral method
            it can be a sequence of tuple (start,end,n_points) for each dimension or a MeshGrid object
        device: device for the fft frequency
        dtype: data type for the fft frequency

    Attributes:
        k (FFTFrequency): fft frequency for each dimension
            It is indexed by the dimension id, e.g., k[0] is the fft frequency for the first dimension
        bk (BroadcastedFFTFrequency): broadcasted fft frequency for each dimension
            It is indexed by the dimension id, e.g., bk[0] is the broadcasted fft frequency for the first dimension
        f_x (torch.Tensor): fft frequency for the first dimension
        f_y (torch.Tensor): fft frequency for the second dimension
        f_z (torch.Tensor): fft frequency for the third dimension
        bf_x (torch.Tensor): broadcasted fft frequency for the first dimension
        bf_y (torch.Tensor): broadcasted fft frequency for the second dimension
        bf_z (torch.Tensor): broadcasted fft frequency for the third dimension
        n_dim (int): number of dimension
        fft_dim (tuple): tuple of the dimension for the fft operation

    Args:
        mesh (Union[Sequence[tuple[float, float, int]],MeshGrid]): mesh information for the Fourier spectral method
            it can be a sequence of tuple (start,end,n_points) for each dimension or a MeshGrid object
        device: device for the fft frequency. If you initialze the obkjct with a MeshGrid object and you dont specify this parameter, the device will be the same as the MeshGrid object.
        dtype: data type for the fft frequency. If you initialze the obkjct with a MeshGrid object and you dont specify this parameter, the dtype will be the same as the MeshGrid object.
    """

    def __init__(
        self,
        mesh: Union[Sequence[tuple[float, float, int]], MeshGrid],
        device=None,
        dtype=None,
    ) -> None:
        if isinstance(mesh, MeshGrid):
            self.mesh_info = mesh.mesh_info
            self.device = default(device, mesh.device)
            self.dtype = default(dtype, mesh.dtype)
            self.device, self.dtype = format_device_dtype(self.device, self.dtype)
        else:
            self.mesh_info = mesh
            self.device, self.dtype = format_device_dtype(device, dtype)
        self.f = FFTFrequency(self.mesh_info, device=self.device, dtype=self.dtype)
        self.bf = BroadcastedFFTFrequency(self.f)
        self.n_dim = len(self.mesh_info)
        self.fft_dim = tuple(-1 * (i + 1) for i in range(self.n_dim))
        self._default_rel_freq_threshold = 2 / 3

    def set_default_rel_freq_threshold(self, threshold: float):
        self._default_rel_freq_threshold = threshold
        #self.low_pass_filter.cache_clear()

    @property
    def f_x(self) -> torch.Tensor:
        """
        Fft frequency for the first dimension
        """
        return self.f.f_x

    @property
    def f_y(self) -> torch.Tensor:
        """
        Fft frequency for the second dimension
        """
        return self.f.f_y

    @property
    def f_z(self) -> torch.Tensor:
        """
        Fft frequency for the third dimension
        """
        return self.f.f_z

    @property
    def bf_x(self) -> torch.Tensor:
        """
        Broadcasted fft frequency for the first dimension
        """
        return self.bf.bf_x

    @property
    def bf_y(self) -> torch.Tensor:
        """
        Broadcasted fft frequency for the second dimension
        """
        return self.bf.bf_y

    @property
    def bf_z(self) -> torch.Tensor:
        """
        Broadcasted fft frequency for the third dimension
        """
        return self.bf.bf_z

    @property
    def bf_vector(self) -> torch.Tensor:
        """
        Broadcasted fft frequency for all dimensions
        """
        return self.bf.bf_vector
    
    @property
    def bf_vector_norm(self) -> torch.Tensor:
        """
        Broadcasted fft frequency vector norm
        """
        return self.bf.bf_vector_norm

    #@lru_cache()
    def grad(self, dim_i: int, order: int) -> FourierTensor["B C H ..."]:
        """
        Linear operator for the nth order gradient w.r.t the ith dimension.
        """
        return (2j * torch.pi * self.bf[dim_i]) ** order

    #@lru_cache()
    def laplacian(self) -> FourierTensor["B C H ..."]:
        """
        Linear operator for the nth order Laplacian.
        """
        return self.nabla(2)

    #@lru_cache()
    def invert_laplacian(self) -> FourierTensor["B C H ..."]:
        """
        Linear operator for the nth order inverse Laplacian.
        """
        lap = self.laplacian()
        return torch.where(lap == 0, 1.0, 1 / lap)

    #@lru_cache()
    def nabla(self, order: int = 1) -> FourierTensor["B C H ..."]:
        """
        Linear operator for the nth order gradient.
        """
        return sum([self.grad(dim_i, order) for dim_i in range(len(self.bf))])

    #@lru_cache()
    def invert_nabla(self, order: int = 1) -> FourierTensor["B C H ..."]:
        """
        Linear operator for the nth order inverse gradient.
        """
        nab = self.nabla(order)
        return torch.where(nab == 0, 1.0, 1 / nab)

    #@lru_cache()
    def nabla_vector(self, order: int) -> FourierTensor["B C H ..."]:
        """
        Linear operator vector for the nth order gradient.
        """
        return (2j * torch.pi * self.bf.bf_vector) ** order

    #@lru_cache()
    def low_pass_filter(self, rel_freq_threshold: Optional[float] = None) -> torch.Tensor:
        """
        Low pass filter mask for the Fourier coefficients.
        
        The mask is a tensor of the same shape as the Fourier coefficients, with values 1 for frequencies below the threshold and 0 for frequencies above the threshold.
        Args:
            rel_freq_threshold (Optional[float]): The relative frequency threshold for the low pass filter. If None, the default frequency threshold will be used.
                Default is None, which uses the value set by `set_default_rel_freq_threshold`.
        
        Returns:
            torch.Tensor: A mask tensor with the same shape as the Fourier coefficients, where values are 1 for frequencies below the threshold and 0 for frequencies above the threshold.
        """
        rel_freq_threshold = default(rel_freq_threshold, self._default_rel_freq_threshold)
        mask = torch.ones_like(self.nabla().real)
        for i in range(len(self.bf)):
            abs_f = self.bf[i].abs()
            mask *= torch.where(abs_f > abs_f.max() * rel_freq_threshold, 0, 1)
        return mask.to(device=self.device, dtype=self.dtype)

    #@lru_cache()
    def abs_low_pass_filter(self, abs_freq_threshold: int) -> torch.Tensor:
        """
        Low pass filter mask for the Fourier coefficients.
        
        The mask is a tensor of the same shape as the Fourier coefficients, with values 1 for frequencies below the threshold and 0 for frequencies above the threshold.
        Args:
            abs_rel_freq_threshold (Optional[float]): The absolute frequency threshold for the low pass filter.
        
        Returns:
            torch.Tensor: A mask tensor with the same shape as the Fourier coefficients, where values are 1 for frequencies below the threshold and 0 for frequencies above the threshold.
        """
        mask = torch.ones_like(self.nabla().real)
        for i in range(len(self.bf)):
            abs_f = self.bf[i].abs()
            mask *= torch.where(abs_f > abs_freq_threshold, 0, 1)
        return mask.to(device=self.device, dtype=self.dtype)
    
    #@lru_cache()
    def normalized_low_pass_filter(self, normalized_freq_threshold: float) -> torch.Tensor:
        """
        Low pass filter mask for the Fourier coefficients based on normalized frequency.
        This is equivalent to the `abs_low_pass_filter` with abs_freq_threshold = normalized_freq_threshold/domain_length

        The mask is a tensor of the same shape as the Fourier coefficients, with values 1 for frequencies below the threshold and 0 for frequencies above the threshold.
        Args:
            normalized_freq_threshold (float): The normalized frequency threshold for the low pass filter.

        Returns:
            torch.Tensor: A mask tensor with the same shape as the Fourier coefficients, where values are 1 for frequencies below the threshold and 0 for frequencies above the threshold.
        """
        mask = torch.ones_like(self.nabla().real)
        for i in range(len(self.bf)):
            abs_f = self.bf[i].abs()
            mask *= torch.where(abs_f > normalized_freq_threshold/(self.mesh_info[i][1]-self.mesh_info[i][0]), 0, 1)
        return mask.to(device=self.device, dtype=self.dtype)
    
    def fft(self, u) -> FourierTensor["B C H ..."]:
        """
        Fast Fourier Transform
        """
        return torch.fft.fftn(u, dim=self.fft_dim)

    def ifft(self, u_fft) -> SpatialTensor["B C H ..."]:
        """
        Inverse Fast Fourier Transform
        """
        return torch.fft.ifftn(u_fft, dim=self.fft_dim)

    def to(self, device=None, dtype=None):
        """
        Move the mesh to a different device and dtype.
        """
        """
        self.__init__(self.mesh_info, device=device, dtype=dtype)
        self.grad.cache_clear()
        self.laplacian.cache_clear()
        self.invert_laplacian.cache_clear()
        self.nabla.cache_clear()
        self.invert_nabla.cache_clear()
        self.nabla_vector.cache_clear()
        self.low_pass_filter.cache_clear()
        """


def mesh_shape(
    mesh: Union[Sequence[tuple[float, float, int]], MeshGrid, FourierMesh],
    batch_size: int = 1,
    n_channel: int = 1,
) -> Tuple:
    """
    Get the shape of the mesh.
    The shape is in the form of (batch_size, n_channel, n1, n2, n3, ...).

    Args:
        mesh (Union[Sequence[tuple[float, float, int]], MeshGrid, FourierMesh]): The mesh to get the shape from.
            If a sequence is provided, it should be in the form of [(x_min, x_max, n_points), ...].
        batch_size (int): The number of batches. Default is 1.
        n_channel (int): The number of channels. Default is 1.

    Returns:
        Tuple: The shape of the mesh.
    """
    if isinstance(mesh, FourierMesh) or isinstance(mesh, MeshGrid):
        mesh = mesh.mesh_info
    return tuple([batch_size, n_channel] + [m[2] for m in mesh])
