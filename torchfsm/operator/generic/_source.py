from .._base import NonlinearFunc, LinearCoef, Operator
from ...mesh import FourierMesh
import torch
from torch import Tensor
from typing import Callable, Optional
from ..._type import FourierTensor, SpatialTensor


class _ImplicitUnitSourceCore(LinearCoef):

    def __call__(
        self, f_mesh: FourierMesh, n_channel: int
    ) -> FourierTensor["B C H W ..."]:
        return torch.ones_like(f_mesh.bf_x)


class _ImplicitFuncSourceCore(NonlinearFunc):

    def __init__(
        self,
        source_func: Callable[[torch.Tensor], torch.Tensor],
        non_linear: bool = True,
    ) -> None:
        super().__init__(non_linear)
        self.source_func = source_func

    def __call__(
        self,
        u_fft: FourierTensor["B C H W ..."],
        f_mesh: FourierMesh,
        n_channel: int,
        u: SpatialTensor["B C H W ..."] | None,
    ) -> FourierTensor["B C H W ..."]:
        if u is None:
            u = f_mesh.ifft(u_fft).real
        return f_mesh.fft(self.source_func(u))


class ImplicitSource(Operator):

    def __init__(
        self,
        source_func: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        non_linear: bool = True,
    ) -> None:
        if source_func is None:
            generator = lambda f_mesh, n_channel: _ImplicitUnitSourceCore()
        else:
            generator = lambda f_mesh, n_channel: _ImplicitFuncSourceCore(
                source_func, non_linear
            )
        super().__init__(generator)
