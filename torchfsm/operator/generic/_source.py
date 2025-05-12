from .._base import NonlinearFunc, LinearCoef, Operator
from ...mesh import FourierMesh
import torch
from torch import Tensor
from typing import Callable, Optional
from ..._type import FourierTensor, SpatialTensor


class _ImplicitUnitSourceCore(LinearCoef):
    r"""
    Implementation of the ImplicitSource operator with unit form.
    """

    def __call__(
        self, f_mesh: FourierMesh, n_channel: int
    ) -> FourierTensor["B C H ..."]:
        return torch.ones_like(f_mesh.bf_x)


class _ImplicitFuncSourceCore(NonlinearFunc):
    r"""
    Implementation of the ImplicitSource operator with function form.
    """

    def __init__(
        self,
        source_func: Callable[[torch.Tensor], torch.Tensor],
        non_linear: bool = True,
    ) -> None:
        super().__init__(non_linear)
        self.source_func = source_func

    def __call__(
        self,
        u_fft: FourierTensor["B C H ..."],
        f_mesh: FourierMesh,
        u: Optional[SpatialTensor["B C H ..."]] = None,
    ) -> FourierTensor["B C H ..."]:
        if u is None:
            u = f_mesh.ifft(u_fft).real
        return f_mesh.fft(self.source_func(u))


class ImplicitSource(Operator):

    r"""
    `ImplicitSource` allows to define a source term in the implicit form.
        Note that this class is an operator wrapper. The actual implementation of the operator is in the `_ImplicitFuncSourceCore` class and `_ImplicitUnitSourceCore`.

    Args:
        source_func (Callable[[torch.Tensor], torch.Tensor], optional): 
            The f(x) function to be used as the source term.
            This function is used to define the source term in the implicit form.
            If None, the source term will be set to the unknown variable itself, i.e., f(x) = x.

        non_linear (bool, optional): 
            If True, the source term is treated as a nonlinear function. 
            If False, it is treated as a linear function. Default is True.
            This actually controls whether the operator wil use the dealiased version of unknown variable for the source term.
            If the source term is a nonlinear function, the dealiased version of the unknown variable will be used.
    """

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
