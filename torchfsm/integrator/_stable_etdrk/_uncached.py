import torch
from abc import ABC, abstractmethod
from typing import Callable, Optional, Sequence
from ._setdrk_step import setdrk1_step, setdrk2_step, setdrk3_step, setdrk4_step


def roots_of_unity(M: int, device=None, dtype=None) -> torch.Tensor:
    """
    Return (complex-valued) array with M roots of unity.
    """
    # return torch.exp(1j * torch.pi * (torch.arange(1, M+1) - 0.5) / M)
    return torch.exp(
        2j * torch.pi * (torch.arange(1, M + 1, device=device, dtype=dtype) - 0.5) / M
    )


class _UnCachedSETDRKBase(ABC):
    r"""
    Base class for the stable ETDRK integrators.

    The stabilizing trick of Kassam & Trefethen evaluates each ETDRK coefficient
    as the mean of a function of ``lr = integration_radius * omega_k + dt *
    linear_coef`` over the ``n_integration_points`` roots of unity ``omega_k``.
    Computing this mean for all points at once requires materializing a tensor of
    shape ``(*linear_coef.shape, n_integration_points)`` (plus several
    intermediates of the same shape such as ``exp(lr)``), i.e.
    ``n_integration_points`` times the size of ``linear_coef``. For a fine mesh or
    a large ``n_integration_points`` this easily exhausts GPU memory, even though
    the result is immediately averaged back down to the size of ``linear_coef``.

    To avoid this, the mean is accumulated chunk-by-chunk along the
    integration-points axis: only ``integration_chunk_size`` points are
    materialized at a time, so the peak memory is ``integration_chunk_size``
    (rather than ``n_integration_points``) times ``linear_coef``, while the
    computation stays entirely on the GPU. The accumulated result is identical up
    to floating-point summation order regardless of the chunk size; only the peak
    memory and the number of (one-time, build-time) loop iterations change.

    Args:
        integration_chunk_size (Optional[int]): Number of integration points
            evaluated at once when building the coefficients. Smaller values use
            less peak memory but require more build-time iterations; larger values
            build faster but use more memory. Defaults to ``1`` (lowest memory),
            which is safe even for very large ``n_integration_points``. Pass
            ``None`` to evaluate all points in a single chunk (fastest build,
            highest memory).
    """

    def __init__(
        self,
        dt: float,
        linear_coef: torch.Tensor,
        nonlinear_func: Callable[[torch.Tensor], torch.Tensor],
        n_integration_points: int = 16,
        integration_radius: float = 1.0,
        integration_chunk_size: Optional[int] = 1,
    ):
        self.dt = dt
        self._nonlinear_func = nonlinear_func
        self._exp_term = torch.exp(self.dt * linear_coef)
        self.n_integration_points = n_integration_points
        self.integration_radius = integration_radius
        # ``None`` means "evaluate all integration points in a single chunk".
        self.integration_chunk_size = (
            n_integration_points
            if integration_chunk_size is None
            else integration_chunk_size
        )

    def _coef_means(
        self,
        linear_coef: torch.Tensor,
        funcs: Sequence[Callable[[torch.Tensor], torch.Tensor]],
    ) -> list:
        """
        For each ``f`` in ``funcs`` compute ``dt * mean_k(f(lr_k)).real`` where
        ``lr_k = integration_radius * omega_k + dt * linear_coef`` and ``omega_k``
        are the ``n_integration_points`` roots of unity.

        The mean over the integration points is accumulated chunk-by-chunk so the
        expanded ``lr`` tensor never exceeds ``integration_chunk_size`` points in
        the last dimension, avoiding the OOM caused by materializing all points.
        """
        M = self.n_integration_points
        chunk_size = min(self.integration_chunk_size, M)
        roots = roots_of_unity(
            M, device=linear_coef.device, dtype=linear_coef.real.dtype
        )
        scaled_coef = (linear_coef * self.dt).unsqueeze(-1)  # (*shape, 1)
        sums = [None] * len(funcs)
        for start in range(0, M, chunk_size):
            lr = (
                self.integration_radius * roots[start : start + chunk_size]
                + scaled_coef
            )  # (*shape, <=chunk_size)
            for i, f in enumerate(funcs):
                partial = f(lr).sum(dim=-1)
                sums[i] = partial if sums[i] is None else sums[i] + partial
        return [(self.dt / M * s).real for s in sums]

    @abstractmethod
    def step(
        self,
        u_hat,
    ):
        """
        Advance the state in Fourier space.
        """


class UnCachedSETDRK1(_UnCachedSETDRKBase):
    """
    First-order UnCachedSETDRK method.
    """

    def __init__(
        self,
        dt: float,
        linear_coef: torch.Tensor,
        nonlinear_func: Callable[[torch.Tensor], torch.Tensor],
        n_integration_points: int = 16,
        integration_radius: float = 1.0,
        integration_chunk_size: Optional[int] = 1,
    ):
        super().__init__(
            dt,
            linear_coef,
            nonlinear_func,
            n_integration_points,
            integration_radius,
            integration_chunk_size,
        )
        (self._coef_1,) = self._coef_means(
            linear_coef,
            [lambda lr: (torch.exp(lr) - 1) / lr],
        )

    def step(
        self,
        u_hat,
    ):
        return setdrk1_step(
            u_hat,
            self._exp_term,
            self._nonlinear_func,
            self._coef_1,
        )


class UnCachedSETDRK2(_UnCachedSETDRKBase):
    """
    Second-order UnCachedSETDRK method.
    """

    def __init__(
        self,
        dt: float,
        linear_coef: torch.Tensor,
        nonlinear_func: Callable[[torch.Tensor], torch.Tensor],
        n_integration_points: int = 16,
        integration_radius: float = 1.0,
        integration_chunk_size: Optional[int] = 1,
    ):
        super().__init__(
            dt,
            linear_coef,
            nonlinear_func,
            n_integration_points,
            integration_radius,
            integration_chunk_size,
        )
        self._coef_1, self._coef_2 = self._coef_means(
            linear_coef,
            [
                lambda lr: (torch.exp(lr) - 1) / lr,
                lambda lr: (torch.exp(lr) - 1 - lr) / lr**2,
            ],
        )

    def step(
        self,
        u_hat,
    ):
        return setdrk2_step(
            u_hat,
            self._exp_term,
            self._nonlinear_func,
            self._coef_1,
            self._coef_2,
        )


class UnCachedSETDRK3(_UnCachedSETDRKBase):
    """
    Third-order UnCachedSETDRK method.
    """

    def __init__(
        self,
        dt: float,
        linear_coef: torch.Tensor,
        nonlinear_func: Callable[[torch.Tensor], torch.Tensor],
        n_integration_points: int = 16,
        integration_radius: float = 1.0,
        integration_chunk_size: Optional[int] = 1,
    ):
        super().__init__(
            dt,
            linear_coef,
            nonlinear_func,
            n_integration_points,
            integration_radius,
            integration_chunk_size,
        )
        self._half_exp_term = torch.exp(0.5 * dt * linear_coef)
        (
            self._coef_1,
            self._coef_2,
            self._coef_3,
            self._coef_4,
            self._coef_5,
        ) = self._coef_means(
            linear_coef,
            [
                lambda lr: (torch.exp(lr / 2) - 1) / lr,
                lambda lr: (torch.exp(lr) - 1) / lr,
                lambda lr: (-4 - lr + torch.exp(lr) * (4 - 3 * lr + lr**2)) / (lr**3),
                lambda lr: (4.0 * (2.0 + lr + torch.exp(lr) * (-2 + lr))) / (lr**3),
                lambda lr: (-4 - 3 * lr - lr**2 + torch.exp(lr) * (4 - lr)) / (lr**3),
            ],
        )

    def step(
        self,
        u_hat,
    ):
        return setdrk3_step(
            u_hat,
            self._exp_term,
            self._half_exp_term,
            self._nonlinear_func,
            self._coef_1,
            self._coef_2,
            self._coef_3,
            self._coef_4,
            self._coef_5,
        )


class UnCachedSETDRK4(_UnCachedSETDRKBase):
    """
    Fourth-order UnCachedSETDRK method.
    """

    def __init__(
        self,
        dt: float,
        linear_coef: torch.Tensor,
        nonlinear_func: Callable[[torch.Tensor], torch.Tensor],
        n_integration_points: int = 16,
        integration_radius: float = 1.0,
        integration_chunk_size: Optional[int] = 1,
    ):
        super().__init__(
            dt,
            linear_coef,
            nonlinear_func,
            n_integration_points,
            integration_radius,
            integration_chunk_size,
        )
        self._half_exp_term = torch.exp(0.5 * dt * linear_coef)
        (
            self._coef_1,
            self._coef_4,
            self._coef_5,
            self._coef_6,
        ) = self._coef_means(
            linear_coef,
            [
                lambda lr: (torch.exp(lr / 2) - 1) / lr,
                lambda lr: (-4 - lr + torch.exp(lr) * (4 - 3 * lr + lr**2)) / (lr**3),
                lambda lr: (2 + lr + torch.exp(lr) * (-2 + lr)) / (lr**3),
                lambda lr: (-4 - 3 * lr - lr**2 + torch.exp(lr) * (4 - lr)) / (lr**3),
            ],
        )
        self._coef_2 = self._coef_1
        self._coef_3 = self._coef_1

    def step(
        self,
        u_hat,
    ):
        return setdrk4_step(
            u_hat,
            self._exp_term,
            self._half_exp_term,
            self._nonlinear_func,
            self._coef_1,
            self._coef_2,
            self._coef_3,
            self._coef_4,
            self._coef_5,
            self._coef_6,
        )
