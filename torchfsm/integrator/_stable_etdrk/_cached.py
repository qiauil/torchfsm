import torch
from abc import ABC, abstractmethod
from typing import Callable
from enum import Enum
from ._uncached import roots_of_unity
from ._setdrk_step import setdrk1_step, setdrk2_step, setdrk3_step, setdrk4_step


class _CachedSETDRKBase(ABC):

    def __init__(
        self,
        dt: float,
        linear_coef: torch.Tensor,
        nonlinear_func: Callable[[torch.Tensor], torch.Tensor],
        n_integration_points: int = 16,
        integration_radius: float = 1.0,
    ):
        self.dt = dt
        self._nonlinear_func = nonlinear_func
        self._exp_term = torch.exp(self.dt * linear_coef)
        self.n_integration_points = n_integration_points
        self.integration_radius = integration_radius

    def _get_lr(self, linear_coef):
        return (
            self.integration_radius
            * roots_of_unity(
                self.n_integration_points, device="cpu", dtype=linear_coef.real.dtype
            )
            + linear_coef.unsqueeze(-1).cpu() * self.dt
        )

    @abstractmethod
    def step(
        self,
        u_hat,
    ):
        """
        Advance the state in Fourier space.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")


class CachedSETDRK1(_CachedSETDRKBase):
    """
    First-order CachedSETDRK method.
    """

    def __init__(
        self,
        dt: float,
        linear_coef: torch.Tensor,
        nonlinear_func: Callable[[torch.Tensor], torch.Tensor],
        n_integration_points: int = 16,
        integration_radius: float = 1.0,
    ):
        super().__init__(
            dt, linear_coef, nonlinear_func, n_integration_points, integration_radius
        )
        lr = self._get_lr(linear_coef)
        self._coef_1 = dt * torch.mean((torch.exp(lr) - 1) / lr, axis=-1).real.to(
            linear_coef.device
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


class CachedSETDRK2(_CachedSETDRKBase):
    """
    Second-order CachedSETDRK method.
    """

    def __init__(
        self,
        dt: float,
        linear_coef: torch.Tensor,
        nonlinear_func: Callable[[torch.Tensor], torch.Tensor],
        n_integration_points: int = 16,
        integration_radius: float = 1.0,
    ):
        super().__init__(
            dt, linear_coef, nonlinear_func, n_integration_points, integration_radius
        )
        lr = self._get_lr(linear_coef)
        self._coef_1 = dt * torch.mean((torch.exp(lr) - 1) / lr, axis=-1).real.to(
            linear_coef.device
        )
        self._coef_2 = dt * torch.mean(
            (torch.exp(lr) - 1 - lr) / lr**2, axis=-1
        ).real.to(linear_coef.device)

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


class CachedSETDRK3(_CachedSETDRKBase):
    """
    Third-order CachedSETDRK method.
    """

    def __init__(
        self,
        dt: float,
        linear_coef: torch.Tensor,
        nonlinear_func: Callable[[torch.Tensor], torch.Tensor],
        n_integration_points: int = 16,
        integration_radius: float = 1.0,
    ):
        super().__init__(
            dt, linear_coef, nonlinear_func, n_integration_points, integration_radius
        )
        lr = self._get_lr(linear_coef)
        self._half_exp_term = torch.exp(0.5 * dt * linear_coef)
        self._coef_1 = dt * torch.mean((torch.exp(lr / 2) - 1) / lr, axis=-1).real.to(
            linear_coef.device
        )
        self._coef_2 = dt * torch.mean((torch.exp(lr) - 1) / lr, axis=-1).real.to(
            linear_coef.device
        )
        self._coef_3 = dt * torch.mean(
            (-4 - lr + torch.exp(lr) * (4 - 3 * lr + lr**2)) / (lr**3), axis=-1
        ).real.to(linear_coef.device)
        self._coef_4 = dt * torch.mean(
            (4.0 * (2.0 + lr + torch.exp(lr) * (-2 + lr))) / (lr**3), axis=-1
        ).real.to(linear_coef.device)
        self._coef_5 = dt * torch.mean(
            (-4 - 3 * lr - lr**2 + torch.exp(lr) * (4 - lr)) / (lr**3), axis=-1
        ).real.to(linear_coef.device)

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


class CachedSETDRK4(_CachedSETDRKBase):
    """
    Fourth-order CachedSETDRK method.
    """

    def __init__(
        self,
        dt: float,
        linear_coef: torch.Tensor,
        nonlinear_func: Callable[[torch.Tensor], torch.Tensor],
        n_integration_points: int = 16,
        integration_radius: float = 1.0,
    ):
        super().__init__(
            dt, linear_coef, nonlinear_func, n_integration_points, integration_radius
        )
        lr = self._get_lr(linear_coef)
        self._half_exp_term = torch.exp(0.5 * dt * linear_coef)
        self._coef_1 = dt * torch.mean((torch.exp(lr / 2) - 1) / lr, axis=-1).real.to(
            linear_coef.device
        )
        self._coef_2 = self._coef_1
        self._coef_3 = self._coef_1
        self._coef_4 = dt * torch.mean(
            (-4 - lr + torch.exp(lr) * (4 - 3 * lr + lr**2)) / (lr**3), axis=-1
        ).real.to(linear_coef.device)
        self._coef_5 = dt * torch.mean(
            (2 + lr + torch.exp(lr) * (-2 + lr)) / (lr**3), axis=-1
        ).real.to(linear_coef.device)
        self._coef_6 = dt * torch.mean(
            (-4 - 3 * lr - lr**2 + torch.exp(lr) * (4 - lr)) / (lr**3), axis=-1
        ).real.to(linear_coef.device)

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
