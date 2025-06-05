"""
[1] Cox, Steven M., and Paul C. Matthews. "Exponential time differencing for stiff systems." Journal of Computational Physics 176.2 (2002): 430-455.
"""

import torch
from typing import Callable
from enum import Enum


class ETDRK0:
    """
    Exactly solve a linear PDE in Fourier space
    """

    def __init__(
        self,
        dt: float,
        linear_coef: torch.Tensor,
    ):
        self.dt = dt
        self._exp_term = torch.exp(self.dt * linear_coef)

    def step(
        self,
        u_hat,
    ):
        return self._exp_term * u_hat


class ETDRK1(ETDRK0):
    """
    First-order ETDRK method.
    """

    def __init__(
        self,
        dt: float,
        linear_coef: torch.Tensor,
        nonlinear_func: Callable[[torch.Tensor], torch.Tensor],
    ):
        super().__init__(dt, linear_coef)
        self._nonlinear_func = nonlinear_func
        self._coef_1 = torch.where(
            linear_coef == 0, self.dt, (self._exp_term - 1) / linear_coef
        )

    def step(
        self,
        u_hat,
    ):
        return self._exp_term * u_hat + self._coef_1 * self._nonlinear_func(u_hat)


class ETDRK2(ETDRK1):
    """
    Second-order ETDRK method.
    """

    def __init__(
        self,
        dt: float,
        linear_coef: torch.Tensor,
        nonlinear_func: Callable[[torch.Tensor], torch.Tensor],
    ):
        super().__init__(dt, linear_coef, nonlinear_func)
        self._coef_2 = torch.where(
            linear_coef == 0,
            self.dt / 2,
            (self._exp_term - 1 - linear_coef * self.dt) / (linear_coef**2 * self.dt),
        )

    def step(
        self,
        u_hat,
    ):
        u_nonlin_hat = self._nonlinear_func(u_hat)
        u_stage_1_hat = self._exp_term * u_hat + self._coef_1 * u_nonlin_hat
        u_stage_1_nonlin_hat = self._nonlinear_func(u_stage_1_hat)
        u_next_hat = u_stage_1_hat + self._coef_2 * (
            u_stage_1_nonlin_hat - u_nonlin_hat
        )
        return u_next_hat


class ETDRKIntegrator(Enum):
    """
    ETDRK Integrator
    Provides a unified interface for all ETDRK methods.
    """

    ETDRK0 = ETDRK0
    ETDRK1 = ETDRK1
    ETDRK2 = ETDRK2
