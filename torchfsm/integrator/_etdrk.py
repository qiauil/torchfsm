"""
Modified from https://github.com/Ceyron/exponax/tree/main/exponax/etdrk

We would like to thank Felix Koehler (https://github.com/Ceyron) for his contribution on the original code and help during the implementation.
"""

import torch
from abc import ABC, abstractmethod
from typing import Callable
from enum import Enum

def roots_of_unity(M: int,device=None,dtype=None) -> torch.Tensor:
    """
    Return (complex-valued) array with M roots of unity.
    """
    # return torch.exp(1j * torch.pi * (torch.arange(1, M+1) - 0.5) / M)
    return torch.exp(2j * torch.pi * (torch.arange(1, M + 1,device=device,dtype=dtype) - 0.5) / M)

class _ETDRKBase(ABC):
    
    """
    Solve
    $$
    u_t=Lu+N(u)
    $$
    using the ETDRK method.
    
    Args:
        dt(float): Time step.
        linear_coef(torch.Tensor): Coefficient of the linear term, i.e., $L$.
        nonlinear_func(Callable[[torch.Tensor],torch.Tensor]): Function that computes the nonlinear term, i.e., $N(u)$.
        num_circle_points(int): Number of points on the unit circle. See [2] for details.
        circle_radius(float): Radius of the unit circle. See [2] for details.
    
    Reference:
        [1] Cox, Steven M., and Paul C. Matthews. "Exponential time differencing for stiff systems." Journal of Computational Physics 176.2 (2002): 430-455.
        [2] Kassam, Aly-Khan, and Lloyd N. Trefethen. "Fourth-order time-stepping for stiff PDEs." SIAM Journal on Scientific Computing 26.4 (2005): 1214-1233.
    """

    def __init__(
        self,
        dt: float,
        linear_coef:torch.Tensor,
        nonlinear_func:Callable[[torch.Tensor],torch.Tensor],
        num_circle_points: int = 16,
        circle_radius: float = 1.0,
        ):
        self.dt = dt
        self._nonlinear_func = nonlinear_func
        self._exp_term = torch.exp(self.dt * linear_coef)
        self.LR=(
            circle_radius * roots_of_unity(num_circle_points,device=linear_coef.device,dtype=linear_coef.real.dtype)
            + linear_coef.unsqueeze(-1) * dt
            )

    @abstractmethod
    def step(
        self,
        u_hat,
        ):
        """
        Advance the state in Fourier space.
        """
        
class ETDRK0():
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

class ETDRK1(_ETDRKBase):

    """
    First-order ETDRK method.
    """

    def __init__(
        self,
        dt: float,
        linear_coef: torch.Tensor,
        nonlinear_func:Callable[[torch.Tensor],torch.Tensor],
        num_circle_points: int = 16,
        circle_radius: float = 1.0,
        ):
        super().__init__(dt, linear_coef,nonlinear_func,num_circle_points,circle_radius)
        self._coef_1 = dt * torch.mean((torch.exp(self.LR) - 1) / self.LR, axis=-1).real

    def step(
        self,
        u_hat,
        ):
        return self._exp_term * u_hat + self._coef_1 * self._nonlinear_func(u_hat)

class ETDRK2(_ETDRKBase):
    """
    Second-order ETDRK method.
    """

    def __init__(
        self,
        dt: float,
        linear_coef: torch.Tensor,
        nonlinear_func:Callable[[torch.Tensor],torch.Tensor],
        num_circle_points: int = 16,
        circle_radius: float = 1.0,
        ):
        super().__init__(dt, linear_coef,nonlinear_func,num_circle_points,circle_radius)
        self._coef_1 = dt * torch.mean((torch.exp(self.LR) - 1) / self.LR, axis=-1).real
        self._coef_2 = dt * torch.mean((torch.exp(self.LR) - 1 - self.LR) / self.LR**2, axis=-1).real

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

class ETDRK3(_ETDRKBase):
    """
    Third-order ETDRK method.
    """
    def __init__(
        self,
        dt: float,
        linear_coef: torch.Tensor,
        nonlinear_func:Callable[[torch.Tensor],torch.Tensor],
        num_circle_points: int = 16,
        circle_radius: float = 1.0,
    ):
        super().__init__(dt, linear_coef,nonlinear_func,num_circle_points,circle_radius)
        self._half_exp_term = torch.exp(0.5 * dt * linear_coef)
        self._coef_1 = dt * torch.mean((torch.exp(self.LR / 2) - 1) / self.LR, axis=-1).real
        self._coef_2 = dt * torch.mean((torch.exp(self.LR) - 1) / self.LR, axis=-1).real
        self._coef_3 = (
            dt
            * torch.mean(
                (-4 - self.LR + torch.exp(self.LR) * (4 - 3 * self.LR + self.LR**2)) / (self.LR**3), axis=-1
            ).real
        )
        self._coef_4 = (
            dt
            * torch.mean(
                (4.0 * (2.0 + self.LR + torch.exp(self.LR) * (-2 + self.LR))) / (self.LR**3), axis=-1
            ).real
        )
        self._coef_5 = (
            dt
            * torch.mean(
                (-4 - 3 * self.LR - self.LR**2 + torch.exp(self.LR) * (4 - self.LR)) / (self.LR**3), axis=-1
            ).real
        )

    def step(
        self,
        u_hat,
        ):
        u_nonlin_hat = self._nonlinear_func(u_hat)
        u_stage_1_hat = self._half_exp_term * u_hat + self._coef_1 * u_nonlin_hat
        u_stage_1_nonlin_hat = self._nonlinear_func(u_stage_1_hat)
        u_stage_2_hat = self._exp_term * u_hat + self._coef_2 * (
            2 * u_stage_1_nonlin_hat - u_nonlin_hat
            )
        u_stage_2_nonlin_hat = self._nonlinear_func(u_stage_2_hat)
        u_next_hat = (
            self._exp_term * u_hat
            + self._coef_3 * u_nonlin_hat
            + self._coef_4 * u_stage_1_nonlin_hat
            + self._coef_5 * u_stage_2_nonlin_hat
            )
        return u_next_hat
    
class ETDRK4(_ETDRKBase):
    """
    Fourth-order ETDRK method.
    """

    def __init__(
        self,
        dt: float,
        linear_coef: torch.Tensor,
        nonlinear_func:Callable[[torch.Tensor],torch.Tensor],
        num_circle_points: int = 16,
        circle_radius: float = 1.0,
    ):
        super().__init__(dt, linear_coef,nonlinear_func,num_circle_points,circle_radius)
        self._half_exp_term = torch.exp(0.5 * dt * linear_coef)
        self._coef_1 = dt * torch.mean((torch.exp(self.LR / 2) - 1) / self.LR, axis=-1).real
        self._coef_2 = self._coef_1
        self._coef_3 = self._coef_1
        self._coef_4 = (
            dt
            * torch.mean(
                (-4 - self.LR + torch.exp(self.LR) * (4 - 3 * self.LR + self.LR**2)) / (self.LR**3), axis=-1
            ).real
        )
        self._coef_5 = (
            dt * torch.mean((2 + self.LR + torch.exp(self.LR) * (-2 + self.LR)) / (self.LR**3), axis=-1).real
        )
        self._coef_6 = (
            dt
            * torch.mean(
                (-4 - 3 * self.LR - self.LR**2 + torch.exp(self.LR) * (4 - self.LR)) / (self.LR**3), axis=-1
            ).real
        )
        
    def step(
        self,
        u_hat,
        ):
        u_nonlin_hat = self._nonlinear_func(u_hat)
        u_stage_1_hat = self._half_exp_term * u_hat + self._coef_1 * u_nonlin_hat
        u_stage_1_nonlin_hat = self._nonlinear_func(u_stage_1_hat)
        u_stage_2_hat = (
            self._half_exp_term * u_hat + self._coef_2 * u_stage_1_nonlin_hat
            )
        u_stage_2_nonlin_hat = self._nonlinear_func(u_stage_2_hat)
        u_stage_3_hat = self._half_exp_term * u_stage_1_hat + self._coef_3 * (
            2 * u_stage_2_nonlin_hat - u_nonlin_hat
            )
        u_stage_3_nonlin_hat = self._nonlinear_func(u_stage_3_hat)
        u_next_hat = (
            self._exp_term * u_hat
            + self._coef_4 * u_nonlin_hat
            + self._coef_5 * 2 * (u_stage_1_nonlin_hat + u_stage_2_nonlin_hat)
            + self._coef_6 * u_stage_3_nonlin_hat
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
    ETDRK3 = ETDRK3
    ETDRK4 = ETDRK4