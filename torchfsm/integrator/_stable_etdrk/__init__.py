"""
Any of the ETDRK schemes of order higher than two suffers from numerical instability.
To perform a stable higher-order ETD scheme, we need special modifications to the ETDRK method.
Theoretical details can be found at:
[1] Kassam, Aly-Khan, and Lloyd N. Trefethen. "Fourth-order time-stepping for stiff PDEs." SIAM Journal on Scientific Computing 26.4 (2005): 1214-1233.


Code modified from https://github.com/Ceyron/exponax/tree/main/exponax/etdrk
We would like to thank Felix Koehler (https://github.com/Ceyron) for his contribution on the original code and help during the implementation.
"""

import torch
from enum import Enum
from typing import Callable, Union
from ._cached import CachedSETDRK1, CachedSETDRK2, CachedSETDRK3, CachedSETDRK4
from ._uncached import (
    UnCachedSETDRK1,
    UnCachedSETDRK2,
    UnCachedSETDRK3,
    UnCachedSETDRK4,
)

class SETDRK1Wrapper:
    
    def __call__(self,
                 dt: float,
                linear_coef: torch.Tensor,
                nonlinear_func: Callable[[torch.Tensor], torch.Tensor],
                n_integration_points: int = 16,
                integration_radius: float = 1.0,
                cpu_cached: bool = False,):
        if cpu_cached:
            return CachedSETDRK1(
                dt, linear_coef, nonlinear_func, n_integration_points, integration_radius
            )
        else:
            return UnCachedSETDRK1(
                dt, linear_coef, nonlinear_func, n_integration_points, integration_radius
            )

class SETDRK2Wrapper:
    
    def __call__(self,
                 dt: float,
                linear_coef: torch.Tensor,
                nonlinear_func: Callable[[torch.Tensor], torch.Tensor],
                n_integration_points: int = 16,
                integration_radius: float = 1.0,
                cpu_cached: bool = False,):
        if cpu_cached:
            return CachedSETDRK2(
                dt, linear_coef, nonlinear_func, n_integration_points, integration_radius
            )
        else:
            return UnCachedSETDRK2(
                dt, linear_coef, nonlinear_func, n_integration_points, integration_radius
            )

class SETDRK3Wrapper:
    
    def __call__(self,
                 dt: float,
                linear_coef: torch.Tensor,
                nonlinear_func: Callable[[torch.Tensor], torch.Tensor],
                n_integration_points: int = 16,
                integration_radius: float = 1.0,
                cpu_cached: bool = False,):
        if cpu_cached:
            return CachedSETDRK3(
                dt, linear_coef, nonlinear_func, n_integration_points, integration_radius
            )
        else:
            return UnCachedSETDRK3(
                dt, linear_coef, nonlinear_func, n_integration_points, integration_radius
            )

class SETDRK4Wrapper:
    
    def __call__(self,
                 dt: float,
                linear_coef: torch.Tensor,
                nonlinear_func: Callable[[torch.Tensor], torch.Tensor],
                n_integration_points: int = 16,
                integration_radius: float = 1.0,
                cpu_cached: bool = False,):
        if cpu_cached:
            return CachedSETDRK4(
                dt, linear_coef, nonlinear_func, n_integration_points, integration_radius
            )
        else:
            return UnCachedSETDRK4(
                dt, linear_coef, nonlinear_func, n_integration_points, integration_radius
            )


class SETDRKIntegrator(Enum):
    """
    Stable ETDRK Integrator
    Provides a unified interface for all ETDRK methods.
    """

    SETDRK1 = SETDRK1Wrapper()
    SETDRK2 = SETDRK2Wrapper()
    SETDRK3 = SETDRK3Wrapper()
    SETDRK4 = SETDRK4Wrapper()
