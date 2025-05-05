from typing import Callable, Optional, Sequence, Union
from torch import Tensor
import torch
from enum import Enum


class _RKBase:
    """
    Base class for Runge-Kutta integrators.
        This class implements the Runge-Kutta method for solving ordinary differential equations (ODEs).
        The Runge-Kutta method is a numerical technique used to solve ODEs by approximating the solution at discrete time steps.
        The class provides a flexible interface for defining different Runge-Kutta methods by specifying the coefficients and weights.
        The class also supports adaptive step size control, allowing for dynamic adjustment of the time step based on the estimated error.

    Args:
        ca (Sequence[float]): Coefficients for the Runge-Kutta method.
        b (Sequence[float]): Weights for the Runge-Kutta method.
        b_star (Optional[Sequence]): Optional coefficients for error estimation.
        adaptive (bool): If True, enables adaptive step size control.
        atol (float): Absolute tolerance for adaptive step size control.
        rtol (float): Relative tolerance for adaptive step size control.
    """

    def __init__(
        self,
        ca: Sequence[float],
        b: Sequence[float],
        b_star: Optional[Sequence] = None,
        adaptive: bool = False,
        atol: float = 1e-6,
        rtol: float = 1e-5,
    ):
        self.ca = ca
        self.b = b
        self.b_star = b_star
        self.adaptive = adaptive
        self.atol = atol
        self.rtol = rtol
        if self.adaptive and self.b_star is None:
            raise ValueError("adaptive step requires b_star")
        self.step= self._adaptive_step if self.adaptive else self._rk_step

    def _rk_step(
        self,
        f: Callable[[Tensor], Tensor],
        x_t: Tensor,
        dt: float,
        return_error: bool = False,
    ):
        ks = [f(x_t)]
        for ca_i in self.ca:
            ks.append(f(x_t + dt * sum([a_i * k for a_i, k in zip(ca_i[1:], ks)])))
        x_new = x_t + dt * sum([b_i * k for b_i, k in zip(self.b, ks)])
        if self.b_star is not None and return_error:
            b_dif = [b_i - b_star_i for b_i, b_star_i in zip(self.b, self.b_star)]
            error = dt * sum([b_dif_i * k for b_dif_i, k in zip(b_dif, ks)])
            return x_new, error
        return x_new

    def _adaptive_step(
            self,
            f: Callable[[Tensor], Tensor],
            x_t: Tensor,
            dt: float,
        ):
        t = 0.0
        t_1 = dt - t
        while t_1 - t > 0:
            dt = min(dt, abs(t_1 - t))
            if self.adaptive:
                while True:
                    y, error = self._rk_step(f, x_t, dt, return_error=True)
                    tolerance = self.atol + self.rtol * torch.max(abs(x_t), abs(y))
                    error = torch.max(error / tolerance).clip(min=1e-9).item()
                    if error < 1.0:
                        x_t, t = y, t + dt
                        break
                    dt = dt * min(10.0, max(0.1, 0.9 / error ** (1 / 5)))
        return x_t


class Euler(_RKBase):
    """
    First-order Euler method.
    """
    def __init__(self):
        super().__init__([[1.0]], [1.0], adaptive=False)


class Midpoint(_RKBase):
    """
    Midpoint method.
    """
    def __init__(self):
        super().__init__([[1 / 2, 1 / 2]], [0, 1], adaptive=False)


class Heun12(_RKBase):
    """
    Heun's second-order method.
    """
    def __init__(self, adaptive: bool = False, atol: float = 1e-6, rtol: float = 1e-5):
        super().__init__(
            [[1, 1]], [1 / 2, 1 / 2], [1, 0], adaptive=adaptive, atol=atol, rtol=rtol
        )


class Ralston12(_RKBase):
    """
    Ralston's second-order method.
    """
    def __init__(self, adaptive: bool = False, atol: float = 1e-6, rtol: float = 1e-5):
        super().__init__(
            [[2 / 3, 2 / 3]],
            [1 / 4, 3 / 4],
            [2 / 3, 1 / 3],
            adaptive=adaptive,
            atol=atol,
            rtol=rtol,
        )


class BogackiShampine23(_RKBase):
    """
    Third-order Bogack and Shampine method.
    """
    def __init__(self, adaptive: bool = False, atol: float = 1e-6, rtol: float = 1e-5):
        super().__init__(
            ca=[
                [1 / 2, 1 / 2],
                [3 / 4, 0, 3 / 4],
                [1, 2 / 9, 1 / 3, 4 / 9],
            ],
            b=[2 / 9, 1 / 3, 4 / 9, 0],
            b_star=[7 / 24, 1 / 4, 1 / 3, 1 / 8],
            adaptive=adaptive,
            atol=atol,
            rtol=rtol,
        )


class RK4(_RKBase):
    """
    Fourth-order Runge-Kutta method.
    """
    def __init__(self):
        super().__init__(
            ca=[
                [1 / 2, 1 / 2],
                [1 / 2, 0, 1 / 2],
                [1, 0, 0, 1],
            ],
            b=[1 / 6, 1 / 3, 1 / 3, 1 / 6],
            adaptive=False,
        )


class RK4_38Rule(_RKBase):
    """
    Fourth-order Runge-Kutta method with 3/8 rule.
    """
    def __init__(self):
        super().__init__(
            ca=[
                [1 / 3, 1 / 3],
                [2 / 3, -1 / 3, 1],
                [1, -1, 1, 1],
            ],
            b=[1 / 8, 3 / 8, 3 / 8, 1 / 8],
            adaptive=False,
        )


class Dorpi45(_RKBase):
    """
    Fifth-order Dormand-Prince method.
    """

    def __init__(self, adaptive: bool = False, atol: float = 1e-6, rtol: float = 1e-5):
        super().__init__(
            ca=[
                [1 / 5, 1 / 5],
                [3 / 10, 3 / 40, 9 / 40],
                [4 / 5, 44 / 45, -56 / 15, 32 / 9],
                [8 / 9, 19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
                [1, 9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
                [1, 35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84],
            ],
            b=[35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84],
            b_star=[
                5179 / 57600,
                0,
                7571 / 16695,
                393 / 640,
                -92097 / 339200,
                187 / 2100,
                1 / 40,
            ],
            adaptive=adaptive,
            atol=atol,
            rtol=rtol,
        )


class Fehlberg45(_RKBase):
    """
    Fehlberg 4(5) method for adaptive step size control.
    This method is a Runge-Kutta method that provides a fourth-order and fifth-order approximation of the solution to an ODE.
    """

    def __init__(self, adaptive: bool = False, atol: float = 1e-6, rtol: float = 1e-5):
        super().__init__(
            ca=[
                [1 / 4, 1 / 4],
                [3 / 8, 3 / 32, 9 / 32],
                [12 / 13, 1932 / 2197, -7200 / 2197, 7296 / 2197],
                [1, 439 / 216, -8, 3680 / 513, -845 / 4104],
                [1 / 2, -8 / 27, 2, -3544 / 2565, 1859 / 4104, -11 / 40],
            ],
            b=[16 / 135, 0, 6656 / 12825, 28561 / 56430, -9 / 50, 2 / 55],
            b_star=[25 / 216, 0, 1408 / 2565, 2197 / 4104, -1 / 5, 0],
            adaptive=adaptive,
            atol=atol,
            rtol=rtol,
        )


class CashKarp45(_RKBase):
    """
    Cash-Karp 4(5) method for adaptive step size control.
    This method is a Runge-Kutta method that provides a fourth-order and fifth-order approximation of the solution to an ODE.
    """

    def __init__(self, adaptive: bool = False, atol: float = 1e-6, rtol: float = 1e-5):
        super().__init__(
            ca=[
                [1 / 5, 1 / 5],
                [3 / 10, 3 / 40, 9 / 40],
                [3 / 5, 3 / 10, -9 / 10, 6 / 5],
                [1, -11 / 54, 5 / 2, -70 / 27, 35 / 27],
                [
                    7 / 8,
                    1631 / 55296,
                    175 / 512,
                    575 / 13824,
                    44275 / 110592,
                    253 / 4096,
                ],
            ],
            b=[37 / 378, 0, 250 / 621, 125 / 594, 0, 512 / 1771],
            b_star=[2825 / 27648, 0, 18575 / 48384, 13525 / 55296, 277 / 14336, 1 / 4],
            adaptive=adaptive,
            atol=atol,
            rtol=rtol,
        )

class RKIntegrator(Enum):
    """
    Enum class for Runge-Kutta integrators.
    This class provides a set of predefined Runge-Kutta integrators for solving ordinary differential equations (ODEs).
    Each integrator is represented as a member of the enum, and can be used to create an instance of the corresponding integrator class.
    The integrators include:
        - Euler: First-order Euler method.
        - Midpoint: Second-order Midpoint method.
        - Heun12: Second-order Heun method.
        - Ralston12: Second-order Ralston method.
        - BogackiShampine23: Third-order
        - RK4: Fourth-order Runge-Kutta method.
        - RK4_38Rule: Fourth-order Runge-Kutta method with 3/8 rule.
        - Dorpi45: Fifth-order Dormand-Prince method.
        - Fehlberg45: Fifth-order Fehlberg method.
        - CashKarp45: Fifth-order Cash-Karp method.
    """
    Euler = Euler
    Midpoint = Midpoint
    Heun12 = Heun12
    Ralston12 = Ralston12
    BogackiShampine23 = BogackiShampine23
    RK4 = RK4
    RK4_38Rule = RK4_38Rule
    Dorpi45 = Dorpi45
    Fehlberg45 = Fehlberg45
    CashKarp45 = CashKarp45