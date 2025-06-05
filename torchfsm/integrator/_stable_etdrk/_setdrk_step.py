import torch
from typing import Callable


def setdrk1_step(
    u_hat: torch.Tensor,
    exp_term: torch.Tensor,
    nonlinear_func: Callable[[torch.Tensor], torch.Tensor],
    coef_1: torch.Tensor,
):
    return exp_term * u_hat + coef_1 * nonlinear_func(u_hat)


def setdrk2_step(
    u_hat: torch.Tensor,
    exp_term: torch.Tensor,
    nonlinear_func: Callable[[torch.Tensor], torch.Tensor],
    coef_1: torch.Tensor,
    coef_2: torch.Tensor,
):
    u_nonlin_hat = nonlinear_func(u_hat)
    u_stage_1_hat = exp_term * u_hat + coef_1 * u_nonlin_hat
    u_stage_1_nonlin_hat = nonlinear_func(u_stage_1_hat)
    u_next_hat = u_stage_1_hat + coef_2 * (u_stage_1_nonlin_hat - u_nonlin_hat)
    return u_next_hat


def setdrk3_step(
    u_hat: torch.Tensor,
    exp_term: torch.Tensor,
    half_exp_term: torch.Tensor,
    nonlinear_func: Callable[[torch.Tensor], torch.Tensor],
    coef_1: torch.Tensor,
    coef_2: torch.Tensor,
    coef_3: torch.Tensor,
    coef_4: torch.Tensor,
    coef_5: torch.Tensor,
):
    u_nonlin_hat = nonlinear_func(u_hat)
    u_stage_1_hat = half_exp_term * u_hat + coef_1 * u_nonlin_hat
    u_stage_1_nonlin_hat = nonlinear_func(u_stage_1_hat)
    u_stage_2_hat = exp_term * u_hat + coef_2 * (
        2 * u_stage_1_nonlin_hat - u_nonlin_hat
    )
    u_stage_2_nonlin_hat = nonlinear_func(u_stage_2_hat)
    u_next_hat = (
        exp_term * u_hat
        + coef_3 * u_nonlin_hat
        + coef_4 * u_stage_1_nonlin_hat
        + coef_5 * u_stage_2_nonlin_hat
    )
    return u_next_hat


def setdrk4_step(
    u_hat: torch.Tensor,
    exp_term: torch.Tensor,
    half_exp_term: torch.Tensor,
    nonlinear_func: Callable[[torch.Tensor], torch.Tensor],
    coef_1: torch.Tensor,
    coef_2: torch.Tensor,
    coef_3: torch.Tensor,
    coef_4: torch.Tensor,
    coef_5: torch.Tensor,
    coef_6: torch.Tensor,
):
    u_nonlin_hat = nonlinear_func(u_hat)
    u_stage_1_hat = half_exp_term * u_hat + coef_1 * u_nonlin_hat
    u_stage_1_nonlin_hat = nonlinear_func(u_stage_1_hat)
    u_stage_2_hat = half_exp_term * u_hat + coef_2 * u_stage_1_nonlin_hat
    u_stage_2_nonlin_hat = nonlinear_func(u_stage_2_hat)
    u_stage_3_hat = half_exp_term * u_stage_1_hat + coef_3 * (
        2 * u_stage_2_nonlin_hat - u_nonlin_hat
    )
    u_stage_3_nonlin_hat = nonlinear_func(u_stage_3_hat)
    u_next_hat = (
        exp_term * u_hat
        + coef_4 * u_nonlin_hat
        + coef_5 * 2 * (u_stage_1_nonlin_hat + u_stage_2_nonlin_hat)
        + coef_6 * u_stage_3_nonlin_hat
    )
    return u_next_hat
