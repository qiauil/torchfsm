from ..operator import  Operator, ImplicitSource, ExplicitSource
import torch

def kolm_force(
    x: torch.Tensor,
    drag_coef: float = -0.1,
    k: float = 4.0,
    length_scale: float = 1.0,
) -> Operator:
    r"""
    Generate cosine force field for 2d kolmogorov flow in vorticity form.
        It is defined as $a \omega - k cos (k l x)$

    Args:
        x (torch.Tensor): The input tensor.
        drag_coef (float): The drag coefficient $a$. Default is -0.1.
        k (float): The wave number. Default is 4.0.
        length_scale (float): The length scale $l$. Default is 1.0.
    """


    return drag_coef * ImplicitSource() - ExplicitSource(
        k * torch.cos(k * length_scale * x)
    )