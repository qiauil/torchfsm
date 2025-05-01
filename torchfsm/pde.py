from .operator import (
    Operator,
    Convection,
    Laplacian,
    Biharmonic,
    KSConvection,
    SpatialDerivative,
    VorticityConvection,
    NSPressureConvection
)
from typing import Optional

def Burgers(nu: float) -> Operator:
    r"""
    Burgers equation:
        $$\frac{\partial \mathbf{u}}{\partial t} =-\mathbf{u} \cdot \nabla \mathbf{u} + \nu \nabla^2 \mathbf{u}$$

    Args:
        nu (float): Viscosity coefficient.    
        
    Returns:
        Operator: The operator representing the Burgers equation.

    """
    return nu * Laplacian() - Convection()

def KuramotoSivashinsky() -> Operator:
    r"""
    1D Kuramoto-Sivashinsky equation:
        $$\frac{\partial \phi}{\partial t}=-\frac{\partial^2 \phi}{\partial x^2} -\frac{\partial^4 \phi}{\partial x^4} - \phi\frac{\partial\phi}{\partial x}$$
    
    Returns:
        Operator: The operator representing the Kuramoto-Sivashinsky equation.
    """
    ks_eqn = -Laplacian() - Biharmonic() - Convection()
    ks_eqn.regisiter_additional_check(
        lambda dim_value, dim_mesh: dim_value == 1 and dim_mesh == 1
    )
    return ks_eqn

def KuramotoSivashinskyHighDim() -> Operator:
    r"""
    High dimensional Kuramoto-Sivashinsky equation:
        $$\frac{\partial \mathbf{\phi}}{\partial t}=-\nabla^2 \phi- \nabla^4 \phi - \frac{1}{2}|\nabla \phi|^2$$
    
    Returns:
        Operator: The operator representing the Kuramoto-Sivashinsky equation.
    """
    return -Laplacian() - Biharmonic() - KSConvection()

def KortewegDeVries(dispersion_coef=1, convection_coef: float = 6.0) -> Operator:
    r"""
    Korteweg-De Vries equation:
        $$\frac{\partial \phi}{\partial t}=-c_1\frac{\partial^3 \phi}{\partial x^3} + c_2 \phi\frac{\partial\phi}{\partial x}$$

    Args:
        dispersion_coef (float): Dispersion coefficient. Default is 1.
        convection_coef (float): Convection coefficient. Default is 6.0.

    Returns:
        Operator: The operator representing the Korteweg-De Vries equation.    

    """
    return -dispersion_coef * SpatialDerivative(0, 3) + convection_coef * Convection()

def NavierStokesVorticity(Re:float,force:Optional[Operator]=None)->Operator:
    r"""
    Navier-Stokes equation in vorticity form:
        $$\frac{\partial \omega}{\partial t} + (\mathbf{u}\cdot\nabla) \omega = \frac{1}{Re} \nabla^2 \omega + \nabla \times \mathbf{f}$$
    
    Args:
        Re (float): Reynolds number.
        force (Optional[Operator]): Optional external force term. Default is None.
            If provided, it will be added to the vorticity equation. Note that the provided force should be $\nabla \times \mathbf{f}$ rather than $\mathbf{f}$ itself.

    Returns:
        Operator: The operator representing the Navier-Stokes equation in vorticity form.

    """
    ns_vorticity=-VorticityConvection() + 1/Re*Laplacian()
    if force is not None:
        ns_vorticity+=force
    return ns_vorticity

def NavierStokes(Re:float,force:Optional[Operator]=None)->Operator:
    r"""
    Navier-Stokes equation:
        $$\frac{\partial\mathbf{u}}{\partial t}=-\nabla (\nabla^{-2} \nabla \cdot (\left(\mathbf{u}\cdot\nabla\right)\mathbf{u}-f))-\left(\mathbf{u}\cdot\nabla\right)\mathbf{u}+\nu \nabla^2 \mathbf{u} + \mathbf{f}$$
    
    Args:
        Re (float): Reynolds number.
        force (Optional[Operator]): Optional external force term. Default is None.
            If provided, it will be added to the vorticity equation. 

    Returns:
        Operator: The operator representing the Navier-Stokes equation.

    """

    return NSPressureConvection(force)+1/Re*Laplacian()