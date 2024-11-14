from .operator import Operator, Convection, Laplacian,Biharmonic,KSConvection,SpatialDerivative

def Burgers(nu:float) -> Operator:
    return nu*Laplacian()-Convection()

def KuramotoSivashinsky() -> Operator:
    ks_eqn = -Laplacian()- Biharmonic()-Convection()
    ks_eqn.regisiter_additional_check(lambda dim_value,dim_mesh: dim_value ==1 and dim_mesh ==1)
    return ks_eqn

def KuramotoSivashinskyHighDim() -> Operator:
    return -Laplacian()- Biharmonic()-KSConvection()

def KortewegDeVries(dispersion_coef=1,convection_coef:float=6.0) -> Operator:
    return -dispersion_coef*SpatialDerivative(0,3)+convection_coef*Convection()