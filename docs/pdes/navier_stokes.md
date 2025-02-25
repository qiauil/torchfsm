Navier-Stokes Equations describes the fluid motion. In this content, we basically discuss the incompressible form of the Navier-Stokes equations: 

$$
\begin{align}
\frac{\partial\mathbf{u}}{\partial t}+\left(\mathbf{u}\cdot\nabla\right)\mathbf{u}&=-\frac{1}{\rho}\nabla p+\nu \nabla^2 \mathbf{u} + \frac{1}{\rho}\mathbf{f} ,\tag{1}\\
\nabla\cdot\mathbf{u}&=0.\tag{2}
\end{align}
$$

where $\mathbf{u}$ is the velocity field, $p$ is the pressure field, $\rho$ is the fluid density, $\nu$ is the kinematic viscosity, and $\mathbf{f}$ is the external force field. The first equation is the momentum equation and the second equation is the continuity equation. 

Solving the Navier-Stokes equations is a challenging task due to the non-linearity of the equations. The non-linearity arises from the convective term $\left(\mathbf{u}\cdot\nabla\right)\mathbf{u}$. The non-linearity makes the equations difficult to solve analytically. However, the equations can be solved numerically using various methods. Here, we introduce two common methods used in Fourier Spectral Methods:

## Vorticity based formulation

The vorticity-streamfunction formulation is a common method to solve the Navier-Stokes equations in two dimensional. The vorticity $\boldsymbol{\omega}$ is defined as the curl of the velocity field $\mathbf{u}$: $\boldsymbol{\omega}=\nabla\times\mathbf{u}$. If we take curl of the momentum equation, we will get:

$$
\nabla\times\frac{\partial\mathbf{u}}{\partial t}+\nabla\times\left(\mathbf{u}\cdot\nabla\right)\mathbf{u}=\nabla\times-\frac{1}{\rho}\nabla p+\nabla\times\nu \nabla^2 \mathbf{u} + \nabla\times\frac{1}{\rho}\mathbf{f}.
$$

For the first term in the left hand side, we have:

$$
\nabla\times\frac{\partial\mathbf{u}}{\partial t}=\frac{\partial\nabla\times\mathbf{u}}{\partial t}=\frac{\partial\boldsymbol{\omega}}{\partial t}.
$$




For the second term in the left hand side, it could be a little bit complicated. We can first introduce the "BAC-CAB" rule for the triple product of vectors:

$$
\mathbf{A}\times(\mathbf{B}\times\mathbf{C})=\mathbf{B}(\mathbf{A}\cdot\mathbf{C})-\mathbf{C}(\mathbf{A}\cdot\mathbf{B}).
$$

Thus, we have

$$
\mathbf{u}\times (\nabla \times \mathbf{u})=\nabla \left(\mathbf{u} \cdot \mathbf{u}\right)-\mathbf{u}(\nabla \cdot \mathbf{u}).
$$

Meanwhile, since

$$
\nabla \left(\mathbf{u} \cdot \mathbf{u}\right)= 2\mathbf{u} \cdot \nabla \mathbf{u}

$$










---

$$
\begin{align}
\frac{1}{2}\nabla\left(\mathbf{u}\cdot\mathbf{u}\right)&=\frac{1}{2}\nabla\left(\mathbf{u}\cdot\mathbf{u}\right)\\
\end{align}
$$

First, we need to introduce two vector identities:



$$
\frac{1}{2}\nabla\left(\mathbf{u}\cdot\mathbf{u}\right)=\mathbf{u}\times\left(\nabla\times\mathbf{u}\right)+\left(\mathbf{u}\cdot\nabla\right)\mathbf{u},
$$

and

$$
\nabla\times\left(\mathbf{u}\times\mathbf{v}\right)=\mathbf{u}\left(\nabla\cdot\mathbf{v}\right)-\mathbf{v}\left(\nabla\cdot\mathbf{u}\right)+\left(\mathbf{v}\cdot\nabla\right)\mathbf{u}-\left(\mathbf{u}\cdot\nabla\right)\mathbf{v}.
$$

These two equations are hard to derive directly. A simple proof could be using the index notation. 

$$
\begin{align}
\nabla  \times \left( {{\mathbf{u}} \times {\mathbf{v}}} \right) &= {{\mathbf{e}}_i} \times {\partial _i}\left( {{u_j}{{\mathbf{e}}_j} \times {v_k}{{\mathbf{e}}_k}} \right)\\
&= {\partial _i}\left( {{u_j}{v_k}} \right){{\mathbf{e}}_i} \times \left( {{{\mathbf{e}}_j} \times {{\mathbf{e}}_k}} \right)\\
&= \left( {{\partial _i}{u_j}{v_k} + {u_j}{\partial _i}{v_k}} \right)\left( {\left( {{{\mathbf{e}}_i} \cdot {{\mathbf{e}}_k}} \right){{\mathbf{e}}_j} - \left( {{{\mathbf{e}}_i} \cdot {{\mathbf{e}}_j}} \right){{\mathbf{e}}_k}} \right)\\
&= \left( {{\partial _i}{u_j}{v_k} + {u_j}{\partial _i}{v_k}} \right)\left( {{\delta _{ik}}{{\mathbf{e}}_j} - {\delta _{ij}}{{\mathbf{e}}_k}} \right)\\
&= {\partial _i}{u_j}{v_i}{{\mathbf{e}}_j} - {\partial _i}{u_i}{v_k}{{\mathbf{e}}_k} + {u_j}{\partial _i}{v_i}{{\mathbf{e}}_j} - {u_i}{\partial _i}{v_k}{{\mathbf{e}}_k}\\
&= {\mathbf{v}} \cdot \nabla {\mathbf{u}} - \left( {\nabla  \cdot {\mathbf{u}}} \right){\mathbf{v}} + \left( {\nabla  \cdot {\mathbf{v}}} \right){\mathbf{u}} - {\mathbf{u}} \cdot \nabla {\mathbf{v}}
\end{align}
$$