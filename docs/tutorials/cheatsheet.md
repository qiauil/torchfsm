# Cheat Sheet
## Data Types

* $[B,C,H,W,\cdots]$: [Batch, Channel, x, y, $\cdots$]
* $[B,T,C,H,W,\cdots]$: [Batch, Trajectory, Channel, x, y, $\cdots$]
* `SpatialTensor`/`SpatialArray`: A tensor/array in physical space.
* `FourierTensor`/`FourierArray`: A tensor/array in Fourier space, i.e., the tensor is a complex tensor.

## Operators

### Basic Operators

* $\mathbf{u}$: $n$ d vector field.
* $\phi$: scalar field (0d vector field.).
* $i$: index of the coordinate.
* $I$: maximum index of the coordinate.
* $u_i$: $i$ th component of the vector field.

| Operator    | Equation | Is linear operator |
| -------- | ------- | ------- |
| SpatialDerivative  | $\frac{\partial ^n}{\partial i} \phi$ | True |
| Gradient | $\nabla \phi = \left[\begin{matrix}\frac{\partial \phi}{\partial x} \\\frac{\partial \phi}{\partial y} \\\cdots \\\frac{\partial \phi}{\partial I} \\\end{matrix}\right]$  | True |
| Divergence    | $\nabla \cdot \mathbf{u} = \sum_{i=0}^I \frac{\partial u_i}{\partial i}$    | True |
| Laplacian  | $\nabla^2\mathbf{u}=\left[\begin{matrix}\sum_{i=0}^I \frac{\partial^2 u_x}{\partial i^2 } \\ \sum_{i=0}^I \frac{\partial^2 u_y}{\partial i^2 } \\ \cdots \\ \sum_{i=0}^I \frac{\partial^2 u_I}{\partial i^2 } \\ \end{matrix} \right]$ | True |
| Biharmonic  | $\nabla^4\mathbf{u}=\left[\begin{matrix}(\sum_{i=0}^I\frac{\partial^2}{\partial i^2 })(\sum_{j=0}^I\frac{\partial^2}{\partial j^2 })u_x \\ (\sum_{i=0}^I\frac{\partial^2}{\partial i^2 })(\sum_{j=0}^I\frac{\partial^2}{\partial j^2 })u_y \\ \cdots \\ (\sum_{i=0}^I\frac{\partial^2}{\partial i^2 })(\sum_{j=0}^I\frac{\partial^2}{\partial j^2 })u_i \\ \end{matrix} \right]$ | True |
| Curl (2D input)  | $\nabla \times \mathbf{u} = \frac{\partial u_y}{\partial x}-\frac{\partial u_x}{\partial y}$ | False |
| Curl (3D input)  | $\nabla \times \mathbf{u} = \left[\begin{matrix} \frac{\partial u_z}{\partial y}-\frac{\partial u_y}{\partial z} \\ \frac{\partial u_x}{\partial z}-\frac{\partial u_z}{\partial x} \\ \frac{\partial u_y}{\partial x}-\frac{\partial u_x}{\partial y} \end{matrix} \right]$ | False |
| ConservativeConvection  | $\nabla \cdot \mathbf{u}\mathbf{u}=\left[\begin{matrix}\sum_{i=0}^I \frac{\partial u_i u_x }{\partial i} \\\sum_{i=0}^I \frac{\partial u_i u_y }{\partial i} \\ \cdots\\ \sum_{i=0}^I \frac{\partial u_i u_I }{\partial i} \\ \end{matrix} \right]$ | True |
| Convection  | $\mathbf{u} \cdot \nabla  \mathbf{u}=\left[\begin{matrix}\sum_{i=0}^I u_i\frac{\partial u_x }{\partial i} \\\sum_{i=0}^I u_i\frac{\partial u_y }{\partial i} \\\cdots\\\sum_{i=0}^I u_i\frac{\partial u_I }{\partial i} \\\end{matrix} \right]$ | True |

All the above operators can be imported from `torchfsm.operator` module. Corresponding functions that directly apply the operator to the input tensor are also available in `torchfsm.functional` module.

### Navier-Stokes Operators
* $\mathbf{u}$: vector field.
* $u,v$: components of 2d velocity field.
* $\omega$: vorticity scalar of a 2d velocity.
* $p$: pressure.

| Operator    | Equation |
| -------- | ------- |
|VorticityConvection|$(\mathbf{u}\cdot\nabla) \omega$|
|NSPressureConvection|$-\nabla (\nabla^{-2} \nabla \cdot (\left(\mathbf{u}\cdot\nabla\right)\mathbf{u}-f))-\left(\mathbf{u}\cdot\nabla\right)\mathbf{u} + \mathbf{f}$|
|Vorticity2Velocity|$[u,v]=[-\frac{\partial \nabla^{-2}\omega}{\partial y},\frac{\partial \nabla^{-2}\omega}{\partial x}]$|
|Velocity2Pressure|$-\nabla^{-2} (\nabla \cdot (\left(\mathbf{u}\cdot\nabla\right)\mathbf{u}-f))$|
|Vorticity2Pressure|$\begin{matrix}\mathbf{u}=[u,v]=[-\frac{\partial \nabla^{-2}\omega}{\partial y},\frac{\partial \nabla^{-2}\omega}{\partial x}]\\ p= \nabla^{-2} (\nabla \cdot (\left(\mathbf{u}\cdot\nabla\right)\mathbf{u}-f))\end{matrix}$|

All the above operators can be imported from `torchfsm.operator` module. Corresponding functions for `Vorticity2Velocity`, `Velocity2Pressure` and `Vorticity2Pressure` are also available in `torchfsm.functional` module.

## Equations
| Operator    | Equation |
| -------- | ------- |
|Burgers|$\frac{\partial \mathbf{u}}{\partial t} =-\mathbf{u} \cdot \nabla \mathbf{u} + \nu \nabla^2 \mathbf{u}$|
|KuramotoSivashinsky|$\frac{\partial \phi}{\partial t}=-\frac{\partial^2 \phi}{\partial x^2} -\frac{\partial^4 \phi}{\partial x^4} - \phi\frac{\partial\phi}{\partial x}$|
|KuramotoSivashinskyHighDim|$\frac{\partial \mathbf{\phi}}{\partial t}=-\nabla^2 \phi- \nabla^4 \phi - \frac{1}{2} \| \nabla \phi \|^2$|
|KortewegDeVries|$\frac{\partial \phi}{\partial t}=-c_1\frac{\partial^3 \phi}{\partial x^3} + c_2 \phi\frac{\partial\phi}{\partial x}$|
|NavierStokesVorticity|$\frac{\partial \omega}{\partial t} + (\mathbf{u}\cdot\nabla) \omega = \frac{1}{Re} \nabla^2 \omega + \nabla \times \mathbf{f}$|
|NavierStokes|$\frac{\partial\mathbf{u}}{\partial t}=-\nabla (\nabla^{-2} \nabla \cdot (\left(\mathbf{u}\cdot\nabla\right)\mathbf{u}-f))-\left(\mathbf{u}\cdot\nabla\right)\mathbf{u}+\nu \nabla^2 \mathbf{u} + \mathbf{f}$|