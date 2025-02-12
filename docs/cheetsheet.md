# Cheetsheet
## Data Types

* $[B,C,H,W,\cdots]$: [Batch, Channel, x, y, $\cdots$]
* $[B,T,C,H,W,\cdots]$: [Batch, Trajectory, Channel, x, y, $\cdots$]
* `SpatialTensor`/`SpatialArray`: A tensor/array in physical space.
* `FourierTensor`/`FourierArray`: A tensor/array in Fourier space, i.e., the tensor is a complex tensor.

## Operators

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