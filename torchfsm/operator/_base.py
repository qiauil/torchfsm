import torch, os
from torch import Tensor
from typing import Union, Sequence, Callable, Optional, Tuple, Literal, List
from .._utils import default,clean_up_memory
from .._type import ValueList, SpatialTensor, FourierTensor
from ..mesh import FourierMesh, MeshGrid
from ..integrator import ETDRKIntegrator, SETDRKIntegrator, RKIntegrator
from ..traj_recorder import _TrajRecorder
from ..errors import NanSimulationError, OutOfMemoryError
from abc import ABC, abstractmethod
from tqdm.auto import tqdm
from typing import Literal
from torch.cuda import OutOfMemoryError as TorchOutOfMemoryError
from warnings import warn
from copy import deepcopy

class LinearCoef(ABC):
    r"""
    Abstract class for linear coefficients.
    """

    @abstractmethod
    def __call__(
        self, f_mesh: FourierMesh, n_channel: int
    ) -> FourierTensor["B C H ..."]:
        r"""
        Abstract method to be implemented by subclasses. It should define the linear coefficient tensor.

        Args:
            f_mesh (FourierMesh): Fourier mesh object.
            n_channel (int): Number of channels of the input tensor.

        Returns:
            FourierTensor: Linear coefficient tensor.
        """

        raise NotImplementedError

    def nonlinear_like(
        self,
        u_fft: FourierTensor["B C H ..."],
        f_mesh: FourierMesh,
        u: Optional[SpatialTensor["B C H ..."]] = None,
    ) -> FourierTensor["B C H ..."]:
        r"""
        Calculate the result out based on the linear coefficient. It is designed to have same pattern as the nonlinear function.

        Args:
            u_fft (FourierTensor): Fourier-transformed input tensor.
            f_mesh (FourierMesh): Fourier mesh object.
            u (Optional[SpatialTensor]): Corresponding tensor of u_fft in spatial domain. This option aims to avoid repeating the inverse FFT operation in operators.

        Returns:
            FourierTensor: Nonlinear-like tensor.
        """
        return self(f_mesh, u_fft.shape[1]) * u_fft


class NonlinearFunc(ABC):
    r"""
    Abstract class for nonlinear functions.

    Args:
        dealiasing_swtich (bool): Whether to apply dealiasing. Default is True.
            If True, the dealiased version of u_fft will be input to the function in operator.
            If False, the original u_fft will be used.
    """

    def __init__(self, dealiasing_swtich: bool = True) -> None:
        self._dealiasing_swtich = dealiasing_swtich

    @abstractmethod
    def __call__(
        self,
        u_fft: FourierTensor["B C H ..."],
        f_mesh: FourierMesh,
        u: Optional[SpatialTensor["B C H ..."]] = None,
    ) -> FourierTensor["B C H ..."]:
        r"""
        Abstract method to be implemented by subclasses. It should define the nonlinear function.

        Args:
            u_fft (FourierTensor): Fourier-transformed input tensor.
            f_mesh (FourierMesh): Fourier mesh object.
            u (Optional[SpatialTensor]): Corresponding tensor of u_fft in spatial domain. This option aims to avoid repeating the inverse FFT operation in operators.

        Returns:
            FourierTensor: Result of the nonlinear function.
        """
        raise NotImplementedError

    def spatial_value(
        self,
        u_fft: FourierTensor["B C H ..."],
        f_mesh: FourierMesh,
        u: Optional[SpatialTensor["B C H ..."]] = None,
    ) -> SpatialTensor["B C H ..."]:
        r"""
        Return the result of the nonlinear function in spatial domain.

        Args:
            u_fft (FourierTensor): Fourier-transformed input tensor.
            f_mesh (FourierMesh): Fourier mesh object.
            u (Optional[SpatialTensor]): Corresponding tensor of u_fft in spatial domain. This option aims to avoid repeating the inverse FFT operation in operators.

        Returns:
            SpatialTensor: Result of the nonlinear function in spatial domain.
        """

        return f_mesh.ifft(self(u_fft, f_mesh, u)).real


class CoreGenerator(ABC):
    r"""
    Abstract class for core generator. A core generator is a callable that generates a linear coefficient or a nonlinear function based on the Fourier mesh and channels of the tensor.
    """

    @abstractmethod
    def __call__(
        self, f_mesh: FourierMesh, n_channel: int
    ) -> Union[LinearCoef, NonlinearFunc]:
        r"""
        Abstract method to be implemented by subclasses. It should define the core generator.

        Args:
            f_mesh (FourierMesh): Fourier mesh object.
            n_channel (int): Number of channels of the input tensor.

        Returns:
            Union[LinearCoef, NonlinearFunc]: Linear coefficient or nonlinear function.
        """
        raise NotImplementedError


GeneratorLike = Union[
    CoreGenerator, Callable[[FourierMesh, int], Union[LinearCoef, NonlinearFunc]]
]
# GeneratorLike is a type that can be either a CoreGenerator or a callable function
# It is used to define the type of generator functions that can be passed to the Operator class.
# This allows for more flexibility in defining the behavior of the operator.

# Operator


def check_value_with_mesh(
    u: SpatialTensor["B C H ..."],
    mesh: Union[Sequence[tuple[float, float, int]], MeshGrid, FourierMesh],
):
    r"""
    Check if the value and mesh are compatible. If not, raise a ValueError.

    Args:
        u (SpatialTensor): Input tensor of shape (B, C, H, ...).
        mesh (Union[Sequence[tuple[float, float, int]],MeshGrid,FourierMesh]): Mesh information or mesh object.
    """
    if isinstance(mesh, FourierMesh) or isinstance(mesh, MeshGrid):
        mesh = mesh.mesh_info
    n_dim = len(mesh)
    value_shape = u.shape
    if len(value_shape) - 2 != n_dim:
        raise ValueError(
            f"the value shape {value_shape} is not compatible with mesh dim {n_dim}"
        )
    for i in range(n_dim):
        if value_shape[i + 2] != mesh[i][2]:
            raise ValueError(
                f"Expect to have {mesh[i][2]} points in dim {i} but got {value_shape[i+2]}"
            )


class _MutableMixIn:
    r"""
    Mixin class for mutable operations. This class supports basic arithmetic operations for the operator.
    """

    def __radd__(self, other):
        return self + other

    def __iadd__(self, other):
        return self + other

    def __sub__(self, other):
        try:
            return self + (-1 * other)
        except Exception:
            return NotImplemented

    def __rsub__(self, other):
        try:
            return other + (-1 * self)
        except Exception:
            return NotImplemented

    def __isub__(self, other):
        return self - other

    def __rmul__(self, other):
        return self * other

    def __imul__(self, other):
        return self * other

    def __truediv__(self, other):
        try:
            return self * (1 / other)
        except:
            return NotImplemented


class _InverseSolveMixin:
    _state_dict: Optional[dict]
    register_mesh: Callable

    r"""
    Mixin class for inverse solving operations. This class supports solving the linear operator equation.
    """

    def solve(
        self,
        b: Optional[torch.Tensor] = None,
        b_fft: Optional[torch.Tensor] = None,
        mesh: Optional[
            Union[Sequence[tuple[float, float, int]], MeshGrid, FourierMesh]
        ] = None,
        n_channel: Optional[int] = None,
        return_in_fourier=False,
    ) -> Union[SpatialTensor["B C H ..."], SpatialTensor["B C H ..."]]:
        r"""
        Solve the linear operator equation $Ax = b$, where $A$ is the linear operator and $b$ is the right-hand side.

        Args:
            b (Optional[torch.Tensor]): Right-hand side tensor in spatial domain. If None, b_fft should be provided.
            b_fft (Optional[torch.Tensor]): Right-hand side tensor in Fourier domain. If None, b should be provided.
            mesh (Optional[Union[Sequence[tuple[float, float, int]], MeshGrid, FourierMesh]]): Mesh information or mesh object. If None, the mesh registered in the operator will be used.
            n_channel (Optional[int]): Number of channels of $x$. If None, the number of channels registered in the operator will be used.
            return_in_fourier (bool): If True, return the result in Fourier domain. If False, return the result in spatial domain.

        Returns:
            Union[SpatialTensor["B C H ..."], FourierTensor["B C H ..."]]: Solution tensor in spatial or Fourier domain.
        """
        if not (mesh is not None and n_channel is not None):
            assert (
                self._state_dict["f_mesh"] is not None
            ), "Mesh and n_channel should be given when calling solve"
        if not (mesh is None and n_channel is None):
            mesh = self._state_dict["f_mesh"] if mesh is None else mesh
            n_channel = (
                self._state_dict["n_channel"] if n_channel is None else n_channel
            )
            self.register_mesh(mesh, n_channel)
        if self._state_dict["invert_linear_coef"] is None:
            self._state_dict["invert_linear_coef"] = torch.where(
                self._state_dict["linear_coef"] == 0,
                1.0,
                1 / self._state_dict["linear_coef"],
            )
        if b_fft is None:
            b_fft = self._state_dict["f_mesh"].fft(b)
        value_fft = b_fft * self._state_dict["invert_linear_coef"]
        if return_in_fourier:
            return value_fft
        else:
            return self._state_dict["f_mesh"].ifft(value_fft).real


class _DeAliasMixin:
    _de_aliasing_rate: float
    _state_dict: Optional[dict]

    r"""
    Mixin class for de-aliasing operations. This class supports setting the de-aliasing rate for the nonlinear operator.
    """

    def set_de_aliasing_rate(self, de_aliasing_rate: float):
        r"""
        Set the de-aliasing rate for the nonlinear operator.
        Args:
            de_aliasing_rate (float): De-aliasing rate. Default is 2/3.
        """

        self._de_aliasing_rate = de_aliasing_rate
        self._state_dict = {key:None for key in self._state_dict.keys()}


class OperatorLike(_MutableMixIn):
    r"""
    Base class for All Operators.

    Args:
        operator_cores (Optional[ValueList[Union[LinearCoef, NonlinearFunc, GeneratorLike]]]): List of operator generators/LinearCoef/NonLinearFunc. Default is None.
            that represent the real manipulations.
        coefs (Optional[List]): List of coefficients for each operator_core. Default is None.
            If None, all coefficients are set to 1.
            The length of the list should match the number of operator_core.

    """

    def __init__(
        self,
        operator_cores: Optional[ValueList[Union[LinearCoef, NonlinearFunc, GeneratorLike]]] = None,
        coefs: Optional[List] = None,
    ) -> None:
        super().__init__()
        self.operator_cores = default(operator_cores, [])
        if not isinstance(self.operator_cores, list):
            self.operator_cores = [self.operator_cores]
        self.coefs = default(coefs, [1] * len(self.operator_cores))
        self._state_dict = {
            "f_mesh": None,
            "n_channel": None,
            "linear_coef": None,
            "nonlinear_func": None,
            "operator": None,
            "integrator": None,
            "invert_linear_coef": None,
        }
        self._de_aliasing_rate = 2 / 3
        self._value_mesh_check_func = lambda dim_value, dim_mesh: True
        self._integrator = "auto"
        self._integrator_config = {}
        self._default_nonlinear_integrator = SETDRKIntegrator.SETDRK4
        self._default_nonlinear_integrator_config  ={}
        self._is_etdrk_integrator = True

    @property
    def is_linear(self) -> bool:
        r"""
        Check if the operator is linear.

        Returns:
            bool: True if the operator is linear, False otherwise.
        """
        assert (
            self._state_dict["f_mesh"] is not None
        ), "Mesh should be registered before checking if the operator is linear"
        return (
            self._state_dict["nonlinear_func"] is None
            and self._state_dict["linear_coef"] is not None
        )

    def _build_linear_coefs(self, linear_coefs: Optional[Sequence[LinearCoef]]):
        r"""
        Build the linear coefficients.

        Args:
            linear_coefs (Optional[Sequence[LinearCoef]]): List of linear coefficients.

        """
        if len(linear_coefs) == 0:
            linear_coefs = None
        else:
            linear_coefs = sum(
                [
                    coef * op(self._state_dict["f_mesh"], self._state_dict["n_channel"])
                    for coef, op in linear_coefs
                ]
            )
        self._state_dict["linear_coef"] = linear_coefs
        clean_up_memory()

    def _build_nonlinear_funcs(
        self, nonlinear_funcs: Optional[Sequence[NonlinearFunc]]
    ):
        r"""
        Build the nonlinear functions.

        Args:
            nonlinear_funcs (Optional[Sequence[NonlinearFunc]]): List of nonlinear functions
        """
        if len(nonlinear_funcs) == 0:
            nonlinear_funcs_all = None
        else:
            self._state_dict["f_mesh"].set_default_rel_freq_threshold(
                self._de_aliasing_rate
            )

            def nonlinear_funcs_all(u_fft):
                result = 0.0
                dealiased_u_fft = None
                dealiased_u = None
                u = None
                for coef, fun in nonlinear_funcs:
                    if fun._dealiasing_swtich:
                        if dealiased_u_fft is None:
                            dealiased_u_fft = u_fft * self._state_dict[
                                "f_mesh"
                            ].low_pass_filter(self._de_aliasing_rate)
                            dealiased_u = (
                                self._state_dict["f_mesh"].ifft(dealiased_u_fft).real
                            )
                        result += coef * fun(
                            dealiased_u_fft,
                            self._state_dict["f_mesh"],
                            dealiased_u,
                        )
                    else:
                        if u is None:
                            u = self._state_dict["f_mesh"].ifft(u_fft).real
                        result += coef * fun(
                            u_fft,
                            self._state_dict["f_mesh"],
                            u,
                        )

                return result

        self._state_dict["nonlinear_func"] = nonlinear_funcs_all
        clean_up_memory()

    def _build_operator(self):
        r"""
        Build the operator based on the linear coefficient and nonlinear function.
        If both linear coefficient and nonlinear function are None, the operator is set to None.
        """
        if self._state_dict["nonlinear_func"] is None:

            def operator(u_fft):
                return self._state_dict["linear_coef"] * u_fft

        elif self._state_dict["linear_coef"] is None:

            def operator(u_fft):
                return self._state_dict["nonlinear_func"](u_fft)

        elif (
            self._state_dict["nonlinear_func"] is not None
            and self._state_dict["linear_coef"] is not None
        ):

            def operator(u_fft):
                return self._state_dict["linear_coef"] * u_fft + self._state_dict[
                    "nonlinear_func"
                ](u_fft)

        else:
            raise ValueError(
                "Both linear coefficient and nonlinear function are None. Cannot build operator."
            )

        self._state_dict["operator"] = operator
        clean_up_memory()

    def _build_integrator(
        self,
        dt: float,
    ):
        r"""
        Build the integrator based on the provided time step and integrator type.

        Args:
            dt (float): Time step for the integrator.
        """
        if self._integrator == "auto":
            if self.is_linear:
                solver = ETDRKIntegrator.ETDRK0
            else:
                solver = self._default_nonlinear_integrator
                self._integrator_config = deepcopy(self._default_nonlinear_integrator_config)
        else:
            solver = self._integrator
        self._is_etdrk_integrator = isinstance(solver, ETDRKIntegrator) or isinstance(
            solver, SETDRKIntegrator
        )
        try:
            if self._is_etdrk_integrator:
                if solver == ETDRKIntegrator.ETDRK0:
                    assert (
                        self.is_linear
                    ), "The ETDRK0 integrator only supports linear term"
                    self._state_dict["integrator"] = solver.value(
                        dt,
                        self._state_dict["linear_coef"],
                        **self._integrator_config,
                    )
                else:
                    if self._state_dict["linear_coef"] is None:
                        linear_coef = torch.tensor(
                            [0.0],
                            dtype=self._state_dict["f_mesh"].dtype,
                            device=self._state_dict["f_mesh"].device,
                        )
                    else:
                        linear_coef = self._state_dict["linear_coef"]
                    self._state_dict["integrator"] = solver.value(
                        dt,
                        linear_coef,
                        self._state_dict["nonlinear_func"],
                        **self._integrator_config,
                    )
                setattr(
                    self._state_dict["integrator"],
                    "forward",
                    lambda u_fft, dt: self._state_dict["integrator"].step(u_fft),
                )
            elif isinstance(solver, RKIntegrator):
                if self._state_dict["operator"] is None:
                    self._build_operator()
                self._state_dict["integrator"] = solver.value(**self._integrator_config)
                setattr(
                    self._state_dict["integrator"],
                    "forward",
                    lambda u_fft, dt: self._state_dict["integrator"].step(
                        self._state_dict["operator"], u_fft, dt
                    ),
                )
            else:
                raise ValueError(
                    "The integrator should be 'auto' or an instance of ETDRKIntegrator, SETDRKIntegrator or RKIntegrator"
                )
        except TorchOutOfMemoryError as e:
            error_msg = ["Cuda out of memory when building the integrator."]
            error_msg.append("Original error message: {}".format(str(e)))
            if isinstance(solver, SETDRKIntegrator):
                error_msg.append(
                    "Since you are using SETDRKIntegrator, there are some options to reduce the memory usage:"
                )
                error_msg.append(
                    "1. set the `cpu_cached` to True when call `Operator.set_integrator(). This will slow down the integrator building speed but not effect the integration speed. However, it will increase the CPU memory usage when building the solver."
                )
                error_msg.append(
                    "2. set the `n_integration_points` to a smaller number when call `Operator.set_integrator(). This will effect the stability of the integrator."
                )
                error_msg.append(
                    "3. Use other integrators, such as ETDRKIntegrator or RKIntegrator or low-order SETDRK. They are more memory efficient but may not be as accurate as SETDRKIntegrator."
                )
            else:
                error_msg.append("Please try to use a smaller mesh or a low-order integrator.")
            raise OutOfMemoryError(os.linesep.join(error_msg))
        clean_up_memory()

    def _pre_check(
        self,
        u: Optional[SpatialTensor["B C H ..."]] = None,
        u_fft: Optional[FourierTensor["B C H ..."]] = None,
        mesh: Union[Sequence[tuple[float, float, int]], MeshGrid, FourierMesh] = None,
    ) -> Tuple[FourierMesh, int]:
        r"""
        Pre-check the input tensor and mesh. If the mesh is not registered, register it.

        Args:
            u (Optional[SpatialTensor]): Input tensor in spatial domain. Default is None.
            u_fft (Optional[FourierTensor]): Input tensor in Fourier domain. Default is None.
                At least one of u or u_fft should be provided.
            mesh (Union[Sequence[tuple[float, float, int]], MeshGrid, FourierMesh]): Mesh information or mesh object. Default is None.
                If None, the mesh registered in the operator will be used.

        Returns:
            Tuple[FourierMesh, int]: Tuple of Fourier mesh and number of channels.
        """

        if u_fft is None and u is None:
            raise ValueError("Either u or u_fft should be given")
        if u_fft is not None and u is not None:
            assert u.shape == u_fft.shape, "The shape of u and u_fft should be the same"
        assert mesh is not None, "Mesh should be given"
        value_device = u.device if u is not None else u_fft.device
        value_dtype = u.dtype if u is not None else u_fft.dtype
        if not isinstance(mesh, FourierMesh):
            if not isinstance(mesh, MeshGrid):
                mesh = FourierMesh(mesh, device=value_device, dtype=value_dtype)
            else:
                mesh = FourierMesh(mesh)
        n_channel = u.shape[1] if u is not None else u_fft.shape[1]
        value_shape = u.shape if u is not None else u_fft.shape
        assert (
            len(value_shape) == mesh.n_dim + 2
        ), f"the value shape {value_shape} is not compatible with mesh dim {mesh.n_dim}"
        for i in range(mesh.n_dim):
            assert (
                value_shape[i + 2] == mesh.mesh_info[i][2]
            ), f"Expect to have {mesh.mesh_info[i][2]} points in dim {i} but got {value_shape[i+2]}"
        assert (
            value_device == mesh.device
        ), "The device of mesh {} and the device of value {} are not the same".format(
            mesh.device, value_device
        )
        # assert value_dtype==mesh.dtype, "The dtype of mesh {} and the dtype of value {} are not the same".format(mesh.dtype,value_dtype)
        # value fft is a complex dtype
        assert self._value_mesh_check_func(
            len(value_shape) - 2, mesh.n_dim
        ), "Value and mesh do not match the requirement"
        return mesh, n_channel

    def register_mesh(
        self,
        mesh: Union[Sequence[tuple[float, float, int]], MeshGrid, FourierMesh],
        n_channel: int,
        device=None,
        dtype=None,
    ):
        r"""
        Register the mesh and number of channels for the operator. Once a mesh is registered, mesh information is not required for integration and operator call.

        Args:
            mesh (Union[Sequence[tuple[float, float, int]], MeshGrid, FourierMesh]): Mesh information or mesh object.
            n_channel (int): Number of channels of the input tensor.
            device (Optional[torch.device]): Device to which the mesh should be moved. Default is None.
            dtype (Optional[torch.dtype]): Data type of the mesh. Default is None.
        """
        if isinstance(mesh, FourierMesh):
            f_mesh = mesh
            if device is not None or dtype is not None:
                f_mesh.to(device=device, dtype=dtype)
        else:
            f_mesh = FourierMesh(mesh, device=device, dtype=dtype)
        for key in self._state_dict:
            self._state_dict[key] = None
        self._state_dict.update(
            {
                "f_mesh": f_mesh,
                "n_channel": n_channel,
            }
        )
        linear_coefs = []
        nonlinear_funcs = []
        for coef, core in zip(self.coefs, self.operator_cores):
            op = (
                core(f_mesh, n_channel)
                if (
                    not isinstance(core, LinearCoef)
                    and not isinstance(core, NonlinearFunc)
                )
                else core
            )
            if isinstance(op, LinearCoef):
                linear_coefs.append((coef, op))
            elif isinstance(op, NonlinearFunc):
                nonlinear_funcs.append((coef, op))
            else:
                raise ValueError(f"Operator {op} is not supported")
        clean_up_memory()
        self._build_linear_coefs(linear_coefs)
        self._build_nonlinear_funcs(nonlinear_funcs)

    def register_additional_check(self, func: Callable[[int, int], bool]):
        r"""
        Register an additional check function for the value and mesh compatibility.

        Args:
            func (Callable[[int, int], bool]): Function that takes the dimension of the value and mesh as input and returns a boolean indicating whether they are compatible.
        """
        self._value_mesh_check_func = func

    def add_core(self,core:Union[LinearCoef,NonlinearFunc,GeneratorLike],coef=1):
        r"""
        Add a generator to the operator.

        Args:
            core (Union[LinearCoef,NonlinearFunc,GeneratorLike]): Core to be added. 
            coef (float): Coefficient for the generator. Default is 1.
        """
        self.operator_cores.append(core)
        self.coefs.append(coef)

    def set_integrator(
        self,
        integrator: Union[
            Literal["auto"], ETDRKIntegrator, SETDRKIntegrator, RKIntegrator
        ],
        **integrator_config,
    ):
        r"""
        Set the integrator for the operator. The integrator is used for time integration of the operator.

        Args:
            integrator (Union[Literal["auto"], ETDRKIntegrator, SETDRKIntegrator, RKIntegrator]): Integrator to be used. If "auto", the integrator will be chosen automatically based on the operator type.
                If "auto", the integrator will be set as ETDRKIntegrator.ETDRK0 for linear operators and ETDRKIntegrator.ETDRK2 for nonlinear operators.
            **integrator_config: Additional configuration for the integrator.
        """

        if isinstance(integrator, str):
            assert (
                integrator == "auto"
            ), "The integrator should be 'auto' or an instance of ETDRKIntegrator, SETDRKIntegrator or RKIntegrator"
        else:
            assert (
                isinstance(integrator, ETDRKIntegrator)
                or isinstance(integrator, SETDRKIntegrator)
                or isinstance(integrator, RKIntegrator)
            ), "The integrator should be 'auto' or an instance of ETDRKIntegrator, SETDRKIntegrator or RKIntegrator"
        self._integrator = integrator
        self._integrator_config = integrator_config
        self._state_dict["integrator"] = None

    def set_default_nonlinear_integrator(
        self,
        integrator: Union[ETDRKIntegrator, SETDRKIntegrator, RKIntegrator],
        **integrator_config,
        ):
        r"""
        Set the default nonlinear integrator for the operator.

        Args:
            integrator (Union[ETDRKIntegrator, SETDRKIntegrator, RKIntegrator]): Integrator to be used.
            **integrator_config: Additional configuration for the integrator.
            
        """
        assert (
                isinstance(integrator, ETDRKIntegrator)
                or isinstance(integrator, SETDRKIntegrator)
                or isinstance(integrator, RKIntegrator)
            ), "The integrator should be 'auto' or an instance of ETDRKIntegrator, SETDRKIntegrator or RKIntegrator"
        self._default_nonlinear_integrator = integrator
        self._default_nonlinear_integrator_config = integrator_config
        if self._integrator == "auto":
            self._state_dict["integrator"] = None
        else:
            warn("Not automatically setting the integrator. The default nonlinear integrator will only be used if the integrator is set to 'auto'.")

    def integrate(
        self,
        u_0: Optional[torch.Tensor] = None,
        u_0_fft: Optional[torch.Tensor] = None,
        dt: float = 1,
        step: int = 1,
        mesh: Optional[
            Union[Sequence[tuple[float, float, int]], MeshGrid, FourierMesh]
        ] = None,
        progressive: bool = False,
        trajectory_recorder: Optional[_TrajRecorder] = None,
        return_in_fourier: bool = False,
        nan_check: bool = False,
    ) -> Union[
        SpatialTensor["B C H ..."],
        SpatialTensor["B T C H ..."],
        FourierTensor["B C H ..."],
        FourierTensor["B T C H ..."],
    ]:
        r"""
        Integrate the operator using the provided initial condition and time step.

        Args:
            u_0 (Optional[torch.Tensor]): Initial condition in spatial domain. Default is None.
            u_0_fft (Optional[torch.Tensor]): Initial condition in Fourier domain. Default is None.
                At least one of u_0 or u_0_fft should be provided.
            dt (float): Time step for the integrator. Default is 1.
            step (int): Number of time steps to integrate. Default is 1.
            mesh (Optional[Union[Sequence[tuple[float, float, int]], MeshGrid, FourierMesh]]): Mesh information or mesh object. Default is None.
                If None, the mesh registered in the operator will be used. You can use `register_mesh` to register a mesh before integration.
            progressive (bool): If True, show a progress bar during integration. Default is False.
            trajectory_recorder (Optional[_TrajRecorder]): Trajectory recorder for recording the trajectory during integration. Default is None.
                If None, no trajectory will be recorded. The function will only return the final frame.
            return_in_fourier (bool): If True, return the result in Fourier domain. If False, return the result in spatial domain. Default is False.
            nan_check (bool): If True, check for NaN values in the result. If NaN values are found, raise a NanSimulationError. Default is False.

        Returns:
            Union[SpatialTensor["B C H ..."], SpatialTensor["B T C H ..."], FourierTensor["B C H ..."], FourierTensor["B T C H ..."]]: Integrated result in spatial or Fourier domain.
                If trajectory_recorder is provided, the result will be a trajectory tensor of shape (B, T, C, H, ...). Otherwise, the result will be a tensor of shape (B, C, H, ...).
                If return_in_fourier is True, the result will be in Fourier domain. Otherwise, it will be in spatial domain.

        """
        if self._state_dict["f_mesh"] is None or mesh is not None:
            mesh, n_channel = self._pre_check(u=u_0, u_fft=u_0_fft, mesh=mesh)
            self.register_mesh(mesh, n_channel)
        else:
            self._pre_check(u=u_0, u_fft=u_0_fft, mesh=self._state_dict["f_mesh"])
        if self._state_dict["integrator"] is None:
            self._build_integrator(dt)
        elif self._is_etdrk_integrator:
            if self._state_dict["integrator"].dt != dt:
                self._build_integrator(dt)
        f_mesh = self._state_dict["f_mesh"]
        if u_0_fft is None:
            u_0_fft = f_mesh.fft(u_0)
        p_bar = tqdm(range(step), desc="Integrating", disable=not progressive)
        clean_up_memory()
        try:
            for i in p_bar:
                if trajectory_recorder is not None:
                    trajectory_recorder.record(i, u_0_fft)
                    if trajectory_recorder._shutdown_flag:
                        break
                u_0_fft = self._state_dict["integrator"].forward(u_0_fft, dt)
                if nan_check:
                    if torch.isnan(u_0_fft).any():
                        raise NanSimulationError(
                            f"NaN values found in the result at step {i}. Please check the input and simulation parameters."
                        )
        except TorchOutOfMemoryError as e:
            error_msg = [
                "Cuda out of memory when integrating the operator.",
                "Original error message: {}".format(str(e)),
                "Please try to use a smaller mesh or a low-order integrator.",
            ]
            raise OutOfMemoryError(os.linesep.join(error_msg))
        if trajectory_recorder is not None:
            trajectory_recorder.record(i + 1, u_0_fft)
            trajectory_recorder.return_in_fourier = return_in_fourier
            return trajectory_recorder.trajectory
        else:
            if return_in_fourier:
                return u_0_fft
            else:
                return f_mesh.ifft(u_0_fft).real

    def __call__(
        self,
        u: Optional[SpatialTensor["B C H ..."]] = None,
        u_fft: Optional[FourierTensor["B C H ..."]] = None,
        mesh: Optional[
            Union[Sequence[tuple[float, float, int]], MeshGrid, FourierMesh]
        ] = None,
        return_in_fourier=False,
    ) -> Union[SpatialTensor["B C H ..."], FourierTensor["B C H ..."]]:
        r"""
        Call the operator with the provided input tensor. The operator will apply the linear coefficient and nonlinear function to the input tensor.

        Args:
            u (Optional[SpatialTensor]): Input tensor in spatial domain. Default is None.
            u_fft (Optional[FourierTensor]): Input tensor in Fourier domain. Default is None.
                At least one of u or u_fft should be provided.
            mesh (Optional[Union[Sequence[tuple[float, float, int]], MeshGrid, FourierMesh]]): Mesh information or mesh object. Default is None.
                If None, the mesh registered in the operator will be used. You can use `register_mesh` to register a mesh before calling the operator.
            return_in_fourier (bool): If True, return the result in Fourier domain. If False, return the result in spatial domain. Default is False.

        Returns:
            Union[SpatialTensor["B C H ..."], FourierTensor["B C H ..."]]: Result of the operator in spatial or Fourier domain.
        """

        if self._state_dict["f_mesh"] is None or mesh is not None:
            mesh, n_channel = self._pre_check(u, u_fft, mesh)
            self.register_mesh(mesh, n_channel)
        else:
            self._pre_check(u=u, u_fft=u_fft, mesh=self._state_dict["f_mesh"])
        if self._state_dict["operator"] is None:
            self._build_operator()
        if u_fft is None:
            u_fft = self._state_dict["f_mesh"].fft(u)
        value_fft = self._state_dict["operator"](u_fft)
        if return_in_fourier:
            return value_fft
        else:
            return self._state_dict["f_mesh"].ifft(value_fft).real

    def to(self, device=None, dtype=None):
        r"""
        Move the operator to the specified device and change the data type.

        Args:
            device (Optional[torch.device]): Device to which the operator should be moved. Default is None.
            dtype (Optional[torch.dtype]): Data type of the operator. Default is None.
        """
        if self._state_dict is not None:
            self._state_dict["f_mesh"].to(device=device, dtype=dtype)
            self.register_mesh(
                self._state_dict["f_mesh"], self._state_dict["n_channel"]
            )


class Operator(OperatorLike, _DeAliasMixin):
    r"""
    Operator class for linear and nonlinear operations.

    Args:
        operator_cores (Optional[ValueList[Union[LinearCoef, NonlinearFunc, GeneratorLike]]]): List of operator generators/LinearCoef/NonLinearFunc. Default is None.
            that represent the real manipulations.
        coefs (Optional[List]): List of coefficients for each operator_core. Default is None.
            If None, all coefficients are set to 1.
            The length of the list should match the number of operator_core.

    """

    def __init__(
        self,
        operator_cores: Optional[ValueList[Union[LinearCoef, NonlinearFunc, GeneratorLike]]] = None,
        coefs: Optional[List] = None,
    ) -> None:
        super().__init__(operator_cores, coefs)

    def __add__(self, other):
        if isinstance(other, OperatorLike):
            return Operator(
                self.operator_cores + other.operator_cores,
                self.coefs + other.coefs,
            )
        elif isinstance(other, Tensor):
            return Operator(
                self.operator_cores
                + [lambda f_mesh, n_channel: _ExplicitSourceCore(other)],
                self.coefs + [1],
            )
        else:
            return NotImplemented

    def __mul__(self, other):
        if isinstance(other, OperatorLike):
            return NotImplemented
        else:
            return Operator(
                self.operator_cores, [coef * other for coef in self.coefs]
            )

    def __neg__(self):
        return Operator(self.operator_cores, [-1 * coef for coef in self.coefs])


class LinearOperator(OperatorLike, _InverseSolveMixin):
    r"""
    Operators that contain only linear operations.

    Args:
        linear_coef (ValueList[Union[LinearCoef, GeneratorLike]]): List of LinearCoef or linear coefficient generators. Default is None.
        coefs (Optional[List]): List of coefficients for each linear_coef. Default is None.
            If None, all coefficients are set to 1.
            The length of the list should match the number of linear_coef.
    """

    def __init__(
        self,
        linear_coef: ValueList[Union[LinearCoef, GeneratorLike]] = None,
        coefs: Optional[List] = None,
    ) -> None:
        if not isinstance(linear_coef, list):
            linear_coef = [linear_coef]
        super().__init__(
            linear_coef,
            coefs=coefs,
        )

    @property
    def is_linear(self):
        return True

    def __add__(self, other):
        if isinstance(other, LinearOperator):
            return LinearOperator(
                self.operator_cores + other.operator_cores,
                self.coefs + other.coefs,
            )
        elif isinstance(other, OperatorLike):
            return Operator(
                self.operator_cores + other.operator_cores,
                self.coefs + other.coefs,
            )
        elif isinstance(other, Tensor):
            return Operator(
                self.operator_cores
                + [lambda f_mesh, n_channel: _ExplicitSourceCore(other)],
                self.coefs + [1],
            )
        else:
            return NotImplemented

    def __mul__(self, other):
        if isinstance(other, OperatorLike):
            return NotImplemented
        else:
            return LinearOperator(
                self.operator_cores, [coef * other for coef in self.coefs]
            )

    def __neg__(self):
        return LinearOperator(
            self.operator_cores, [-1 * coef for coef in self.coefs]
        )


class NonlinearOperator(OperatorLike, _DeAliasMixin):
    r"""
    Operators that contain only nonlinear operations.

    Args:
        nonlinear_func (ValueList[Union[NonlinearFunc, GeneratorLike]]): List of NonlinearFunc or nonlinear function generators. Default is None.
        coefs (Optional[List]): List of coefficients for each nonlinear nonlinear_func. Default is None.
            If None, all coefficients are set to 1.
            The length of the list should match the number of nonlinear nonlinear_func.
    """

    def __init__(
        self,
        nonlinear_func: ValueList[Union[NonlinearFunc, GeneratorLike]] = None,
        coefs: Optional[List] = None,
    ) -> None:
        if not isinstance(nonlinear_func, list):
            nonlinear_func = [nonlinear_func]
        super().__init__(
            nonlinear_func,
            coefs=coefs,
        )

    @property
    def is_linear(self):
        return False

    def __add__(self, other):
        if isinstance(other, NonlinearOperator):
            return NonlinearOperator(
                self.operator_cores + other.operator_cores,
                self.coefs + other.coefs,
            )
        elif isinstance(other, OperatorLike):
            return Operator(
                self.operator_cores + other.operator_cores,
                self.coefs + other.coefs,
            )
        elif isinstance(other, Tensor):
            return Operator(
                self.operator_cores
                + [lambda f_mesh, n_channel: _ExplicitSourceCore(other)],
                self.coefs + [1],
            )
        else:
            return NotImplemented

    def __mul__(self, other):
        if isinstance(other, OperatorLike):
            return NotImplemented
        else:
            return NonlinearOperator(
                self.operator_cores, [coef * other for coef in self.coefs]
            )

    def __neg__(self):
        return NonlinearOperator(
            self.operator_cores, [-1 * coef for coef in self.coefs]
        )


# Explicit Source


class _ExplicitSourceCore(NonlinearFunc):
    r"""
    Implementation of the explicit source term for the operator.

    Args:
        source (SpatialTensor["B C H ..."]): Source term in spatial domain. This is a tensor that represents the source term in the spatial domain.
    """

    def __init__(self, source: SpatialTensor["B C H ..."]) -> None:
        super().__init__(dealiasing_swtich=False)
        fft_dim = [i + 2 for i in range(source.dim() - 2)]
        self.source = torch.fft.fftn(source, dim=fft_dim)

    def __call__(
        self,
        u_fft: FourierTensor["B C H ..."],
        f_mesh: FourierMesh,
        u: Optional[SpatialTensor["B C H ..."]] = None,
    ) -> FourierTensor["B C H ..."]:
        if self.source.device != f_mesh.device:
            self.source = self.source.to(f_mesh.device)
        return self.source


class ExplicitSource(NonlinearOperator):
    r"""
    Explicit source term for the operator. This class is used to represent an explicit source term in the operator.
        Note that this class is an operator wrapper. The real implementation of the source term is in the _ExplicitSourceCore class.

    Args:
        source (torch.Tensor): Source term in spatial domain. This is a tensor that represents the source term in the spatial domain.
    """

    def __init__(self, source: torch.Tensor) -> None:
        super().__init__(_ExplicitSourceCore(source))
