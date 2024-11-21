from typing import _SpecialForm,_type_check,List,Union,_tp_cache,_AnnotatedAlias
from torch import Tensor

@_SpecialForm
def ValueList(self, parameters):
    arg = _type_check(parameters, f"{self} requires a single type.")
    return Union[arg, List[arg]]

class PhysicalTensor:

    __slots__ = ()

    def __new__(cls, *args, **kwargs):
        raise TypeError("Type PhysicalTensor cannot be instantiated.")

    @_tp_cache
    def __class_getitem__(cls, shape:str):
        return _AnnotatedAlias(Tensor, shape)

    def __init_subclass__(cls, *args, **kwargs):
        raise TypeError(
            "Cannot subclass {}.PhysicalTensor".format(cls.__module__)
        )
    
class SpectralTensor:
    
    __slots__ = ()

    def __new__(cls, *args, **kwargs):
        raise TypeError("Type SpectralTensor cannot be instantiated.")

    @_tp_cache
    def __class_getitem__(cls, shape:str):
        return _AnnotatedAlias(Tensor, shape)

    def __init_subclass__(cls, *args, **kwargs):
        raise TypeError(
            "Cannot subclass {}.SpectralTensor".format(cls.__module__)
        )