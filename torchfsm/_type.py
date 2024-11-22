from typing import _SpecialForm,_type_check,List,Union,_tp_cache,_AnnotatedAlias
from torch import Tensor

@_SpecialForm
def ValueList(self, parameters):
    arg = _type_check(parameters, f"{self} requires a single type.")
    return Union[arg, List[arg]]

class SpatialTensor:

    __slots__ = ()

    def __new__(cls, *args, **kwargs):
        raise TypeError("Type SpatialTensor cannot be instantiated.")

    @_tp_cache
    def __class_getitem__(cls, shape:str):
        return _AnnotatedAlias(Tensor, shape)

    def __init_subclass__(cls, *args, **kwargs):
        raise TypeError(
            "Cannot subclass {}.SpatialTensor".format(cls.__module__)
        )
    
class FourierTensor:
    
    __slots__ = ()

    def __new__(cls, *args, **kwargs):
        raise TypeError("Type FourierTensor cannot be instantiated.")

    @_tp_cache
    def __class_getitem__(cls, shape:str):
        return _AnnotatedAlias(Tensor, shape)

    def __init_subclass__(cls, *args, **kwargs):
        raise TypeError(
            "Cannot subclass {}.FourierTensor".format(cls.__module__)
        )