# torchfsm/_util.py
# Since the function provided here are both used in utils and mesh modules, we put them in _util.py to avoid circular import.

import torch,gc
from typing import Union, Optional,Tuple

def default(value, default):
    """
    Return the default value if the value is None.

    Args:
        value: The value to check.
        default: The default value to return if value is None.

    Returns:
        The value if it is not None, otherwise the default value.
    """
    return value if value is not None else default

def format_device_dtype(
    device: Optional[Union[torch.device, str]] = None,
    dtype: Optional[torch.dtype] = None,
)-> Tuple[torch.device, torch.dtype]:
    """
    Format the device and dtype for PyTorch.

    Args:
        device (Optional[Union[torch.device, str]]): The device to use. If None, defaults to CPU.
        dtype (Optional[torch.dtype]): The data type to use. If None, defaults to float32.

    Returns:
        tuple[torch.device, torch.dtype]: The formatted device and dtype.
    """
    if device is None:
        device = torch.device("cpu")
    elif isinstance(device, str):
        device = torch.device(device)
        if device.index is None and device.type != "cpu":
            device = torch.device(device.type, 0)
    dtype = default(dtype, torch.float32)
    return device, dtype

def clean_up_memory():
    """
    Clean up the memory by calling garbage collector and emptying the cache.
    """
    gc.collect()
    torch.cuda.empty_cache()

def print_gpu_memory(prefix="",device="cuda:1"):
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    print(f"{prefix}Allocated: {allocated / 1024**2:.2f} MB, Reserved: {reserved / 1024**2:.2f} MB")
