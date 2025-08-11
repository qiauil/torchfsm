import torch
import gc

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
