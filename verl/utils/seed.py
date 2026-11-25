import sys
import os
import random
import importlib.util

def seed_everything(seed: int, deterministic: bool = False, warn_only: bool = False):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch` and/or `tf` (if installed).

    Args:
        seed (`int`):
            The seed to set.
        deterministic (`bool`, *optional*, defaults to `False`):
            Whether to use deterministic algorithms where available. Can slow down training.
        warn_only (`bool`, *optional*, defaults to `False`):
            Whether to only warn about the use of non-deterministic algorithms instead of raising an error.
            This is useful for debugging purposes, but should be set to `False` in production code.
    """
    random.seed(seed)
    
    # numpy
    if importlib.util.find_spec('numpy') is not None:
        import numpy
        print(f"Setting numpy seed to {seed}")
        numpy.random.seed(seed)
        
    # torch
    if importlib.util.find_spec('torch') is not None:
        import torch
        print(f"Setting torch seed to {seed}")
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
        
        # torch.cuda
        if torch.cuda.is_available():
            print(f"Setting torch cuda seed to {seed}")
            torch.cuda.manual_seed_all(seed)
            
        if deterministic:
            torch.use_deterministic_algorithms(True)
            torch.use_deterministic_algorithms(True, warn_only=warn_only)

            # Enable CUDNN deterministic mode
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.enable = False
            
    # ^^ safe to call this function even if cuda is not available
    if deterministic:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        os.environ["ASCEND_LAUNCH_BLOCKING"] = "1"
        os.environ['NCCL_DETERMINISTIC'] = "1"
        os.environ['LCCL_DETERMINISTIC'] = "1"
        os.environ["FLASH_ATTENTION_DETERMINISTIC"] = "1"
        os.environ['CLOSE_MATMUL_K_SHIFT'] = "1"
        
if __name__ == "__main__":
    seed = 42
    deterministic = True
    warn_only = False
    seed_everything(seed, deterministic, warn_only)
    print("Seed set to:", seed)
    print("Deterministic mode:", deterministic)
    print("Warn only mode:", warn_only)