import time

import torch


class CudaRuntimeProfiler:
    def __init__(self, description):
        self.description = description
        self.last_time = 0.0

    def __enter__(self):
        torch.cuda.synchronize()
        self.last_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        torch.cuda.synchronize()
        duration = time.time() - self.last_time
        print(f"{self.description}: {duration:.3f}s")


class CpuRuntimeProfiler:
    def __init__(self, description):
        self.description = description
        self.last_time = 0.0

    def __enter__(self):
        self.last_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        duration = time.time() - self.last_time
        print(f"{self.description}: {duration:.3f}s")


def profile_cpu_runtime(description):
    """Profile CPU runtime.

    Usage:

    ```
    with profile_cpu_runtime("call function"):
        outputs = function_call(inputs)
    ```

    Args:
        description (str): The description string for the code.

    Returns:
        A CpuRuntimeProfiler object.
    """
    return CpuRuntimeProfiler(description)


def profile_cuda_runtime(description):
    """Profile Cuda runtime.

    Usage:

    ```
    with profile_cuda_runtime("call function"):
        outputs = function_call(inputs)
    ```

    The CUDA device is synchronized during profiling.

    Args:
        description (str): The description string for the code.

    Returns:
        A CudaRuntimeProfiler object.
    """
    return CudaRuntimeProfiler(description)
