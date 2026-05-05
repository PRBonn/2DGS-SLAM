"""PyTorch C++ log level must be set before the first ``import torch`` in a process."""

import os


def ensure_before_torch_import() -> None:
    """Raise c10 minimum level to ERROR so CUDA IPC bookkeeping spam is hidden.

    Emits from native code (e.g. ``CudaIPCTypes.cpp``) when many GPU tensors cross
    ``multiprocessing`` queues. Not visible to :mod:`warnings`.
    """
    os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")
