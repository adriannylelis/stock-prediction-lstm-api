"""Reproducibility utilities for setting random seeds.

This module provides utilities to set random seeds across different libraries
(PyTorch, NumPy, Python random) to ensure reproducible results.
"""

import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across all libraries.

    Sets seeds for:
    - Python's random module
    - NumPy
    - PyTorch (CPU and CUDA)

    This ensures that the same results are obtained across multiple runs,
    which is crucial for:
    - Debugging
    - Comparing different model configurations
    - Scientific reproducibility

    Args:
        seed: Random seed value. Common choices: 42, 0, 2024, etc.

    Example:
        >>> set_seed(42)
        >>> # Now all random operations will be reproducible
        >>> tensor = torch.randn(10, 10)  # Same values every time

    Note:
        For complete reproducibility, also set:
        - torch.backends.cudnn.deterministic = True
        - torch.backends.cudnn.benchmark = False
        However, this may impact performance.
    """
    # Python random module
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)

    # CUDA (GPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # For complete reproducibility (may reduce performance)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def set_deterministic_mode(enabled: bool = True) -> None:
    """Enable or disable deterministic mode for PyTorch operations.

    When enabled, PyTorch operations will behave deterministically where possible.
    This is useful for debugging and reproducibility, but may reduce performance.

    Args:
        enabled: If True, enable deterministic mode. If False, disable it.

    Warning:
        Enabling deterministic mode may slow down training, especially on GPU.
        Use only when reproducibility is critical.
    """
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = enabled
        torch.backends.cudnn.benchmark = not enabled


def get_random_state() -> dict:
    """Get the current random state from all libraries.

    Returns:
        Dictionary containing random states from Python, NumPy, and PyTorch.

    Example:
        >>> state = get_random_state()
        >>> # Do some random operations
        >>> restore_random_state(state)  # Restore to previous state
    """
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }

    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state()

    return state


def restore_random_state(state: dict) -> None:
    """Restore random state from a previously saved state.

    Args:
        state: Dictionary of random states (from get_random_state()).
    """
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])

    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state(state["torch_cuda"])
