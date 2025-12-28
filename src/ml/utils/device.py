"""Device detection and management utilities.

This module provides utilities for automatically detecting and configuring
the appropriate device (CPU or CUDA GPU) for PyTorch computations.
"""

import torch


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Detect and return the best available device for PyTorch.

    Automatically detects if CUDA (NVIDIA GPU) is available and returns
    the appropriate device. Provides informative logging about the detected hardware.

    Args:
        prefer_cuda: If True, use CUDA if available. If False, force CPU usage.

    Returns:
        torch.device: Either torch.device("cuda") or torch.device("cpu").

    Example:
        >>> device = get_device()
        >>> model = model.to(device)
        >>> tensor = torch.randn(10, 10).to(device)
    """
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        return device
    else:
        device = torch.device("cpu")
        return device


def print_device_info(device: torch.device) -> None:
    """Print detailed information about the current device.

    Args:
        device: The PyTorch device to get information about.
    """
    print("=" * 70)
    print("🖥️  HARDWARE CONFIGURATION")
    print("=" * 70)
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print("\n🚀 TRAINING WILL USE GPU (10-50x faster!)")
    else:
        print("\n⚠️  GPU NOT DETECTED - using CPU")
        print("   For faster training, consider:")
        print("   - Using a machine with NVIDIA GPU")
        print("   - Using Google Colab with GPU runtime")

    print("=" * 70)


def get_device_name(device: torch.device) -> str:
    """Get a human-readable name for the device.

    Args:
        device: The PyTorch device.

    Returns:
        String name of the device.
    """
    if device.type == "cuda":
        return torch.cuda.get_device_name(0)
    return "CPU"


def set_device_for_model(
    model: torch.nn.Module, device: torch.device | None = None
) -> tuple[torch.nn.Module, torch.device]:
    """Move model to the appropriate device.

    Args:
        model: PyTorch model to move.
        device: Target device. If None, automatically detects best device.

    Returns:
        Tuple of (model on device, device used).

    Example:
        >>> model = StockLSTM()
        >>> model, device = set_device_for_model(model)
    """
    if device is None:
        device = get_device()

    model = model.to(device)
    return model, device
