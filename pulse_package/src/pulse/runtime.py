'''Keep device setup and backend checks separate from data, embedding, and clustering logic.'''

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

from pulse.config import RuntimeConfig


ResolvedDeviceType = Literal["cpu", "cuda"]


@dataclass
class RuntimeInfo:
    """
    Resolved runtime settings for a pipeline run.

    Attributes:
        requested_device: Device string requested in configuration.
        device_type: General device type used by the pipeline.
        embedding_device: Device string passed to the embedding model.
        cuda_visible_devices: Value assigned to CUDA_VISIBLE_DEVICES, if any.
        cuda_available: Whether CUDA is available through PyTorch.
    """

    requested_device: str
    device_type: ResolvedDeviceType
    embedding_device: str
    cuda_visible_devices: str | None
    cuda_available: bool


def configure_runtime(config: RuntimeConfig) -> RuntimeInfo:
    """
    Resolve and apply runtime settings.

    Args:
        config: Runtime configuration.

    Returns:
        Resolved runtime information.
    """

    requested_device = config.device.strip()

    if requested_device == "cpu":
        return RuntimeInfo(
            requested_device=requested_device,
            device_type="cpu",
            embedding_device="cpu",
            cuda_visible_devices=None,
            cuda_available=_torch_cuda_available(),
        )

    if requested_device.startswith("cuda"):
        cuda_visible_devices = _extract_cuda_visible_devices(requested_device)

        if cuda_visible_devices is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

        cuda_available = _torch_cuda_available()

        if not cuda_available:
            raise RuntimeError(
                "CUDA was requested, but PyTorch does not report CUDA as available. "
                "Use runtime.device='cpu' or run on a CUDA-enabled environment."
            )

        return RuntimeInfo(
            requested_device=requested_device,
            device_type="cuda",
            embedding_device=_embedding_device_from_cuda_request(requested_device),
            cuda_visible_devices=cuda_visible_devices,
            cuda_available=True,
        )

    raise ValueError(
        "Unsupported runtime.device value. Expected 'cpu', 'cuda', 'cuda:0', "
        "'cuda:1', or a comma-separated CUDA request such as 'cuda:0,1'."
    )


def require_backend_available(
    backend: str,
    device_type: ResolvedDeviceType,
) -> None:
    """
    Validate that the requested backend is available.

    Args:
        backend: Backend name such as ``sklearn`` or ``cuml``.
        device_type: Resolved device type.

    Raises:
        RuntimeError: If the requested backend cannot be used.
    """

    if backend == "sklearn":
        _require_import("sklearn", "scikit-learn is required for backend='sklearn'.")
        return

    if backend == "cuml":
        if device_type != "cuda":
            raise RuntimeError(
                "backend='cuml' requires a CUDA runtime. "
                "Use backend='sklearn' when running on CPU."
            )

        _require_import(
            "cuml",
            "cuML is required for backend='cuml'. Install CUDA dependencies first.",
        )
        return

    raise ValueError(f"Unsupported backend: {backend}")


def _extract_cuda_visible_devices(device: str) -> str | None:
    """
    Extract CUDA device IDs from a configured device string.
    """

    if device == "cuda":
        return None

    if not device.startswith("cuda:"):
        return None

    return device.removeprefix("cuda:")


def _embedding_device_from_cuda_request(device: str) -> str:
    """
    Return the device string used by SentenceTransformer.
    """

    if device == "cuda":
        return "cuda"

    visible_devices = _extract_cuda_visible_devices(device)

    if not visible_devices:
        return "cuda"

    first_device = visible_devices.split(",", maxsplit=1)[0].strip()

    if not first_device:
        return "cuda"

    return "cuda:0"


def _torch_cuda_available() -> bool:
    """
    Check CUDA availability through PyTorch.
    """

    try:
        import torch
    except ImportError:
        return False

    return bool(torch.cuda.is_available())


def _require_import(
    module_name: str,
    message: str,
) -> None:
    """
    Require that a module can be imported.
    """

    try:
        __import__(module_name)
    except ImportError as exc:
        raise RuntimeError(message) from exc