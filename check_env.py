"""Inspect local Python, PyTorch, CUDA, and related runtime dependencies."""

from __future__ import annotations

import importlib
import platform
import sys


def print_header(title: str) -> None:
    print(f"\n== {title} ==")


def safe_import(name: str):
    try:
        module = importlib.import_module(name)
        return module, None
    except Exception as exc:  # pragma: no cover - best-effort diagnostics
        return None, exc


def main() -> None:
    print_header("Python")
    print(f"executable: {sys.executable}")
    print(f"version: {sys.version.splitlines()[0]}")
    print(f"platform: {platform.platform()}")

    print_header("Packages")
    torch, torch_err = safe_import("torch")
    open_clip, open_clip_err = safe_import("open_clip")

    if torch is None:
        print(f"torch: MISSING ({torch_err})")
    else:
        print(f"torch: OK ({getattr(torch, '__version__', 'unknown version')})")

    if open_clip is None:
        print(f"open_clip: MISSING ({open_clip_err})")
    else:
        print(
            f"open_clip: OK ({getattr(open_clip, '__version__', 'version unknown')})"
        )

    print_header("CUDA")
    if torch is None:
        print("PyTorch unavailable, skipping CUDA and GPU checks.")
        return

    cuda_built = getattr(torch.version, "cuda", None)
    cuda_available = torch.cuda.is_available()
    print(f"torch built with CUDA: {cuda_built or 'NO'}")
    print(f"torch.cuda.is_available(): {cuda_available}")

    try:
        cuDNN = torch.backends.cudnn.version()
    except Exception:
        cuDNN = None
    print(f"cuDNN version: {cuDNN or 'unavailable'}")

    if not cuda_available:
        print("No CUDA device visible to PyTorch. CPU execution will be used.")
        return

    count = torch.cuda.device_count()
    print(f"GPU count: {count}")
    current = torch.cuda.current_device()
    print(f"current device index: {current}")

    for idx in range(count):
        props = torch.cuda.get_device_properties(idx)
        total_mem_gb = props.total_memory / (1024 ** 3)
        print(
            f"GPU {idx}: {props.name} | compute capability "
            f"{props.major}.{props.minor} | VRAM {total_mem_gb:.2f} GiB"
        )

    print_header("Tensor Test")
    try:
        x = torch.randn(2, 2, device="cuda")
        y = x @ x
        print("CUDA tensor operation: OK")
        print(f"sample result device: {y.device}")
        print(f"sample result shape: {tuple(y.shape)}")
    except Exception as exc:  # pragma: no cover - runtime-only diagnostics
        print(f"CUDA tensor operation: FAILED ({exc})")


if __name__ == "__main__":
    main()
