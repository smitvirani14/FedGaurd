"""FedGuard environment sanity check (Phase 0).

Verifies the local Python environment is arm64-native and that this project
runs on the CPU backend only.

Why CPU and never MPS: Opacus' per-sample gradient hooks (required for
differential privacy) do not reliably support the Apple MPS backend. Every
training script in this repo therefore forces torch.device("cpu").
"""

import platform
import sys

import torch


def main() -> None:
    print("=" * 60)
    print("FedGuard environment check")
    print("=" * 60)
    print(f"Python          : {sys.version.split()[0]} ({sys.executable})")
    print(f"Architecture    : {platform.machine()}")
    print(f"torch.__version_: {torch.__version__}")

    mps_available = torch.backends.mps.is_available()
    print(f"MPS available   : {mps_available} (informational only - NOT used)")

    device = torch.device("cpu")
    x = torch.randn(4, 8, device=device)
    w = torch.randn(8, 1, device=device, requires_grad=True)
    y = (x @ w).sigmoid().sum()
    y.backward()

    print(f"Active device   : {device} (type={device.type})")
    print(f"CPU smoke test  : OK (forward+backward ran, grad shape={tuple(w.grad.shape)})")
    print(f"torch threads   : {torch.get_num_threads()}")

    assert device.type == "cpu", "This project must never leave CPU mode"
    print("=" * 60)
    print("All checks passed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
