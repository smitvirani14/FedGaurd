"""Shared model definition used by every node and by the centralized baseline.

FraudNet is deliberately tiny (30 -> 64 -> 32 -> 1) so it trains on CPU in
seconds: free-tier cloud VMs (1 vCPU, ~1 GB RAM) must be able to run a client
without swapping to death, and Opacus per-sample gradient bookkeeping keeps a
full copy of the activations in memory.
"""

import torch
import torch.nn as nn

DEVICE = torch.device("cpu")  # never MPS: Opacus per-sample grads are CPU-only here


class FraudNet(nn.Module):
    """Small MLP binary classifier for credit-card fraud.

    Ends in Sigmoid so outputs are probabilities used with nn.BCELoss
    (as specified for this project).
    """

    def __init__(self, input_dim: int = 30, hidden1: int = 64, hidden2: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_model(input_dim: int = 30) -> FraudNet:
    """Construct a fresh FraudNet already moved to the CPU device."""
    return FraudNet(input_dim=input_dim).to(DEVICE)


def predict_probs(model: nn.Module, X, batch_size: int = 4096) -> list:
    """Run inference in mini-batches and return a flat list of fraud probabilities."""
    import numpy as np

    model.eval()
    X_arr = np.asarray(X, dtype=np.float32)
    preds = []
    with torch.no_grad():
        for start in range(0, len(X_arr), batch_size):
            batch = torch.from_numpy(X_arr[start:start + batch_size]).to(DEVICE)
            preds.append(model(batch).cpu().numpy().ravel())
    return np.concatenate(preds).tolist() if preds else []
