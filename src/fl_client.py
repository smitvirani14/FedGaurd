"""Phase 2 - Flower client: one process per simulated cloud node.

This file is intentionally CLOUD-AGNOSTIC (no AWS/GCP-specific code): the very
same script runs as a local process during development and is later copied
verbatim onto the EC2 and Compute Engine VMs in Phase 4. The only thing that
changes is the `--partition` argument and the `--server_address` it points at.

    local : python src/fl_client.py --partition node1 --server_address localhost:8080
    cloud : python fl_client.py  --partition node1 --server_address <orchestrator>:8080

PRIVACY NOTE (important for the report/viva): a node never transmits a single
row of raw transaction data. The only things that cross the client->server
boundary are (a) model weight arrays from `get_parameters`/`fit` and (b)
scalar metrics from `evaluate`. See the assertion in `fit` below.
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import flwr as fl  # noqa: E402

from data_utils import (  # noqa: E402
    N_FEATURES,
    class_weights,
    load_env,
    load_partition,
    stratified_split,
)
from metrics_logger import compute_metrics  # noqa: E402
from model import DEVICE, build_model, predict_probs  # noqa: E402

# ---- local training hyper-parameters (identical on every node) ----
LOCAL_EPOCHS = 1
BATCH_SIZE = 512
LR = 1e-3

# Phase 3 will introduce DP here; kept as a module-level toggle so the demo can
# flip privacy on/off between runs and compare the results.
USE_DP = False


def _weights_tensor(y_train, pos_weight: float) -> torch.Tensor:
    """Class-balanced per-sample loss weights (same scheme as the baseline)."""
    y = np.asarray(y_train).reshape(-1)
    w = torch.full((len(y), 1), 1.0, dtype=torch.float32)
    w[y == 1] = pos_weight
    return w


class FlowerClient(fl.client.NumPyClient):
    """Trains FraudNet on ONE node's partition only."""

    def __init__(self, partition: str):
        load_env()
        self.partition = partition

        X, y = load_partition(partition)
        X_train, X_test, y_train, y_test = stratified_split(X, y, test_size=0.2, random_state=42)

        self.X_train = X_train.to_numpy(dtype=np.float32)
        self.y_train = y_train.to_numpy(dtype=np.float32)
        self.X_test = X_test.to_numpy(dtype=np.float32)
        self.y_test = y_test.to_numpy(dtype=np.float32)

        self.pos_weight = class_weights(y_train)
        self.loader = DataLoader(
            TensorDataset(
                torch.from_numpy(self.X_train),
                torch.from_numpy(self.y_train).view(-1, 1),
                _weights_tensor(self.y_train, self.pos_weight),
            ),
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
        )

        self.model = build_model(input_dim=N_FEATURES)
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)

        print(f"[client:{partition}] loaded {len(self.X_train):,} train / "
              f"{len(self.X_test):,} test rows, pos_weight={self.pos_weight:.1f}", flush=True)

    # ------------------------------------------------------------------ #
    def get_parameters(self, config) -> list[np.ndarray]:
        """Return the current local weights as plain numpy arrays."""
        return [v.detach().cpu().numpy() for v in self.model.state_dict().values()]

    def _set_parameters(self, parameters: list[np.ndarray]) -> None:
        keys = list(self.model.state_dict().keys())
        if len(keys) != len(parameters):
            raise ValueError(f"parameter count mismatch: {len(parameters)} vs {len(keys)}")
        state = {k: torch.tensor(v, dtype=torch.float32) for k, v in zip(keys, parameters)}
        self.model.load_state_dict(state)

    # ------------------------------------------------------------------ #
    def fit(self, parameters, config):
        """Receive global weights, train locally, return updated weights."""
        self._set_parameters(parameters)

        self.model.train()
        for _ in range(int(config.get("local_epochs", LOCAL_EPOCHS))):
            for xb, yb, wb in self.loader:
                xb, yb, wb = xb.to(DEVICE), yb.to(DEVICE), wb.to(DEVICE)
                self.optimizer.zero_grad()
                out = self.model(xb)
                loss = (self.criterion(out, yb) * wb).mean()
                loss.backward()
                self.optimizer.step()

        weights = self.get_parameters(config)

        # ---- privacy invariant, asserted at runtime for the report ----
        # Everything sent back to the orchestrator is model weights + scalars.
        # No raw sample from this node's CSV is ever placed in this return value.
        # The size check makes this concrete: FraudNet has ~4k parameters while
        # one row of node1's raw data would already be 31 floats x 170k rows.
        assert all(isinstance(w, np.ndarray) for w in weights), \
            "fit() may only return numpy weight arrays - never raw data"
        total_payload = int(sum(w.size for w in weights))
        assert total_payload < 50_000, \
            f"payload of {total_payload} values is far larger than the model - data leak?"
        print(f"[client:{self.partition}] fit() sent {total_payload} weight values "
              f"(0 raw rows)", flush=True)

        fit_metrics = {
            "train_samples": float(len(self.X_train)),
            "pos_weight": float(self.pos_weight),
        }
        if USE_DP:
            eps = self._dp_epsilon()
            if eps is not None:
                fit_metrics["epsilon"] = float(eps)
                print(f"[client:{self.partition}] local DP epsilon = {eps:.4f}", flush=True)

        return weights, len(self.X_train), fit_metrics

    def evaluate(self, parameters, config):
        """Evaluate the given global weights on this node's local test split."""
        self._set_parameters(parameters)
        y_prob = predict_probs(self.model, self.X_test)
        metrics = compute_metrics(self.y_test, y_prob)

        xb = torch.from_numpy(self.X_test)
        yb = torch.from_numpy(self.y_test).view(-1, 1)
        with torch.no_grad():
            loss = self.criterion(self.model(xb), yb).item()

        print(f"[client:{self.partition}] round-eval "
              f"acc={metrics['accuracy']:.4f} recall={metrics['recall']:.4f} "
              f"auc={metrics['auc']:.4f}", flush=True)
        return loss, len(self.X_test), metrics

    # Phase 3 replaces this with the real Opacus budget query.
    def _dp_epsilon(self):
        return None


def main() -> None:
    global LOCAL_EPOCHS

    parser = argparse.ArgumentParser(description="FedGuard Flower client")
    parser.add_argument("--partition", choices=["node1", "node2"], required=True,
                        help="which local CSV this node trains on")
    parser.add_argument("--server_address", "--server-address", dest="server_address",
                        default=None,
                        help="orchestrator host:port (default: localhost from .env)")
    parser.add_argument("--local-epochs", type=int, default=LOCAL_EPOCHS)
    args = parser.parse_args()

    LOCAL_EPOCHS = args.local_epochs

    load_env()
    server_address = args.server_address or f"localhost:{os.getenv('FLOWER_SERVER_PORT', '8080')}"

    print(f"[client:{args.partition}] connecting to orchestrator at {server_address}", flush=True)

    client = FlowerClient(args.partition)

    try:
        from flwr.client import start_numpy_client  # older Flower layouts
    except ImportError:
        from flwr.compat.client.app import start_numpy_client  # Flower 1.38+

    start_numpy_client(server_address=server_address, client=client)


if __name__ == "__main__":
    main()
