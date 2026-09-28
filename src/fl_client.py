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
from dp_engine import (  # noqa: E402
    DELTA,
    MAX_GRAD_NORM,
    NOISE_MULTIPLIER,
    get_privacy_spent,
    make_private,
    unwrap_private,
)
from metrics_logger import compute_metrics  # noqa: E402
from model import DEVICE, build_model, predict_probs  # noqa: E402

# ---- local training hyper-parameters (identical on every node) ----
LOCAL_EPOCHS = 1
BATCH_SIZE = 512
LR = 1e-3        # plain federated / centralized training

# DP-SGD needs a higher learning rate than plain training: the injected noise
# inflates Adam's second-moment estimate, which otherwise shrinks every step
# into nothing (measured: lr=1e-3 -> recall 0.00, lr=5e-3 -> recall 0.67 after
# the same 10 rounds at the same epsilon). Everything else - batch size, epochs,
# loss, class weights - stays identical so the comparison stays honest.
DP_LR = 5e-3

# Phase 3 privacy toggle: run with USE_DP = True (or --use-dp) to add DP noise
# before weights leave the node, and USE_DP = False (or --no-dp) for the plain
# federated baseline. The comparison of these two runs IS the project's
# privacy-utility tradeoff evidence, so it must be easy to flip in a demo.
USE_DP = False

# Phase 5: when set (--save-weights PATH), every fit() writes this node's local
# weights to disk so src/manual_fedavg.py can re-derive the server's aggregate
# by hand and prove it matches Flower's FedAvg tensor-for-tensor.
SAVE_WEIGHTS_PATH = None


def _save_weights(path: str, state_dict, num_examples: int, partition: str) -> None:
    """Persist this node's post-fit weights for Phase 5's manual FedAvg check.

    Only tensors and scalars are written - the same no-raw-data invariant that
    `fit` asserts over the network applies to disk too.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "state_dict": {k: v.detach().cpu() for k, v in state_dict.items()},
        "num_examples": int(num_examples),
        "partition": partition,
    }, path)
    print(f"[client:{partition}] saved local weights -> {path} "
          f"(n={num_examples})", flush=True)


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
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=DP_LR if USE_DP else LR)

        # DP state: the engine is created once and reused every round so its
        # accountant accumulates epsilon across the whole training run.
        self.privacy_engine = None
        self.epsilon = None

        print(f"[client:{partition}] loaded {len(self.X_train):,} train / "
              f"{len(self.X_test):,} test rows, pos_weight={self.pos_weight:.1f} "
              f"DP={'ON' if USE_DP else 'OFF'}", flush=True)

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
    def _train_local(self, model, optimizer, loader, epochs: int) -> None:
        """The one training loop shared by the plain and the DP-SGD path.

        Keeping them identical (same loss, same class weights, same epochs) is
        what makes the DP vs no-DP comparison meaningful.
        """
        model.train()
        for _ in range(epochs):
            for xb, yb, wb in loader:
                xb, yb, wb = xb.to(DEVICE), yb.to(DEVICE), wb.to(DEVICE)
                optimizer.zero_grad()
                out = model(xb)
                loss = (self.criterion(out, yb) * wb).mean()
                loss.backward()
                optimizer.step()

    def fit(self, parameters, config):
        """Receive global weights, train locally, return updated weights."""
        self._set_parameters(parameters)
        epochs = int(config.get("local_epochs", LOCAL_EPOCHS))
        # evaluate() left the module in eval() mode; Opacus refuses to wrap a
        # model that is not in training mode, so restore it before make_private.
        self.model.train()

        if USE_DP:
            # DP-SGD: clip every sample's gradient and add Gaussian noise
            # BEFORE the update, so the weights we return are already noisy.
            self.privacy_engine, private_model, private_optimizer, private_loader = make_private(
                model=self.model,
                optimizer=self.optimizer,
                data_loader=self.loader,
                noise_multiplier=NOISE_MULTIPLIER,
                max_grad_norm=MAX_GRAD_NORM,
                privacy_engine=self.privacy_engine,   # reuse -> cumulative epsilon
            )
            self._train_local(private_model, private_optimizer, private_loader, epochs)
            unwrap_private(private_model, private_optimizer)
            self.epsilon = get_privacy_spent(self.privacy_engine, DELTA)
            print(f"[client:{self.partition}] DP run: noise_multiplier={NOISE_MULTIPLIER}, "
                  f"max_grad_norm={MAX_GRAD_NORM}, cumulative epsilon={self.epsilon:.4f}",
                  flush=True)
        else:
            self._train_local(self.model, self.optimizer, self.loader, epochs)

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
        if USE_DP and self.epsilon is not None:
            fit_metrics["epsilon"] = float(self.epsilon)

        # Phase 5: keep the exact weights this node just sent, so
        # src/manual_fedavg.py can re-derive the server's aggregate by hand.
        if SAVE_WEIGHTS_PATH:
            _save_weights(SAVE_WEIGHTS_PATH, self.model.state_dict(),
                          len(self.X_train), self.partition)

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

    def _dp_epsilon(self):
        """Cumulative epsilon spent by this node's accountant (None when DP is off)."""
        return self.epsilon


def main() -> None:
    global LOCAL_EPOCHS, USE_DP, SAVE_WEIGHTS_PATH

    parser = argparse.ArgumentParser(description="FedGuard Flower client")
    parser.add_argument("--partition", choices=["node1", "node2"], required=True,
                        help="which local CSV this node trains on")
    parser.add_argument("--server_address", "--server-address", dest="server_address",
                        default=None,
                        help="orchestrator host:port (default: localhost from .env)")
    parser.add_argument("--local-epochs", type=int, default=LOCAL_EPOCHS)
    parser.add_argument("--use-dp", dest="use_dp", action="store_true", default=None,
                        help="force differential privacy ON for this run")
    parser.add_argument("--no-dp", dest="use_dp", action="store_false",
                        help="force differential privacy OFF for this run")
    parser.add_argument("--save-weights", dest="save_weights", default=None,
                        metavar="PATH",
                        help="Phase 5: write this node's local weights to PATH after every fit()")
    args = parser.parse_args()

    LOCAL_EPOCHS = args.local_epochs
    if args.use_dp is not None:
        USE_DP = args.use_dp
    SAVE_WEIGHTS_PATH = args.save_weights

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
