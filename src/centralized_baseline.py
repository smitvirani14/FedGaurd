"""Phase 1 - Centralized baseline (the accuracy benchmark for everything later).

Trains FraudNet on the FULL pooled dataset with no federation at all. Its
recall/AUC is the number the federated and federated+DP models are compared
against in report/comparison_report.md.

Usage:
    python src/centralized_baseline.py [--epochs 10] [--batch-size 512] [--lr 1e-3]
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_utils import (  # noqa: E402
    N_FEATURES,
    class_weights,
    load_data,
    prepare_partitions,
    preprocess,
    stratified_split,
)
from metrics_logger import compute_metrics, log_round  # noqa: E402
from model import DEVICE, build_model, predict_probs  # noqa: E402

RESULTS_JSON = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "centralized_baseline.json"
)


def train(X_train, y_train, epochs, batch_size, lr, pos_weight):
    model = build_model(input_dim=N_FEATURES)
    # Class-balanced BCELoss: with ~0.17% fraud an unweighted loss collapses to
    # "always predict not-fraud" (99.83% accuracy, 0% recall) - useless.
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    weights = torch.full((len(y_train), 1), 1.0, dtype=torch.float32)
    weights[y_train.to_numpy() == 1] = pos_weight

    dataset = TensorDataset(
        torch.from_numpy(X_train.to_numpy(dtype=np.float32)),
        torch.from_numpy(y_train.to_numpy(dtype=np.float32)).view(-1, 1),
        weights,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model.train()
    for epoch in range(1, epochs + 1):
        total, correct, running = 0, 0, 0.0
        for xb, yb, wb in loader:
            xb, yb, wb = xb.to(DEVICE), yb.to(DEVICE), wb.to(DEVICE)
            optimizer.zero_grad()
            out = model(xb)
            loss = (criterion(out, yb) * wb).mean()
            loss.backward()
            optimizer.step()

            running += loss.item() * len(xb)
            correct += ((out >= 0.5).float() == yb).sum().item()
            total += len(xb)
        print(f"[centralized] epoch {epoch:>2}/{epochs}  loss={running / total:.5f}  "
              f"batch_acc={correct / total:.4f}", flush=True)

    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="FedGuard centralized baseline")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    print("=" * 70)
    print("PHASE 1 - Centralized baseline")
    print("=" * 70)

    df = load_data()
    X, y = preprocess(df)
    print(f"[centralized] loaded {len(X):,} rows, fraud ratio {y.mean():.4%}")

    # Phase 1 data-prep deliverable: build the two non-IID node partitions that
    # Phase 2's federated clients will each load (70% of fraud -> Node 1).
    prepare_partitions()

    X_train, X_test, y_train, y_test = stratified_split(X, y, test_size=args.test_size)
    pos_weight = class_weights(y_train)
    print(f"[centralized] train={len(X_train):,}  test={len(X_test):,}  pos_weight={pos_weight:.1f}")

    model = train(X_train, y_train, args.epochs, args.batch_size, args.lr, pos_weight)

    y_prob = predict_probs(model, X_test)
    metrics = compute_metrics(y_test, y_prob)
    print("[centralized] test metrics:")
    for k, v in metrics.items():
        print(f"    {k:<10} {v:.4f}")

    os.makedirs(os.path.dirname(RESULTS_JSON), exist_ok=True)
    payload = {"source": "centralized", "epochs": args.epochs, "batch_size": args.batch_size,
               "lr": args.lr, "n_train": int(len(X_train)), "n_test": int(len(X_test)),
               "pos_weight": float(pos_weight), **metrics}
    with open(RESULTS_JSON, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[centralized] wrote {RESULTS_JSON}")

    log_round(0, "centralized",
              metrics["accuracy"], metrics["precision"], metrics["recall"],
              metrics["f1"], metrics["auc"])

    if metrics["auc"] <= 0.85:
        print(f"[centralized] WARNING: AUC {metrics['auc']:.4f} <= 0.85 target - "
              "check the dataset or training epochs.")
        sys.exit(1)
    print("[centralized] OK - AUC above the 0.85 target.")


if __name__ == "__main__":
    main()
