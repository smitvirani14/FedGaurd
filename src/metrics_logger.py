"""Uniform round-by-round metrics logging for every phase of FedGuard.

All phases (centralized baseline, federated, federated+DP) must report through
`log_round` so `results/results.csv` keeps one schema that both the Streamlit
dashboard and the final report can rely on.
"""

import csv
import os
from datetime import datetime, timezone

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
DEFAULT_CSV = os.path.join(RESULTS_DIR, "results.csv")

FIELDS = ["timestamp", "round", "source", "accuracy", "precision", "recall", "f1", "auc", "epsilon"]


def compute_metrics(y_true, y_prob, threshold=0.5):
    """Turn raw fraud probabilities into the metric dict used by every phase.

    Fraud is a rare-class problem (~0.17% positive): accuracy alone is
    meaningless (a model predicting "not fraud" every time scores 99.83%).
    Recall and ROC-AUC are the numbers that actually matter - they measure
    whether fraud is caught at all, and how well the score separates classes.
    """
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true = [int(v) for v in y_true]
    y_prob = [float(v) for v in y_prob]
    y_pred = [1 if p >= threshold else 0 for p in y_prob]

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else 0.5,
    }


def _round(value, digits=6):
    if value is None:
        return ""
    return round(float(value), digits)


def log_round(round_num, source, accuracy, precision, recall, f1, auc, epsilon=None, results_path=DEFAULT_CSV):
    """Append one evaluation row to the results CSV.

    Args:
        round_num: federated round number (0 is used for the centralized baseline).
        source:    tag identifying the run, e.g. "centralized", "federated",
                   "federated_dp", "federated_nodp".
        accuracy / precision / recall / f1 / auc: sklearn metrics on the eval split.
        epsilon:   differential-privacy budget spent so far (None when DP is off).
        results_path: override the target CSV (used to write per-run extracts).
    """
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    write_header = not os.path.exists(results_path) or os.path.getsize(results_path) == 0

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "round": round_num,
        "source": source,
        "accuracy": _round(accuracy),
        "precision": _round(precision),
        "recall": _round(recall),
        "f1": _round(f1),
        "auc": _round(auc),
        "epsilon": _round(epsilon, 4),
    }

    with open(results_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    return row


def reset_results(results_path=DEFAULT_CSV):
    """Delete an existing results file so a fresh run starts from a clean slate."""
    if os.path.exists(results_path):
        os.remove(results_path)


def filter_results(source, results_path=DEFAULT_CSV, out_path=None):
    """Return (and optionally save) only the rows whose `source` column matches."""
    if not os.path.exists(results_path):
        return []
    with open(results_path, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["source"] == source]

    if out_path and rows:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
    return rows
