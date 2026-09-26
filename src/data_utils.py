"""Data loading, preprocessing and cross-node partitioning for FedGuard.

Pipeline:
    raw creditcard.csv  ->  preprocess (scale Time/Amount)
                         ->  split_for_nodes (non-IID, 70/30 fraud skew)
                         ->  data/node1_partition.csv  (simulated AWS node)
                             data/node2_partition.csv  (simulated GCP node)

Each partition is self-contained (features + label) because the exact same
file is later copied onto a real cloud VM, where the node must be able to
train without ever touching the other node's data.

If the Kaggle credentials are missing or the download fails, a synthetic
dataset with the identical schema (30 numeric features, ~0.17% fraud) is
generated so the whole pipeline stays testable end-to-end.
"""

import os
import zipfile

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
RAW_CSV = os.path.join(RAW_DIR, "creditcard.csv")
NODE1_CSV = os.path.join(DATA_DIR, "node1_partition.csv")
NODE2_CSV = os.path.join(DATA_DIR, "node2_partition.csv")

FEATURE_COLS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
LABEL_COL = "Class"
N_FEATURES = len(FEATURE_COLS)  # 30

KAGGLE_DATASET_URL = "https://www.kaggle.com/api/v1/datasets/download/mlg-ulb/creditcardfraud"


def load_env():
    """Load `.env` from the repo root. Credentials are never hardcoded anywhere."""
    load_dotenv(os.path.join(ROOT_DIR, ".env"))


# --------------------------------------------------------------------------- #
# Dataset acquisition
# --------------------------------------------------------------------------- #
def download_from_kaggle() -> bool:
    """Download the official 'Credit Card Fraud Detection' dataset via the
    Kaggle REST API using credentials from `.env`. Returns True on success."""
    import requests

    load_env()
    username = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY")
    if not username or not key:
        print("[data] KAGGLE_USERNAME / KAGGLE_KEY not set in .env - skipping download")
        return False

    print("[data] Downloading creditcard.csv from Kaggle ...")
    try:
        resp = requests.get(KAGGLE_DATASET_URL, auth=(username, key), stream=True, timeout=300)
        if resp.status_code != 200:
            print(f"[data] Kaggle download failed (HTTP {resp.status_code})")
            return False

        os.makedirs(RAW_DIR, exist_ok=True)
        zip_path = os.path.join(RAW_DIR, "creditcard.zip")
        with open(zip_path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)

        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(RAW_DIR)
        os.remove(zip_path)

        if os.path.exists(RAW_CSV):
            print(f"[data] Saved {RAW_CSV}")
            return True
        print("[data] Zip did not contain creditcard.csv")
        return False
    except Exception as exc:  # network / auth errors must never block the project
        print(f"[data] Kaggle download error: {exc}")
        return False


def generate_synthetic(rows: int = 284807, fraud_rate: float = 0.001727, seed: int = 42) -> pd.DataFrame:
    """Build a stand-in dataset with the real schema (Time, V1..V28, Amount, Class).

    Fraud rows get mean shifts on a handful of the V-features so the class is
    learnable (the pipeline must reach AUC > 0.85 to be a meaningful test),
    while preserving the extreme class imbalance of the real data.
    """
    rng = np.random.default_rng(seed)
    n_fraud = max(1, int(round(rows * fraud_rate)))
    n_legit = rows - n_fraud

    data = {
        "Time": rng.uniform(0, 172800, size=rows),
        "Amount": rng.lognormal(mean=3.0, sigma=1.2, size=rows),
        "Class": np.array([1] * n_fraud + [0] * n_legit),
    }
    for i in range(1, 29):
        data[f"V{i}"] = rng.normal(0.0, 1.0, size=rows)

    df = pd.DataFrame(data)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    fraud_mask = df[LABEL_COL] == 1
    shifts = {"V1": -2.0, "V2": 1.8, "V3": -1.6, "V7": 1.4, "V10": -1.7, "V14": -2.0}
    for col, shift in shifts.items():
        df.loc[fraud_mask, col] = df.loc[fraud_mask, col] + shift
    df.loc[fraud_mask, "Amount"] = df.loc[fraud_mask, "Amount"] * 3.0

    return df


def ensure_dataset() -> str:
    """Make sure `data/raw/creditcard.csv` exists, trying Kaggle then synthetic."""
    if os.path.exists(RAW_CSV):
        print(f"[data] Using existing dataset: {RAW_CSV}")
        return RAW_CSV

    print("[data] WARNING: data/raw/creditcard.csv not found.")
    if download_from_kaggle() and os.path.exists(RAW_CSV):
        return RAW_CSV

    print("[data] WARNING: falling back to a SYNTHETIC placeholder dataset "
          "(same schema, ~0.17% fraud). Replace with the real Kaggle CSV when "
          "credentials are available.")
    df = generate_synthetic()
    os.makedirs(RAW_DIR, exist_ok=True)
    df.to_csv(RAW_CSV, index=False)
    print(f"[data] Wrote synthetic dataset to {RAW_CSV} ({len(df)} rows)")
    return RAW_CSV


# --------------------------------------------------------------------------- #
# Core API required by the plan
# --------------------------------------------------------------------------- #
def load_data(path: str = None) -> pd.DataFrame:
    """Load the raw credit-card dataset into a DataFrame."""
    if path is None:
        path = ensure_dataset()
    df = pd.read_csv(path)
    missing = [c for c in FEATURE_COLS + [LABEL_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")
    return df


def preprocess(df: pd.DataFrame):
    """Scale the `Time` and `Amount` columns (V1..V28 are already PCA-normalised).

    The scaler is fitted once on the full dataset so both nodes share identical
    feature scaling - required for their weight updates to be averaged meaningfully.
    Real multi-institution deployments would fit scalers locally; see the
    limitations section of report/comparison_report.md.

    Returns (X, y) where X is a 30-column DataFrame and y is the Class label.
    """
    X = df[FEATURE_COLS].copy()
    y = df[LABEL_COL].astype(int)

    scaler = StandardScaler()
    X[["Time", "Amount"]] = scaler.fit_transform(X[["Time", "Amount"]])
    return X, y


def split_for_nodes(X: pd.DataFrame, y: pd.Series, strategy: str = "non_iid",
                    node1_fraud_share: float = 0.7, node1_normal_share: float = 0.6):
    """Split the dataset into two node partitions.

    strategy="non_iid" simulates two genuinely different institutions:
      - Node 1 (AWS,  "credit-card transactions"): 60% of all rows, 70% of all fraud
      - Node 2 (GCP,  "bank logs")              : 40% of all rows, 30% of all fraud

    Different sizes AND different fraud ratios => FedAvg has real non-IID work
    to do, and the two nodes end up with different sample counts (n1, n2) which
    is exactly what Phase 5's manual weighted average depends on.

    Partitions are written to data/node1_partition.csv and data/node2_partition.csv.
    """
    if strategy != "non_iid":
        raise ValueError(f"Unsupported strategy: {strategy}")

    frame = X.copy()
    frame[LABEL_COL] = y.values

    fraud_idx = frame.index[frame[LABEL_COL] == 1]
    normal_idx = frame.index[frame[LABEL_COL] == 0]

    rng = np.random.default_rng(42)
    fraud_idx = rng.permutation(fraud_idx)
    normal_idx = rng.permutation(normal_idx)

    n1_fraud = int(round(len(fraud_idx) * node1_fraud_share))
    n1_normal = int(round(len(normal_idx) * node1_normal_share))

    node1 = frame.loc[np.concatenate([fraud_idx[:n1_fraud], normal_idx[:n1_normal]])]
    node2 = frame.loc[np.concatenate([fraud_idx[n1_fraud:], normal_idx[n1_normal:]])]

    node1 = node1.sample(frac=1.0, random_state=42).reset_index(drop=True)
    node2 = node2.sample(frac=1.0, random_state=43).reset_index(drop=True)

    node1.to_csv(NODE1_CSV, index=False)
    node2.to_csv(NODE2_CSV, index=False)

    print(f"[data] Node 1 (AWS): {len(node1):>7,} rows, fraud ratio {node1[LABEL_COL].mean():.4%} -> {NODE1_CSV}")
    print(f"[data] Node 2 (GCP): {len(node2):>7,} rows, fraud ratio {node2[LABEL_COL].mean():.4%} -> {NODE2_CSV}")
    print(f"[data] Fraud-ratio skew: {node1[LABEL_COL].mean() / max(node2[LABEL_COL].mean(), 1e-9):.2f}x "
          f"(Node 1 vs Node 2), total rows conserved: {len(node1) + len(node2) == len(frame)}")

    return (node1[FEATURE_COLS], node1[LABEL_COL], node2[FEATURE_COLS], node2[LABEL_COL])


def prepare_partitions():
    """Create data/node1_partition.csv + data/node2_partition.csv from the raw CSV.

    Called by the centralized baseline (Phase 1) and automatically by
    `load_partition` if the partitions don't exist yet.
    Returns (X1, y1, X2, y2).
    """
    df = load_data()
    X, y = preprocess(df)
    return split_for_nodes(X, y, strategy="non_iid")


def load_partition(partition: str):
    """Load one node's partition CSV ('node1' or 'node2') -> (X, y).

    This is what every Flower client calls: a node only ever reads its own file.
    """
    path = NODE1_CSV if partition == "node1" else NODE2_CSV if partition == "node2" else None
    if path is None:
        raise ValueError("partition must be 'node1' or 'node2'")
    if not os.path.exists(path):
        print(f"[data] {os.path.basename(path)} missing - preparing partitions first ...")
        prepare_partitions()

    df = pd.read_csv(path)
    return df[FEATURE_COLS], df[LABEL_COL].astype(int)


def class_weights(y: pd.Series) -> float:
    """Positive-class weight for BCELoss so 0.17% fraud isn't ignored.

    w_pos = n_neg / n_pos makes the loss class-balanced; negatives keep weight 1.
    The identical weighting is used for the centralized and federated runs so
    the comparison stays fair.
    """
    y = np.asarray(y)
    n_pos = max(int((y == 1).sum()), 1)
    n_neg = int((y == 0).sum())
    return n_neg / n_pos


def stratified_split(X, y, test_size=0.2, random_state=42):
    """Stratified train/test split (keeps the fraud ratio in both halves)."""
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
