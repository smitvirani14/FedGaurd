# FedGuard

**Cross-cloud federated learning with differential privacy for credit-card fraud detection.**

Two institutions — one on AWS, one on GCP — jointly train a fraud-detection model without any transaction ever leaving the cloud that owns it. Each node trains locally on its own non-IID partition, ships only ~4,097 model weights to a Flower orchestrator, and (in the privacy-preserving configuration) adds calibrated DP-SGD noise before those weights are published. The project measures exactly what that privacy costs: recall drops from 0.842 to 0.768 while the run buys a formal guarantee of ε ≈ 1.14 at δ = 1e-5.

---

## Folder structure

```
├── Project.md                    # the full phase-by-phase build specification
├── requirements.txt              # pinned dependencies (Python 3.11)
├── .env.example                  # config template — copy to .env, never commit .env
│
├── src/
│   ├── check_env.py              # Phase 0  — environment sanity check
│   ├── data_utils.py             # Phase 1  — Kaggle download, preprocessing, non-IID split
│   ├── model.py                  # Phase 1  — FraudNet (30 → 64 → 32 → 1), CPU-only
│   ├── metrics_logger.py         # Phase 1  — one CSV schema for every phase
│   ├── centralized_baseline.py   # Phase 1  — the accuracy benchmark everything else compares to
│   ├── fl_client.py              # Phase 2/3 — Flower client (cloud-agnostic, DP toggle)
│   ├── fl_server.py              # Phase 2/4 — Flower orchestrator (FedAvg, binds 0.0.0.0)
│   ├── dp_engine.py              # Phase 3  — Opacus wrapper (make_private / get_privacy_spent)
│   └── manual_fedavg.py          # Phase 5  — FedAvg written by hand, verified vs Flower
│
├── dashboard/app.py              # Phase 6  — Streamlit results dashboard
├── report/comparison_report.md   # Phase 7  — the write-up (results, threat model, limits)
│
├── deployment/
│   ├── aws_setup.md              # Phase 4  — EC2 node1: launch, firewall, install, run
│   ├── gcp_setup.md              # Phase 4  — GCP orchestrator + node2
│   └── run_node.sh               # identical launcher used on both VMs
│
├── data/
│   ├── raw/creditcard.csv        # 284,807 rows (gitignored — download it yourself)
│   ├── node1_partition.csv       # AWS node's slice   (gitignored)
│   └── node2_partition.csv       # GCP node's slice   (gitignored)
│
└── results/
    ├── results.csv               # all local runs (centralized / federated / +DP)
    ├── cloud_results.csv         # the cross-cloud run
    ├── cloud_server.log          # orchestrator log incl. client source IPs
    ├── cloud_node1.log           # AWS client log
    ├── cloud_node2.log           # GCP client log
    ├── dp_summary.md             # Phase 3 privacy budget summary
    ├── federated_dp.csv          # DP run (10 rounds)
    ├── federated_nodp.csv        # non-DP run (10 rounds)
    └── centralized_baseline.json # Phase 1 baseline
```

---

## Setup

Requires **Python 3.11** (arm64 on Apple Silicon, x86_64 on the cloud VMs).

```bash
git clone https://github.com/smitvirani14/FedGaurd.git
cd FedGaurd

python3.11 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip

# torch from the CPU index first — avoids pulling ~2 GB of CUDA wheels
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

cp .env.example .env      # then edit: Kaggle keys for Phase 1, AWS/GCP IPs for Phase 4
python src/check_env.py   # must print "All checks passed."
```

### Dataset

`src/data_utils.py` resolves the dataset in this order:

1. `data/raw/creditcard.csv` if it already exists,
2. Kaggle download (`KAGGLE_USERNAME` / `KAGGLE_KEY` in `.env`),
3. otherwise a **synthetic placeholder** with the same schema (30 features,
   ~0.17% fraud) so the pipeline still runs end-to-end — clearly logged as a
   placeholder if it happens.

Manual download: [Credit Card Fraud Detection on Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud) → place `creditcard.csv` in `data/raw/`.

---

## Running each phase

All commands below assume the virtualenv is active and you are in the repo root.

### Phase 1 — Centralized baseline (do this first)

```bash
python src/centralized_baseline.py
# → results/centralized_baseline.json, data/node{1,2}_partition.csv, results/results.csv row 0
```

Expected: AUC > 0.95, recall ≈ 0.82. This number is the benchmark for everything else.

### Phase 2 — Local federated simulation (3 terminals)

```bash
# terminal 1 — orchestrator
python src/fl_server.py --num_rounds 10 --source federated

# terminal 2
python src/fl_client.py --partition node1

# terminal 3
python src/fl_client.py --partition node2
```

The server blocks until **both** clients connect (`min_available_clients=2`).

### Phase 3 — With differential privacy

```bash
python src/fl_server.py --num_rounds 10 --source federated_dp
python src/fl_client.py --partition node1 --use-dp      # terminal 2
python src/fl_client.py --partition node2 --use-dp      # terminal 3

# and again with --no-dp, --source federated_nodp, for the comparison
```

ε is logged every round; a summary lands in `results/dp_summary.md`.

### Phase 4 — Cross-cloud (AWS + GCP)

Follow `deployment/gcp_setup.md` (orchestrator + node2) and
`deployment/aws_setup.md` (node1), then start both nodes with the *same* script:

```bash
./run_node.sh node1 8.231.92.55 8080   # on the AWS VM
./run_node.sh node2 localhost  8080    # on the GCP VM (co-located with server)
```

Evidence of a real cross-cloud run lives in `results/cloud_server.log`:

```
[server] client connected: cid=... peer=ipv4:65.1.248.113:41032 connected=2
```

> The two VMs and their firewall rules cost $0 under the AWS/GCP free tiers —
> **stop them when you are not demoing** (commands at the bottom of each setup doc).

### Phase 5 — Manual FedAvg check

```bash
python src/fl_server.py --num_rounds 3 --source manual_check \
    --results_path results/manual_check.csv \
    --save-weights 'results/global_weights_{round}.pt'
python src/fl_client.py --partition node1 --save-weights results/node1_weights.pt
python src/fl_client.py --partition node2 --save-weights results/node2_weights.pt

python src/manual_fedavg.py
# → RESULT: PASS - manual FedAvg matches Flower's FedAvg (worst element diff 2.980e-08)
```

### Phase 6 — Dashboard

```bash
streamlit run dashboard/app.py
```

Opens charts of accuracy/recall/F1/AUC vs. round, ε vs. round, the
privacy–utility tradeoff panel, and an architecture explainer. Auto-refreshes
every 5 seconds while a run is writing to `results/results.csv`.

### Phase 7 — Report

`report/comparison_report.md` — problem statement, architecture, results
tables, threat model, limitations and future work, with every number pulled
from `results/`.

---

## Results at a glance

| Run | Accuracy | Recall | AUC | ε |
|---|---|---|---|---|
| Centralized baseline | 0.9994 | 0.8163 | 0.9793 | — |
| Federated, no DP | 0.9994 | 0.8423 | 0.9902 | — |
| Federated + DP | 0.9993 | 0.7675 | 0.9327 | 1.1377 |
| Federated, cross-cloud | 0.9994 | 0.8423 | 0.9915 | — |

Accuracy is nearly meaningless at a 0.17% fraud rate — recall, F1 and AUC are
the metrics that matter here.

---

## Configuration

Everything credential-shaped lives in `.env` (copied from `.env.example`) and is
read via `python-dotenv`. Nothing secret is ever hardcoded in `src/`.

| Variable | Used by |
|---|---|
| `KAGGLE_USERNAME`, `KAGGLE_KEY` | Phase 1 dataset download |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION` | Phase 4 EC2 |
| `AWS_EC2_PUBLIC_IP`, `AWS_EC2_KEYPAIR_NAME` | Phase 4 SSH + firewall rules |
| `GCP_PROJECT_ID`, `GCP_ZONE`, `GCP_VM_PUBLIC_IP` | Phase 4 Compute Engine |
| `ORCHESTRATOR_PUBLIC_ADDRESS`, `FLOWER_SERVER_PORT` | client → orchestrator address |
