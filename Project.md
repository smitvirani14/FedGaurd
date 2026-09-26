# FedGuard: Cross-Cloud Federated Fraud Detection Platform
### AI Agent Implementation Plan (Phase-Wise, Step-by-Step)

> **Purpose of this document:** This is a build specification meant to be handed to an AI coding agent (e.g., Antigravity, Claude Code, Cursor, etc.). It describes exactly what to build, in what order, which files to create, what each file should contain, and how to validate each phase before moving to the next. Follow phases strictly in order — each phase depends on the previous one working correctly.

---

## 0. Project Overview (Read First)

**Goal:** Build a federated learning system that trains a fraud-detection model across two simulated "cloud nodes" (AWS EC2 + GCP Compute Engine) without either node sharing raw transaction data. A central orchestrator aggregates model updates using FedAvg. Differential Privacy (DP) noise is added to updates before transmission to prove data cannot be reverse-engineered.

**Final deliverables:**
1. Working federated learning pipeline (local simulation + cloud deployment).
2. Differential privacy integration with measurable privacy budget (ε).
3. A dashboard showing training rounds, accuracy/recall, and privacy budget over time.
4. A comparison report: Centralized model vs. Federated model vs. Federated+DP model.
5. Deployment scripts/instructions for AWS EC2 and GCP Compute Engine.

**Tech stack (fixed — do not substitute without reason):**
- Python 3.10+ (arm64 build — see Phase 0 for MacBook Air M2 setup)
- PyTorch (model + training) — **CPU backend, not MPS** (see Phase 0 for why)
- Flower (`flwr`) — federated learning orchestration framework
- Opacus — differential privacy for PyTorch
- pandas, scikit-learn — data handling and metrics
- python-dotenv — loads credentials/config from `.env` (see Phase -1)
- Streamlit — results dashboard
- Kaggle "Credit Card Fraud Detection" dataset (public, `creditcard.csv`)

> **Development machine:** MacBook Air M2 (Apple Silicon, arm64). All tool choices below are picked to work natively on arm64 without emulation. Phase 0 covers exact setup.

**Repository structure to create (target end-state):**
```
fedguard/
├── .env.example                 # template of all required user-supplied values
├── .env                         # actual secrets/config (gitignored) - created from user input
├── .gitignore                   # must exclude .env, data/raw/, *.pem, kaggle.json
├── data/
│   ├── raw/                     # original creditcard.csv
│   ├── node1_partition.csv      # simulated AWS node data
│   └── node2_partition.csv      # simulated GCP node data
├── src/
│   ├── model.py                 # shared model definition
│   ├── data_utils.py            # loading, splitting, preprocessing
│   ├── centralized_baseline.py  # Phase 1
│   ├── fl_client.py             # Phase 2 - Flower client
│   ├── fl_server.py             # Phase 2 - Flower server (FedAvg)
│   ├── dp_engine.py             # Phase 3 - Opacus wrapper
│   ├── manual_fedavg.py         # Phase 5 - hand-written FedAvg for viva explanation
│   ├── check_env.py             # Phase 0 - environment sanity check
│   └── metrics_logger.py        # writes round-by-round results to results.csv
├── dashboard/
│   └── app.py                   # Streamlit dashboard
├── deployment/
│   ├── aws_setup.md             # EC2 setup instructions
│   ├── gcp_setup.md             # GCE setup instructions
│   └── run_node.sh              # startup script for each cloud VM
├── results/
│   └── results.csv              # generated during training
├── requirements.txt
├── README.md
└── report/
    └── comparison_report.md     # final centralized vs FL vs FL+DP writeup
```

Create this folder structure at the start of Phase 1 and keep adding files into it exactly as named above — later phases and the report reference these exact paths.

---

## PHASE -1 — External Requirements & User Input Collection (Do This First, Before Any Code)

**Objective:** This project depends on external accounts, credentials, and dataset access that the AI agent cannot obtain on its own. Before writing any code, the agent must check what's available, ask the user (via the terminal/chat) for anything missing, and store everything in a single `.env` file that every later phase reads from — so no credential is ever hardcoded into source files.

### 1. External accounts/resources the user must have or create

| # | Requirement | Who provides it | Cost | Blocking for which phase? |
|---|---|---|---|---|
| 1 | **AWS account** with a card on file, free tier active | User | Free (with billing alerts) | Phase 4 |
| 2 | **AWS Access Key ID + Secret Access Key** (IAM user with EC2 permissions, not root) | User | Free | Phase 4 |
| 3 | **GCP account** with billing enabled (free trial or always-free tier) | User | Free (with billing alerts) | Phase 4 |
| 4 | **GCP Project ID** + a service account JSON key, or `gcloud auth login` completed interactively | User | Free | Phase 4 |
| 5 | **Kaggle account** + Kaggle API token (`kaggle.json`, containing `username` and `key`) | User | Free | Phase 1 |
| 6 | **GitHub (or GitLab) account** + a repo URL for this project | User | Free | Ongoing (version control) |
| 7 | A stable way to reach the orchestrator from both cloud VMs — either a public IP on a third VM, or a tunneling tool (`ngrok`/`cloudflared`) if running the orchestrator on the user's own laptop behind CGNAT/NAT | User to decide | Free tier available for ngrok/cloudflared | Phase 4 |

### 2. Instruction to the AI agent: how to collect these

Before starting Phase 0, run an interactive setup step:

1. Create `.env.example` in the repo root listing every variable below **with empty/placeholder values and a comment explaining each one**:
   ```bash
   # ---- Kaggle ----
   KAGGLE_USERNAME=
   KAGGLE_KEY=

   # ---- AWS ----
   AWS_ACCESS_KEY_ID=
   AWS_SECRET_ACCESS_KEY=
   AWS_DEFAULT_REGION=              # e.g. us-east-1
   AWS_EC2_KEYPAIR_NAME=            # name of the .pem keypair for SSH access
   AWS_EC2_PUBLIC_IP=               # filled in AFTER the instance is launched in Phase 4

   # ---- GCP ----
   GCP_PROJECT_ID=
   GCP_ZONE=                        # e.g. us-central1-a
   GCP_SERVICE_ACCOUNT_JSON_PATH=   # local path to the downloaded service account key file
   GCP_VM_PUBLIC_IP=                # filled in AFTER the instance is launched in Phase 4

   # ---- Orchestrator ----
   ORCHESTRATOR_PUBLIC_ADDRESS=     # public IP:port or ngrok/cloudflared URL the two cloud clients connect to
   FLOWER_SERVER_PORT=8080

   # ---- GitHub ----
   GIT_REPO_URL=
   ```
2. Copy `.env.example` to `.env`.
3. **Ask the user, one at a time, in plain language, for each value that is still blank**, in this order (earlier ones unblock earlier phases, so ask in this sequence rather than all at once):
   - "Do you have a Kaggle account? If yes, please provide your Kaggle username and API key (from kaggle.com → Account → Create New API Token). If you don't have one yet or don't want to set it up right now, I'll use a synthetic dataset instead so we can keep moving."
   - "Do you have an AWS account with billing/free-tier active? If yes, please create an IAM user with EC2 permissions (not your root account) and provide the Access Key ID, Secret Access Key, and preferred region (e.g. us-east-1). If not, let's pause Phase 4 until this is ready — Phases 0-3 and 5-7 don't need it."
   - "Do you have a GCP account with billing enabled? Please provide your Project ID and either run `gcloud auth login` yourself, or download a service account JSON key and give me its local file path."
   - "How would you like to run the orchestrator — on your Mac (I'll help you set up ngrok/cloudflared so the cloud VMs can reach it), or on a third small cloud VM?"
   - "Do you have a GitHub repo ready for this project? If yes, share the URL so I can help you push commits after each phase."
4. Write the confirmed values into `.env`. **Never print the actual key/secret values back into chat or logs after they're entered** — only confirm "saved" for each.
5. Add `.env`, `*.pem`, `kaggle.json`, and `data/raw/` to `.gitignore` immediately, before any commit is made, so secrets and the raw dataset are never pushed to GitHub.
6. Every later script (`data_utils.py`, `fl_client.py`, deployment scripts, etc.) must load configuration via `.env` (e.g., using `python-dotenv`) rather than ever hardcoding a key, IP, or path — add `python-dotenv` to `requirements.txt`.

### 3. Graceful degradation if something is missing

The agent should never block the entire project on one missing external resource:
- **No Kaggle access yet** → proceed with the synthetic-data fallback already defined in Phase 1, and revisit real data later once credentials arrive.
- **No AWS/GCP access yet** → complete Phases 0, 1, 2, 3, 5, 6, 7 fully (all of which run locally), and clearly mark Phase 4 as "pending external credentials" in `results/` and the report, rather than skipping the cloud architecture explanation entirely.
- **No public IP / stuck behind CGNAT for the orchestrator** → ask the user whether they want to set up `ngrok` (fastest) or run the orchestrator on a third free-tier VM instead; do not silently fail.

### 4. Validation before moving to Phase 0
- [ ] `.env.example` exists and is committed to git; `.env` exists locally and is **not** committed (confirm via `git status` that it's ignored).
- [ ] Every value the user provided has been sanity-checked (e.g., `aws sts get-caller-identity` succeeds using the provided keys; `gcloud projects describe $GCP_PROJECT_ID` succeeds; a test Kaggle API call succeeds) before moving on.
- [ ] The user has explicitly confirmed which orchestrator hosting option they're using.

---

## PHASE 0 — Environment Setup for MacBook Air M2 (Apple Silicon)

**Objective:** Get a clean, native arm64 Python environment so nothing later silently falls back to Rosetta emulation (slow) or breaks on an x86-only package. Do this before touching Phase 1.

### Tool choices and why (all arm64-native, no emulation needed):

| Purpose | Tool | Notes for M2 |
|---|---|---|
| Package/formula manager | **Homebrew** (arm64 build, installs to `/opt/homebrew`) | Confirm with `brew --prefix` → should print `/opt/homebrew`, not `/usr/local` (the latter means it's running under Rosetta) |
| Python version manager | **pyenv** (via Homebrew) | Install Python 3.11 (not 3.13 yet — Opacus/Flower version compatibility lags newest Python releases). Avoid the macOS system Python. |
| Virtual environment | **venv** (built into Python) | Simpler than conda here; conda's arm64 channel support can be inconsistent for some scientific packages. `python3 -m venv .venv` |
| Deep learning framework | **PyTorch (CPU build)** | Install via `pip install torch --index-url https://download.pytorch.org/whl/cpu`. M2 does have a GPU-accelerated `mps` backend, but **do not use it for this project** — Opacus's per-sample gradient hooks (needed for differential privacy) do not reliably support the MPS backend yet and will throw errors or silently give wrong gradients. Force CPU everywhere: `device = torch.device("cpu")`. The models in this project are small enough that CPU training is fast regardless. |
| Federated learning | **Flower (`flwr`)** | Pure Python + gRPC, works natively on arm64, no special setup. |
| Differential privacy | **Opacus** | Pure Python/PyTorch, works fine on arm64 as long as the model runs on CPU (see above). |
| Editor/agent | **Antigravity** (already chosen) or VS Code | Both run natively on Apple Silicon. |
| Terminal | Built-in **Terminal.app** or **iTerm2** | Either is fine; iTerm2 is arm64-native and gives you split panes, useful for running server + 2 clients side by side in Phase 2. |
| Cloud CLIs | **AWS CLI v2** (arm64 pkg) and **Google Cloud SDK** (`gcloud`, arm64 tarball) | Both publish native Apple Silicon installers — do not use the older x86 pkg via Rosetta. |
| SSH | Built-in `ssh` client (macOS) | No extra tooling needed to connect to the EC2/GCE VMs. |
| Optional: local container simulation | **Docker Desktop for Mac (Apple Silicon)** or **OrbStack** (lighter, arm64-native, popular M-series alternative to Docker Desktop) | Optional but recommended: before paying for/using real cloud VMs, you can run "Node 1" and "Node 2" as two separate Docker containers on your Mac to rehearse the exact cross-network setup (different IPs, ports) with less friction than juggling two free-tier VMs from day one. |
| Dashboard | **Streamlit** | Pure Python, works fine natively. |

### Setup steps for the AI agent:
1. Verify Homebrew is arm64: `brew --prefix` → expect `/opt/homebrew`.
2. Install pyenv: `brew install pyenv`, then `pyenv install 3.11.9 && pyenv local 3.11.9`.
3. Create and activate a virtual environment: `python3 -m venv .venv && source .venv/bin/activate`.
4. Install PyTorch CPU build explicitly (not the default `pip install torch`, which may pull an MPS-enabled build you then have to remember not to use): 
   ```
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   ```
5. Install the rest: `pip install flwr[simulation] opacus pandas scikit-learn streamlit`.
6. Add a one-line sanity check script `src/check_env.py` that prints `torch.__version__`, confirms `torch.backends.mps.is_available()` (informational only — expect `True` on M2, but this project intentionally does not use it), and confirms the active device is set to `"cpu"` throughout the codebase.
7. Install cloud CLIs: `brew install awscli` (arm64 native) and `brew install --cask google-cloud-sdk`.

### Validation before moving to Phase 1:
- [ ] `python -c "import torch; print(torch.__version__)"` runs without errors inside the venv.
- [ ] `pip show opacus flwr` both resolve without falling back to source builds that require Rosetta/x86 toolchains.
- [ ] `aws --version` and `gcloud --version` both report as arm64/darwin builds (check output text for `arm64` or `darwin_arm64`).

---

## PHASE 1 — Data Preparation & Centralized Baseline

**Objective:** Get a working non-federated model first. This becomes the accuracy benchmark that federated learning will be compared against later.

### Steps for the AI agent:
1. Create the folder structure above.
2. Download or accept the Kaggle "Credit Card Fraud Detection" dataset (`creditcard.csv`, ~285k rows, 30 anonymized features + `Class` label where 1 = fraud). Place it at `data/raw/creditcard.csv`. If the dataset cannot be downloaded automatically (Kaggle requires auth), generate a synthetic stand-in dataset with the same schema (30 numeric features, highly imbalanced binary label ~0.17% positive class) so the pipeline is testable end-to-end, and clearly log a warning that this is a placeholder.
3. Write `src/data_utils.py` with functions:
   - `load_data(path)` → returns pandas DataFrame.
   - `preprocess(df)` → scales `Amount` and `Time` columns (StandardScaler), returns features `X` and labels `y`.
   - `split_for_nodes(X, y, strategy="non_iid")` → splits data into two partitions simulating Node 1 (AWS, "credit card transactions") and Node 2 (GCP, "bank logs"). For `non_iid` strategy, skew the fraud ratio differently between the two partitions (e.g., Node 1 gets 70% of fraud cases, Node 2 gets 30%) to make the federated learning problem realistic and non-trivial. Save outputs to `data/node1_partition.csv` and `data/node2_partition.csv`.
4. Write `src/model.py` defining a small PyTorch model:
   - `class FraudNet(nn.Module)`: input layer sized to feature count → hidden layer(s) (e.g., 64 → 32) with ReLU → single sigmoid output.
   - Keep it small and CPU-trainable (this must run on free-tier cloud VMs).
5. Write `src/centralized_baseline.py`:
   - Loads full dataset (not split), trains `FraudNet` for N epochs with `BCELoss` and Adam optimizer.
   - Evaluates using accuracy, precision, recall, F1, and ROC-AUC (recall and AUC matter most since fraud is a rare-class problem — say this explicitly in code comments).
   - Saves results to `results/centralized_baseline.json`.
6. Write `src/metrics_logger.py` with a reusable function `log_round(round_num, source, accuracy, precision, recall, f1, auc, epsilon=None)` that appends a row to `results/results.csv`. All later phases must call this after every evaluation.

### Validation before moving to Phase 2:
- [ ] `results/centralized_baseline.json` exists and contains recall/AUC values noticeably above random guessing (AUC > 0.85 is a reasonable target for this dataset).
- [ ] `data/node1_partition.csv` and `data/node2_partition.csv` exist, together account for all rows in the original dataset, and have different fraud ratios (print and log this difference).

---

## PHASE 2 — Federated Learning Simulation (Single Machine, Multiple Processes)

**Objective:** Get FedAvg working correctly on one machine (simulating two nodes as separate processes) before adding cloud/network complexity.

### Steps for the AI agent:
1. Install `flwr` (`pip install flwr[simulation]`).
2. Write `src/fl_client.py`:
   - Implements a `FlowerClient(fl.client.NumPyClient)` class with:
     - `get_parameters()` — returns current local model weights.
     - `fit(parameters, config)` — sets model weights from server, trains locally for a few epochs on this node's partition only, returns updated weights + number of local samples.
     - `evaluate(parameters, config)` — evaluates the given global weights on this node's local test split, returns loss + metrics dict.
   - Accepts a command-line argument `--partition` (`node1` or `node2`) to decide which CSV to load — this is critical, since this same file will later run unmodified on the real AWS/GCP VMs.
3. Write `src/fl_server.py`:
   - Starts a Flower server using `fl.server.strategy.FedAvg` with parameters: `min_fit_clients=2`, `min_available_clients=2`, and a custom `evaluate_metrics_aggregation_fn` that computes weighted-average accuracy/recall/AUC across clients.
   - Runs for a configurable number of rounds (start with 10).
   - After each round, calls `metrics_logger.log_round(..., source="federated")`.
4. Run locally as a smoke test: start `fl_server.py` in one terminal, then `fl_client.py --partition node1` and `fl_client.py --partition node2` in two other terminals (or use Flower's simulation API `fl.simulation.start_simulation` to run both clients in one script for convenience during development).

### Validation before moving to Phase 3:
- [ ] Training completes 10 rounds without errors.
- [ ] `results/results.csv` shows federated accuracy/recall converging toward (not necessarily matching exactly) the centralized baseline from Phase 1.
- [ ] Confirm in logs that node1's raw data is never sent anywhere — only `get_parameters()` outputs (weight arrays) cross the client→server boundary. Add an explicit code comment/assertion here for the report/viva.

---

## PHASE 3 — Add Differential Privacy (Opacus)

**Objective:** Ensure that the weights each node sends to the orchestrator have calibrated noise added, so raw training data cannot be reverse-engineered from gradients.

### Steps for the AI agent:
1. Install `opacus` (`pip install opacus`).
2. Write `src/dp_engine.py`:
   - Function `make_private(model, optimizer, data_loader, noise_multiplier, max_grad_norm)` that wraps the model/optimizer/loader using `opacus.PrivacyEngine().make_private(...)`.
   - Function `get_privacy_spent(privacy_engine, delta=1e-5)` that returns the current epsilon (ε) value.
3. Modify `src/fl_client.py`'s `fit()` method to route local training through the DP-wrapped model/optimizer instead of the plain PyTorch ones. After each local training round, log the resulting epsilon via `metrics_logger.log_round(..., epsilon=eps)`.
4. Add a config flag `USE_DP = True/False` at the top of `fl_client.py` so the agent (and later you, in the demo) can toggle DP on/off and compare results — this comparison is required for the final report.
5. Re-run the Phase 2 simulation with `USE_DP=True` for 10 rounds, then again with `USE_DP=False`, saving both result sets separately (`results/federated_dp.csv`, `results/federated_nodp.csv`).

### Validation before moving to Phase 4:
- [ ] Epsilon (ε) values are being logged and decrease in privacy (increase in ε) sensibly over rounds — this is expected DP behavior (privacy budget accumulates over training rounds).
- [ ] Accuracy with DP is somewhat lower than without DP (this is the expected privacy-utility tradeoff — do not treat a small accuracy drop as a bug).
- [ ] Document the final ε achieved and its accuracy in a short markdown note at `results/dp_summary.md`.

---

## PHASE 4 — Cloud Deployment (AWS EC2 + GCP Compute Engine)

**Objective:** Move the exact same `fl_client.py` code (unmodified) onto two real cloud VMs so the "cross-cloud" claim is genuinely demonstrated, not simulated.

### Steps for the AI agent:
1. Write `deployment/aws_setup.md` with instructions to:
   - Launch a `t2.micro` (free-tier) EC2 instance, Ubuntu 22.04.
   - Open inbound port `8080` (Flower's default) in the security group, restricted to the orchestrator's IP only (not `0.0.0.0/0`) — note this as a security best practice in the file.
   - SSH in, install Python 3.10+, `pip install -r requirements.txt`.
   - Copy `src/fl_client.py`, `src/model.py`, `src/dp_engine.py`, `src/data_utils.py`, and `data/node1_partition.csv` onto this instance.
2. Write `deployment/gcp_setup.md` with equivalent instructions for a GCP `e2-micro` Compute Engine VM (Debian/Ubuntu image), same firewall/port logic, and copying `data/node2_partition.csv` instead.
3. Write `deployment/run_node.sh`, a shell script that each VM runs to start its client, e.g.:
   ```bash
   #!/bin/bash
   python3 fl_client.py --partition $1 --server_address $2:8080
   ```
   where `$1` is `node1` or `node2` and `$2` is the orchestrator's public IP.
4. Update `src/fl_server.py` to accept a `--server_address 0.0.0.0:8080` argument so it can bind to all interfaces (needed to accept remote connections rather than just localhost).
5. Decide where the orchestrator itself runs — simplest for a student demo is your own laptop with port forwarding, or a small third free-tier VM on either cloud. Document whichever choice is made in `deployment/aws_setup.md` / `gcp_setup.md`.

### Validation before moving to Phase 5:
- [ ] Orchestrator log shows exactly 2 connected clients, one from the AWS EC2 public IP and one from the GCP public IP (verify via server-side connection logs).
- [ ] Training completes at least 5 real cross-cloud rounds without a client dropping.
- [ ] Take a screenshot/log excerpt of this cross-cloud run for the final report/demo — faculty will specifically ask to see this.

---

## PHASE 5 — Manual FedAvg Implementation (For Viva/Understanding)

**Objective:** Flower's `FedAvg` strategy hides the aggregation math. Faculty commonly ask students to explain FedAvg in detail, so build a small standalone script proving understanding of the underlying algorithm.

### Steps for the AI agent:
1. Write `src/manual_fedavg.py`:
   - Loads two sets of saved model weights (`node1_weights.pt`, `node2_weights.pt`) and their respective sample counts (`n1`, `n2`).
   - Implements weighted averaging manually:
     ```
     global_weight[i] = (n1 * node1_weight[i] + n2 * node2_weight[i]) / (n1 + n2)
     ```
     applied per-layer, per-parameter-tensor.
   - Prints/logs a comparison showing this manual result is numerically identical (within floating-point tolerance) to what Flower's built-in `FedAvg` produced in Phase 2, using `torch.allclose(...)`.
2. Add a short docstring at the top of this file explaining *why* FedAvg weights by sample count rather than doing a simple unweighted average (nodes with more data should have proportionally more influence on the global model).

### Validation:
- [ ] `torch.allclose()` check passes (or is close within a documented tolerance), proving the manual implementation matches Flower's internal behavior.

---

## PHASE 6 — Results Dashboard (Streamlit)

**Objective:** Provide a visual, live-updating way to present results during the demo/viva.

### Steps for the AI agent:
1. Write `dashboard/app.py`:
   - Reads `results/results.csv` (auto-refresh every few seconds using `st.rerun()` or a manual refresh button).
   - Plots line charts: accuracy/recall/AUC vs. training round (federated vs. centralized baseline as a reference horizontal line).
   - Plots epsilon (ε) vs. round for the DP run.
   - Adds a summary table at the top: Centralized vs. Federated (no DP) vs. Federated (with DP) — final accuracy, recall, AUC, epsilon.
   - Adds a short static explanation panel (markdown block within the Streamlit app) describing the architecture (AWS node, GCP node, orchestrator) so it's self-explanatory during a live demo.
2. Command to run: `streamlit run dashboard/app.py`.

### Validation:
- [ ] Dashboard loads without errors and correctly reflects the latest `results.csv` contents.
- [ ] All three comparison numbers (centralized / FL / FL+DP) are visibly different from each other in the summary table, demonstrating a real tradeoff rather than three identical numbers (which would suggest DP/FL wasn't actually applied).

---

## PHASE 7 — Final Report & Documentation

**Objective:** Produce the write-up faculty will actually read/grade, and a top-level README so the project is understandable to anyone opening the repo.

### Steps for the AI agent:
1. Write `report/comparison_report.md` containing:
   - Problem statement and why cross-cloud federated learning is needed (PCI-DSS/GDPR data-sharing restrictions).
   - Architecture diagram (describe in text/ASCII, or generate an image if tooling allows).
   - Final results table: Centralized vs. FL vs. FL+DP (accuracy, precision, recall, F1, AUC, epsilon).
   - A short "Threat Model" section: what an attacker could still infer from gradients even with DP applied, and why DP's epsilon value represents a privacy guarantee rather than absolute secrecy — this section specifically demonstrates depth of understanding to evaluators.
   - Limitations section: e.g., only 2 nodes (real deployments use many more), free-tier VM compute constraints, synthetic non-IID split rather than genuinely separate real-world institutions.
   - Future work: differential privacy with secure aggregation, more nodes, real-time streaming data instead of a static CSV.
2. Write the top-level `README.md`:
   - What the project does, in 3–4 sentences.
   - Folder structure explanation.
   - Exact setup commands (`pip install -r requirements.txt`, dataset placement, how to run each phase's scripts locally and on cloud).
   - How to run the dashboard.
3. Write `requirements.txt` pinning versions of: `torch`, `flwr`, `opacus`, `pandas`, `scikit-learn`, `streamlit`.

### Validation (final project acceptance criteria):
- [ ] A fresh clone of the repo, following only `README.md`, can reproduce the centralized baseline and the local federated simulation end-to-end.
- [ ] `report/comparison_report.md` contains real numbers pulled from `results/results.csv`, not placeholder text.
- [ ] All cloud deployment steps in `deployment/` are precise enough that someone unfamiliar with the project could redeploy it on a fresh AWS/GCP account.

---

## Notes for the AI Agent Executing This Plan

- Do not skip Phase 1's centralized baseline — every later comparison depends on having this number first.
- Keep `fl_client.py` cloud-agnostic: it should not contain any AWS- or GCP-specific code. The only cloud-specific parts belong in `deployment/`.
- Log everything through `metrics_logger.py` consistently so `results.csv` has a uniform schema across all phases — the dashboard and report both depend on this.
- If any cloud deployment step (Phase 4) is not achievable due to account/billing constraints, still complete Phases 1–3, 5–7 fully, and clearly document in the report that Phase 4 was simulated on localhost using two processes with distinct data partitions as a stand-in, so the project remains complete and demoable.
- Favor small, CPU-only models and small batch sizes throughout — this must run on free-tier cloud instances (1 vCPU, ~1GB RAM), not a GPU cluster.
- The development machine is a MacBook Air M2 — always install arm64-native tools (see Phase 0), always force `torch.device("cpu")` in code (never `mps`), and never assume x86_64-only packages will work without Rosetta.
- Never hardcode credentials, IPs, or file paths in source files. Always read them from `.env` via `python-dotenv`, as set up in Phase -1. If a required `.env` value is still blank when a script runs, stop and ask the user for it rather than guessing or using a fake placeholder silently.
