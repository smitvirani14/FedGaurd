# FedGuard — Comparison Report

**Cross-cloud federated learning with differential privacy for fraud detection**

---

## 1. Problem statement

Payment-fraud models are only useful when trained on *more* data than any one
institution holds, but the data itself is exactly what regulation forbids from
moving: **PCI-DSS** restricts where cardholder records may be stored and
processed, **GDPR** treats transaction histories as personal data that cannot be
freely aggregated across borders, and bank secrecy laws do the same at national
level. The result is a stalemate — every bank has a fragment of the fraud
picture, none of them can pool it, and each model stays blind to the attack
patterns visible only to the others.

**Federated learning inverts the flow**: instead of shipping data to a central
trainer, the model travels to the data. Each institution trains locally and
publishes only a weight update. FedGuard demonstrates this end-to-end across
*two different public clouds* — an AWS EC2 node and a GCP Compute Engine node —
so the claim "the data never left its owner" is verifiable in the network logs
rather than merely asserted.

Sharing only weights is not, however, the same as sharing nothing: gradient
inversion and membership-inference attacks can partially reconstruct training
records from model updates. FedGuard therefore adds **differential privacy
(DP-SGD via Opacus)** so each published update carries calibrated noise, turning
"we believe the weights are harmless" into a quantified guarantee expressed as
ε.

---

## 2. Architecture

```
              ┌──────────────────────────────────────────────┐
              │        ORCHESTRATOR — GCP e2-micro           │
              │        fl_server.py  (FedAvg, Flower)        │
              │        8.231.92.55:8080  (bind 0.0.0.0)      │
              │   • averages client weights by sample count  │
              │   • logs one metrics row per round           │
              └────────▲──────────────────────────▲──────────┘
        weights only ▲ │                          ▲ │ ▲ weights only
      (4,097 float32)  │ │                          │ │   (4,097 float32)
   ┌───────────────────┴─┴──┐                 ┌────┴─┴─────────────────┐
   │  AWS EC2 — t3.micro    │                 │  GCP — e2-micro        │
   │  fl_client node1       │                 │  fl_client node2       │
   │  65.1.248.113          │                 │  localhost (same VM)   │
   │  data/node1_partition  │                 │  data/node2_partition  │
   │  170,933 rows (70% of  │                 │  113,874 rows (30% of  │
   │  fraud cases)          │                 │  fraud cases)          │
   └────────────────────────┘                 └────────────────────────┘
                       ▲                                ▲
                       └──────── laptop (dev) ──────────┘
                        Streamlit dashboard + results/*.csv
```

**Why the orchestrator sits on GCP:** the developer laptop is behind carrier
CGNAT and cannot accept inbound connections, and a tunnel account would have
cost money. Hosting `fl_server.py` on the already-free GCP VM gives a stable
public IP with no tunnel and no extra account. Each cloud node dials *out* to
it, so neither VM needs a wide-open firewall rule — the GCP rule for port 8080
is restricted to the AWS node's `/32`, never `0.0.0.0/0`.

**What actually crosses the network.** Every `fit()` returns ~4,097 float32
values — the FraudNet state dict. An assertion in `src/fl_client.py` enforces
this at runtime:

```
[client:node1] fit() sent 4097 weight values (0 raw rows)
```

No row of raw transaction data is ever serialized into a Flower message. The
payload is ~16 KB against a 170k-row dataset of ~21 MB.

---

## 3. Results

All numbers are the final round of each run, pulled from `results/results.csv`
and `results/cloud_results.csv` (never hand-typed).

### 3.1 Headline comparison

| Run | Rounds | Accuracy | Precision | Recall | F1 | AUC | ε |
|---|---|---|---|---|---|---|---|
| **Centralized baseline** | 10 epochs | 0.9994 | 0.8163 | 0.8163 | 0.8163 | 0.9793 | — |
| **Federated, local (no DP)** | 10 | 0.9994 | 0.8233 | 0.8423 | 0.8294 | 0.9902 | — |
| **Federated + DP (Opacus)** | 10 | 0.9993 | 0.8088 | **0.7675** | 0.7851 | **0.9327** | **1.1377** |
| **Federated, cross-cloud (AWS↔GCP)** | 5 | 0.9994 | 0.8151 | 0.8423 | 0.8256 | 0.9915 | — |

Reading the table:

1. **Federation costs nothing here — it may even help.** The local federated run
   *exceeds* the centralized baseline on recall (0.8423 vs 0.8163) and AUC
   (0.9902 vs 0.9793). Two explanations: each node sees a non-IID slice and
   therefore trains an ensemble of perspectives rather than one averaged view,
   and the federated runs perform 10 rounds × 1 local epoch of Adam updates
   versus 10 full-dataset epochs — comparable compute, different schedule. The
   important validation criterion is that federated performance *converges
   toward* the baseline rather than collapsing, which it does.
2. **DP costs real accuracy — as it must.** Recall drops 0.8423 → 0.7675 and AUC
   0.9902 → 0.9327. If all three runs reported identical numbers it would mean
   DP was never applied; the gap is the evidence that it was.
3. **Accuracy is a misleading metric on this dataset.** At a 0.17% fraud rate,
   an all-negative classifier scores 99.83% accuracy. Recall, F1 and AUC carry
   the signal — which is why the dashboard and this report lead with them.

### 3.2 The privacy budget

| Round | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| ε (mean over both nodes) | 0.4157 | 0.5443 | 0.6467 | 0.7352 | 0.8148 | 0.8877 | 0.9556 | 1.0195 | 1.0800 | 1.1377 |

ε grows monotonically (every round spends budget) but *sub-linearly* (advanced
composition) — the correct DP behaviour. Settings: σ = 1.0, clip bound C = 1.0,
δ = 1e-5, batch 512, 1 local epoch, lr 5e-3 (raised from 1e-3 because DP-SGD
noise inflates Adam's second-moment estimate; measured at equal ε,
lr=1e-3 → recall 0.00 vs lr=5e-3 → recall 0.67).

**Final ε = 1.1377 at δ = 1e-5** — a strong practical guarantee (research
deployments commonly report ε in the 1–8 range).

### 3.3 Cross-cloud validation (Phase 4)

The orchestrator log (`results/cloud_server.log`) records both client
connections with their source addresses — this is the evidence faculty asked for:

```
[server] client connected: cid=952a7ed0... peer=ipv6:%5B::1%5D:39850  connected=1
[server] client connected: cid=cd6ce931... peer=ipv4:65.1.248.113:41032 connected=2
```

- `65.1.248.113` — the **AWS EC2** public IP, connecting from ap-south-1.
- `[::1]` — node2, co-located on the **GCP orchestrator VM** itself.

Five real cross-cloud rounds completed with zero client drops:

| Round | Accuracy | Recall | AUC |
|---|---|---|---|
| 1 | 0.9988 | 0.4255 | 0.9395 |
| 2 | 0.9994 | 0.8116 | 0.9789 |
| 3 | 0.9994 | 0.8423 | 0.9883 |
| 4 | 0.9994 | 0.8423 | 0.9906 |
| 5 | 0.9994 | 0.8423 | 0.9915 |

Log excerpts: `results/cloud_server.log`, `results/cloud_node1.log`,
`results/cloud_node2.log`. Redeployment instructions for a fresh AWS/GCP
account are in `deployment/aws_setup.md` and `deployment/gcp_setup.md`.

### 3.4 Manual FedAvg verification (Phase 5)

`src/manual_fedavg.py` re-implements

```
global[i] = (n₁·node1[i] + n₂·node2[i]) / (n₁ + n₂)
```

per tensor element from the two nodes' saved weights, then compares against the
global parameters Flower's own `FedAvg` strategy produced:

```
loaded results/node1_weights.pt   n=136,746
loaded results/node2_weights.pt   n= 91,099
formula: global = (136,746·node1 + 91,099·node2) / 227,845

layer              max abs diff   allclose
net.0.weight         2.980e-08     PASS
net.0.bias           2.980e-08     PASS
net.2.weight         2.980e-08     PASS
net.2.bias           1.490e-08     PASS
net.4.weight         1.490e-08     PASS
net.4.bias           0.000e+00     PASS

RESULT: PASS (worst element diff 2.980e-08)
```

The residual ~3e-8 is float32 summation-order noise, far below the 1e-5
tolerance. This confirms the aggregation formula described in the report is the
one Flower actually executes.

---

## 4. Threat model

**What differential privacy *does* guarantee.** With σ=1.0, C=1.0, δ=1e-5 and a
final ε≈1.14, the distribution of any single node's released update sequence is
formally close (within ε) to what would have been produced had that node's data
point been absent entirely. An adversary holding the global model, every
round's gradients, and unlimited compute cannot determine with materially
better-than-prior probability whether any *particular* transaction was in the
training set. DP protects **membership**, and it protects it against an
unbounded adversary — no assumption about the attacker's compute is needed.

**What it does *not* guarantee.**

- **It is not encryption, and not secrecy.** ε bounds the *privacy loss of a
  membership query*; it says nothing about whether the learned model is
  *accurate* about a person, nor does it prevent the model from encoding
  population-level statistics. It is a probabilistic guarantee with an explicit
  failure probability δ, not absolute anonymity.
- **Attribute inference survives DP.** If an attacker already knows most of a
  victim's transaction history, DP does not stop them from using the published
  model to infer the *unknown* attributes — DP bounds participation evidence,
  not what the model has learned about distributions.
- **Reconstruction attacks on the *no-DP* baseline remain viable.** Without
  noise, gradient-inversion work (Zhu et al., "Deep Leakage from Gradients")
  and activation-leakage attacks can recover input features from gradients,
  especially on small batches. FedGuard's own claim here is narrow and
  testable: only 4,097 aggregate weights leave the node per round, never
  per-batch gradients — a much smaller attack surface than per-step gradient
  sharing, and DP covers the residual.
- **The orchestrator is a trusted-but-curious hub.** It sees every update in
  the clear. Malicious-aggregator attacks (model poisoning, backdoor
  injection, or simply logging updates for later analysis) are out of scope
  and would require secure aggregation + robust aggregation rules.
- **Two colluding honest-but-curious parties** could differ their updates to
  isolate one node's contribution. With only 2 nodes this is a real weakness —
  secure aggregation (which hides individual updates behind a sum) is the
  standard mitigation and is listed under future work.
- **Side channels are unaddressed:** update timing, update magnitudes on rounds
  where a node did not participate, and metadata (who connected, when) are all
  visible to the orchestrator. Our own connection log is an example.

**Threats *out* of scope:** compromised node software (a malicious client can
report anything), physical access to a VM, and model-extraction attacks that
query the model rather than its training data.

---

## 5. Limitations

1. **Only two nodes.** Real cross-silo FL uses dozens to hundreds of
   institutions; with n=2 the statistical behaviour of FedAvg and the
   robustness of aggregation are far easier than in production, and the
   collusion concern above has no cheap fix at n=2.
2. **Free-tier compute.** `t3.micro` (2 vCPU / 1 GB) and `e2-micro` (1 shared
   vCPU / 1 GB) cap us at a 4,097-parameter MLP and 5 rounds on the cross-cloud
   run. The GCP VM needed a 2 GB swap file and `OMP_NUM_THREADS=1` to hold the
   server and a torch client simultaneously.
3. **Simulated non-IID split, not genuinely separate institutions.** Both
   partitions come from one public Kaggle CSV, skewed 70/30 on fraud cases.
   Real institutions differ in schema, label policy, fraud definitions, sensor
   quality and time period — differences no re-weighting of a single dataset
   reproduces.
4. **Static batch data.** Training runs over a fixed CSV; production fraud
   arrives as a continuous stream with drift, concept decay and delayed labels.
5. **No secure aggregation.** DP protects each update in isolation, but the
   orchestrator still observes individual node updates rather than only their
   sum.
6. **Honest-but-curious participants.** There is no verification that a client
   trained on its claimed data or applied the advertised noise; malicious
   clients can poison the global model or skip DP entirely.
7. **One DP hyper-parameter setting.** σ=1.0/C=1.0 was not swept; a
   privacy-utility frontier over several σ values would be more informative
   than a single operating point.

---

## 6. Future work

- **Secure aggregation** (Bonawitz et al.) so the orchestrator sees only the
  summed update, closing the trusted-hub gap and the n=2 collusion weakness
  simultaneously.
- **More nodes** — 5–10 participants, ideally with heterogeneous features, to
  exercise FedAvg's behaviour under realistic client availability and stragglers.
- **Adaptive DP accounting** — RDP/PRV accountants, per-node budget allocation,
  and a spent-budget stop condition rather than a fixed round count.
- **Streaming data** with periodic re-training and drift detection instead of a
  static CSV.
- **Robust aggregation** (Krum, trimmed mean, or median) to resist model
  poisoning once participants are no longer all trusted.
- **Real institutional data** behind a data-sharing agreement, or a
  privacy-preserving synthetic generator calibrated to multiple banks' marginals.

---

## 7. Reproducing this report

```bash
pip install -r requirements.txt
python src/centralized_baseline.py            # Phase 1 → results/centralized_baseline.json
# Phase 2/3: start fl_server.py, then fl_client.py --partition node1 / node2
#            (add --use-dp for the DP run)
python src/manual_fedavg.py                   # Phase 5 → PASS/FAIL vs Flower
streamlit run dashboard/app.py                # Phase 6 → live charts
```

Cloud deployment: `deployment/aws_setup.md`, `deployment/gcp_setup.md`.
Raw per-run CSVs: `results/results.csv`, `results/cloud_results.csv`,
`results/federated_dp.csv`, `results/federated_nodp.csv`.
