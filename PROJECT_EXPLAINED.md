# My Project — In Simple Words (FedGuard)

> Read this once before the viva. Everything here is in plain English, and every
> number is real — it came from my own runs, saved in `results/`.

---

## 1. One-line answer

**"I built a fraud-detection model that two banks, sitting on two different
clouds (AWS and GCP), train together — without ever sharing a single customer
transaction with each other — and I added mathematical noise so even the model
updates can't be reversed back into real customer data."**

---

## 2. The problem, explained simply

Banks want to catch credit-card fraud. Fraud patterns keep changing, so a bank
needs to train its model on **more data than it has**. Other banks have that
data — but they are **legally not allowed to share it**:

- **PCI-DSS** (card industry rule) says card data must stay where it is stored.
- **GDPR** (European law) says a person's transaction history is personal data
  and can't just be shipped around.

So everyone is stuck: each bank sees only its own slice of the fraud picture,
and no bank sees the whole picture.

**My project solves this by reversing the direction:** instead of sending the
data to the model, I send the **model to the data**.

---

## 3. What is Federated Learning (FL) — in 4 steps

1. A central server (I call it the **orchestrator**) sends the current model to
   each bank.
2. Each bank trains that model **on its own data, inside its own cloud**.
3. Each bank sends back **only the updated numbers (weights)** — not one row of
   data.
4. The server **averages** those updates into one better global model, and the
   cycle repeats.

That averaging step is called **FedAvg** (Federated Averaging). The formula is:

```
new model = (n1 × update_from_bank1 + n2 × update_from_bank2) / (n1 + n2)
```

It's weighted by **how much data each bank used** — a bank with more data gets
proportionally more say. That's the whole idea: weight by sample count, so the
result is statistically close to training on all the data in one place.

---

## 4. What makes MY project more than a tutorial

Four things, and I can prove each one:

| # | Claim | Proof in my repo |
|---|---|---|
| 1 | It runs on **two real clouds**, not one laptop | AWS EC2 node + GCP VM, logs in `results/cloud_server.log` |
| 2 | **No raw data ever crosses the network** | every round logs `sent 4097 weight values (0 raw rows)` |
| 3 | **Differential privacy** is actually applied | ε (epsilon) is logged every round in `results/results.csv` |
| 4 | The FedAvg math is **verified by hand** | `src/manual_fedavg.py` reproduces Flower's answer exactly |

### The cross-cloud proof (this is the line faculty look for)

My orchestrator log shows both clients connecting, with their real IP addresses:

```
[server] client connected: peer=ipv6:::1        connected=1   ← GCP node (local)
[server] client connected: peer=ipv4:65.1.248.113 connected=2 ← AWS EC2 node (real internet)
```

So yes — one model, trained across two different companies' clouds, in 5 rounds,
with zero drops.

---

## 5. Differential Privacy (DP) — explained simply

Sharing weights is safer than sharing data, but it's **not zero risk**. A
clever attacker can sometimes work backwards from model updates and guess what
was in the training data (this is a real research area — e.g. "Deep Leakage
from Gradients").

**DP-SGD fixes this with two mechanical steps:**

1. **Clip** each example's gradient so no single transaction can shout too loud.
2. **Add Gaussian noise** before the update leaves the bank.

After that, my guarantee becomes a number: **ε (epsilon) = 1.14** (at δ = 1e-5).

- **Lower ε = stronger privacy.** ε around 1–8 is considered strong in
  research deployments.
- ε **grows every round**, because every round spends a bit more of the privacy
  budget. That's correct behaviour, not a bug.

I used **Opacus** (Facebook/Meta's library) to do this, toggled with a simple
`--use-dp` flag so I can demo both ways.

---

## 6. The results — say these numbers out loud

| Run | Recall | AUC | ε |
|---|---|---|---|
| Centralized (normal training, all data in one place) | 0.816 | 0.979 | — |
| Federated, **no** DP | 0.842 | 0.990 | — |
| Federated **with** DP | 0.768 | 0.933 | 1.14 |
| Federated across AWS + GCP | 0.842 | 0.991 | — |

**How to interpret this in one breath:**

- Federated learning gave me essentially the **same accuracy as normal
  centralized training** — so federation costs me nothing.
- Adding DP **lowered recall from 0.842 → 0.768 and AUC from 0.990 → 0.933**.
  That drop is the **price of privacy**, and it's a *good* sign — if all three
  rows were identical, it would mean DP wasn't really doing anything.
- I report **recall and AUC, not accuracy**, because only 0.17% of transactions
  are fraud. A model that always says "not fraud" already gets 99.8% accuracy —
  accuracy is meaningless here.

---

## 7. What I'd show in a live demo (30 seconds)

```bash
streamlit run dashboard/app.py
```

That opens a dashboard with: the comparison table, metrics-vs-round chart, the
privacy budget curve (ε climbing), the privacy-vs-accuracy tradeoff, and a
drawing of the architecture.

If asked to prove it runs: I have real logs from today's cross-cloud run saved
in `results/`.

---

## 8. The parts of the system (know these names)

| Piece | File | What it does |
|---|---|---|
| Orchestrator / server | `src/fl_server.py` | averages everyone's updates (FedAvg) |
| Client (runs on each bank) | `src/fl_client.py` | trains locally, sends weights only |
| The model | `src/model.py` | tiny neural net, FraudNet: 30→64→32→1 |
| Privacy engine | `src/dp_engine.py` | Opacus DP-SGD wrapper |
| Hand-written FedAvg check | `src/manual_fedavg.py` | proves I understand the math |
| Baseline | `src/centralized_baseline.py` | the number everything is compared to |
| Dashboard | `dashboard/app.py` | the visual demo |
| Cloud setup docs | `deployment/*.md` | how to redeploy from scratch |

**Tools used:** Python 3.11, PyTorch (CPU), **Flower** (federated framework),
**Opacus** (differential privacy), scikit-learn, Streamlit, AWS EC2, GCP
Compute Engine.

---

## 9. Likely faculty questions + short answers

**Q: Why federated learning? Why not just share the data?**
→ Because PCI-DSS and GDPR forbid moving cardholder/personal data between
institutions. FL lets them collaborate without breaking the law.

**Q: What exactly is sent over the network?**
→ 4,097 float32 numbers (about 16 KB of model weights), and scalar metrics.
Never a raw transaction — and my code *asserts* this at runtime, so it would
crash if raw data were ever included.

**Q: What is FedAvg, precisely?**
→ Weighted average of the clients' updates, weighted by each client's sample
count: `(n1·w1 + n2·w2)/(n1+n2)`. Weighted (not simple average) so banks with
more data have proportionally more influence — that's what makes the result
statistically close to centralized training.

**Q: How do you know your FedAvg is correct?**
→ I wrote it myself in `src/manual_fedavg.py` and compared it against Flower's
built-in FedAvg with `torch.allclose`. Result: **PASS**, worst difference
2.98e-08 — that's just floating-point rounding.

**Q: What does epsilon mean?**
→ It bounds how much one person's presence in the training data can change the
output distribution. ε = 1.14 at δ = 1e-5 is a strong practical guarantee. It's
a *quantified* promise, not an absolute one.

**Q: Does DP have a cost?**
→ Yes — recall dropped about 7.5 percentage points (0.842 → 0.768). That's the
privacy-utility tradeoff, and it's the whole point of running both versions.

**Q: Isn't accuracy 99.9% just because fraud is rare?**
→ Exactly right, and that's why I don't use accuracy as my headline metric.
Recall and AUC are what matter for a 0.17% fraud rate.

**Q: Is sharing weights 100% safe?**
→ No, and I don't claim it is. Weight sharing is *safer*, but gradient
inversion attacks exist. That's precisely why I added DP on top.

**Q: What are the limitations?**
→ Only 2 nodes (real systems use dozens); free-tier VMs are tiny (1 GB RAM), so
the cloud run is 5 rounds; the data comes from one public dataset split
artificially into two "banks", not two real institutions; no secure aggregation,
so the server still sees each update individually; honest-but-curious clients
only — I don't defend against a malicious bank.

**Q: What would you do next?**
→ Secure aggregation (server sees only the combined sum), more nodes, adaptive
privacy budgeting, streaming data instead of a fixed CSV, and robust aggregation
to resist poisoning.

**Q: Why is the orchestrator on GCP and not AWS?**
→ My laptop is behind carrier-grade NAT (can't accept incoming connections), and
a tunnel service like ngrok would have cost money. Putting the server on the
free-tier GCP VM gives a stable public IP for free. Both nodes dial *out* to it,
so the firewall only ever opens port 8080 to the AWS node's specific IP — never
to the whole internet.

**Q: What's the difference between your two clouds?**
→ AWS EC2 `t3.micro` runs bank 1's data. GCP `e2-micro` runs bank 2's data
*and* the orchestrator. Same client code on both — only a command-line flag
differs.

---

## 10. Suggested 60-second opening speech

> "Fraud detection needs more data than any single bank has, but PCI-DSS and
> GDPR stop banks from pooling customer transactions. My project, FedGuard,
> applies federated learning: the model travels to the data instead of the data
> travelling to the model.
>
> I deployed it across two real public clouds — an AWS EC2 instance and a Google
> Compute Engine VM — with a Flower orchestrator doing FedAvg aggregation. Each
> round, every node sends only about 4,000 model weights; an assertion in the
> code guarantees no raw transaction ever enters that payload, and my server log
> records both clients' real IP addresses connecting from the two different
> clouds.
>
> On top of that I added differential privacy with Opacus, which clips gradients
> and adds Gaussian noise, reaching epsilon 1.14 at delta 1e-5. The tradeoff is
> measurable: recall falls from 0.842 to 0.768 — that drop is the actual price
> of the privacy guarantee.
>
> I also re-implemented FedAvg by hand and verified it matches Flower's result
> to within 3 times 10 to the minus 8, and I built a Streamlit dashboard showing
> all the comparisons live."

---

## 11. Cheat-sheet of the numbers

| Thing | Value |
|---|---|
| Dataset | Credit Card Fraud, 284,807 rows, 30 features, 0.17% fraud |
| Split | Node 1 (AWS): 170,933 rows / 70% of fraud · Node 2 (GCP): 113,874 rows |
| Model | FraudNet MLP, 30→64→32→1, sigmoid output, ~4,097 parameters, CPU-only |
| Rounds | 10 local · 5 cross-cloud |
| DP settings | σ=1.0, clip C=1.0, δ=1e-5, batch 512, 1 epoch, lr 5e-3 |
| Final ε | **1.1377** |
| Best recall (FL, no DP) | **0.842** |
| Best AUC (cross-cloud) | **0.991** |
| Manual FedAvg check | **PASS**, max diff **2.98e-08** |
| Payload per round | **4,097 weights / 0 raw rows** |
| Cloud IPs | AWS `65.1.248.113` · GCP orchestrator `8.231.92.55` |
