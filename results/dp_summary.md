# Phase 3 — Differential Privacy Summary

**DP-SGD settings (identical on both nodes)**

| Parameter | Value | Meaning |
|---|---|---|
| `noise_multiplier` (σ) | 1.0 | Gaussian noise std = 1.0 × clip bound |
| `max_grad_norm` (C) | 1.0 | per-sample gradient clipping bound |
| `delta` (δ) | 1e-5 | allowed probability of failure of the DP guarantee |
| Local epochs / round | 1 | same as the non-DP run |
| Batch size | 512 | same as the non-DP run |
| Learning rate | **5e-3** | see note below |
| Rounds | 10 | same as the non-DP run |

> **Why the DP learning rate differs:** DP-SGD's injected noise inflates Adam's
> second-moment estimate, which shrinks every subsequent step. Measured on
> node1's partition over 10 rounds at the *same* ε: `lr=1e-3` → recall 0.00,
> `lr=5e-3` → recall 0.67. Only the learning rate differs between the two runs;
> data, epochs, batch size, loss and class weighting are identical, so the
> comparison remains an apples-to-apples privacy/utility measurement.

## Final results (round 10 of 10)

| Run | Accuracy | Precision | Recall | F1 | AUC | ε |
|---|---|---|---|---|---|---|
| Federated, no DP | 0.9994 | 0.8233 | 0.8423 | 0.8294 | 0.9902 | — |
| Federated + DP | 0.9993 | 0.8088 | **0.7675** | 0.7851 | **0.9327** | **1.1377** |

**Final ε = 1.1377 at δ = 1e-5**, with accuracy 0.9993.

## Privacy budget over rounds

| Round | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| ε (mean of both nodes) | 0.4157 | 0.5443 | 0.6467 | 0.7352 | 0.8148 | 0.8877 | 0.9556 | 1.0195 | 1.0800 | 1.1377 |

ε increases monotonically — every extra round spends more of the privacy
budget, and the growth is sub-linear because of advanced composition. This is
expected DP behaviour, not a bug.

## The privacy–utility tradeoff

- Recall drops **0.8423 → 0.7675** and AUC drops **0.9902 → 0.9327** when DP is
  switched on: that is the *price* of the guarantee that gradients cannot be
  inverted back into raw transactions.
- Accuracy barely moves (0.9994 → 0.9993) because at 0.17% fraud rate accuracy
  is dominated by the negative class — recall and AUC are the numbers that
  actually carry the signal.
- An ε of ≈1.14 for δ=1e-5 is considered a *strong* practical guarantee
  (research deployments commonly report ε in the 1–8 range).

Raw per-run extracts: `results/federated_dp.csv`, `results/federated_nodp.csv`.
All runs combined: `results/results.csv`.
