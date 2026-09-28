"""Phase 5 - FedAvg written out by hand, then checked against Flower's own math.

WHY WEIGHT BY SAMPLE COUNT (and not a plain average of the two nodes)?

FedAvg computes

    global[i] = (n1 * node1[i] + n2 * node2[i]) / (n1 + n2)

for every tensor element i, where n1/n2 are the number of training samples each
node used. A plain unweighted mean ((node1 + node2) / 2) would treat both nodes
as equally authoritative no matter how much data they actually hold. In this
project node1 holds 136,746 training rows and node2 holds 91,099, and in a real
deployment the skew is far larger (a regional bank vs. a branch office). Weighting
by n gives each institution influence over the global model proportional to the
evidence it contributed - which is exactly the property that makes federated
learning statistically comparable to centralized training.

This script proves the formula is what Flower actually executes: it loads the
local weights each node sent during a run, applies the weighted average by hand,
and asserts (torch.allclose) that the result is numerically identical to the
global parameters Flower's FedAvg strategy saved on the server.

Usage (after a run started with --save-weights on both server and clients):

    python src/manual_fedavg.py \
        --node1 results/node1_weights.pt \
        --node2 results/node2_weights.pt \
        --global-weights results/global_weights_5.pt
"""

import argparse
import glob
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def manual_weighted_average(node_states: list[dict], counts: list[int]) -> dict:
    """Hand-written FedAvg: per-element (sum_i n_i * w_i) / (sum_i n_i)."""
    if len(node_states) != len(counts):
        raise ValueError("each node needs a matching sample count")
    total = sum(counts)
    if total <= 0:
        raise ValueError("total sample count must be positive")

    averaged: dict[str, torch.Tensor] = {}
    for key in node_states[0]:
        acc = None
        for state, n in zip(node_states, counts):
            tensor = state[key].to(torch.float64) * float(n)
            acc = tensor if acc is None else acc + tensor
        averaged[key] = (acc / total).to(torch.float32)
    return averaged


def _latest_round_path(pattern: str) -> str:
    """Turn 'results/global_weights_{round}.pt' into the highest round on disk."""
    if "{round}" not in pattern:
        return pattern
    regex = re.compile(
        "^" + re.escape(pattern).replace(re.escape("{round}"), r"(\d+)") + "$"
    )
    matches = []
    for path in glob.glob(pattern.replace("{round}", "*")):
        m = regex.match(path)
        if m:
            matches.append((int(m.group(1)), path))
    if not matches:
        raise FileNotFoundError(f"no file matches {pattern}")
    return max(matches)[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-derive Flower's FedAvg aggregate by hand and verify it"
    )
    parser.add_argument("--node1", default="results/node1_weights.pt")
    parser.add_argument("--node2", default="results/node2_weights.pt")
    parser.add_argument("--global-weights", dest="global_weights",
                        default="results/global_weights_{round}.pt",
                        help="Flower's saved aggregate; '{round}' expands to the "
                             "latest round available")
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-6)
    args = parser.parse_args()

    print("=" * 70)
    print("PHASE 5 - Manual FedAvg vs Flower's FedAvg")
    print("=" * 70)

    node_files = [args.node1, args.node2]
    states, counts = [], []
    for path in node_files:
        blob = torch.load(path, map_location="cpu", weights_only=True)
        states.append(blob["state_dict"])
        counts.append(int(blob["num_examples"]))
        print(f"  loaded {path:<34} partition={blob.get('partition', '?'):<6} "
              f"n={counts[-1]:,}")

    global_path = _latest_round_path(args.global_weights)
    flower = torch.load(global_path, map_location="cpu", weights_only=True)
    flower_state = flower["state_dict"]
    print(f"  loaded {global_path:<34} server_round={flower.get('server_round', '?')}")

    print(f"\n  formula: global = ({counts[0]:,} * node1 + {counts[1]:,} * node2) "
          f"/ {sum(counts):,}")

    manual = manual_weighted_average(states, counts)

    print(f"\n  {'layer':<28} {'max abs diff':>14}  allclose")
    all_ok = True
    worst = 0.0
    for key in manual:
        diff = (manual[key].to(torch.float64) - flower_state[key].to(torch.float64)).abs().max().item()
        ok = torch.allclose(manual[key], flower_state[key],
                            rtol=args.rtol, atol=args.atol)
        all_ok &= ok
        worst = max(worst, diff)
        print(f"  {key:<28} {diff:>14.3e}  {'PASS' if ok else 'FAIL'}")

    print("\n" + "-" * 70)
    if all_ok:
        print(f"RESULT: PASS - manual FedAvg matches Flower's FedAvg "
              f"(worst element diff {worst:.3e}, rtol={args.rtol}, atol={args.atol}).")
    else:
        print(f"RESULT: FAIL - worst element diff {worst:.3e} exceeds tolerance.")
        sys.exit(1)


if __name__ == "__main__":
    main()
