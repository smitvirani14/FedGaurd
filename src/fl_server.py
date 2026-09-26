"""Phase 2 - Flower orchestrator: FedAvg aggregation server.

Runs the central coordinator that cloud/localhost nodes connect to. It performs
NO training itself - it only distributes global weights and averages the updates
the nodes send back:

    global_weight = (n1 * node1_weight + n2 * node2_weight) / (n1 + n2)

(src/manual_fedavg.py re-derives this by hand in Phase 5.)

Usage:
    python src/fl_server.py --num_rounds 10
    python src/fl_server.py --server_address 0.0.0.0:8080   # accept remote VMs
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import flwr as fl  # noqa: E402
from flwr.common import ndarrays_to_parameters  # noqa: E402
from flwr.server import ServerConfig  # noqa: E402

from metrics_logger import (  # noqa: E402
    DEFAULT_CSV,
    log_round,
    reset_results_for_source,
)
from model import build_model  # noqa: E402


def weighted_average(results) -> dict:
    """Weighted mean of client metrics: sum(n_i * m_i) / sum(n_i).

    `results` is [(num_examples, metrics_dict), ...]. Weighting by sample count
    is what makes this a *federated* average rather than a naive one - the node
    holding more test data contributes proportionally more to the reported score.
    """
    if not results:
        return {}
    totals: dict[str, float] = {}
    weights: dict[str, float] = {}
    for n, metrics in results:
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                totals[key] = totals.get(key, 0.0) + float(value) * n
                weights[key] = weights.get(key, 0.0) + n
    return {k: totals[k] / weights[k] for k in totals if weights[k]}


class LoggingFedAvg(fl.server.strategy.FedAvg):
    """FedAvg that writes one metrics row per round to results/results.csv."""

    def __init__(self, source: str, results_path=None, **kwargs):
        super().__init__(**kwargs)
        self.source = source
        self.results_path = results_path
        self.last_fit_metrics: dict = {}

    def aggregate_fit(self, server_round, results, failures):
        aggregated = super().aggregate_fit(server_round, results, failures)
        if aggregated is not None:
            self.last_fit_metrics = aggregated[1]
            eps = aggregated[1].get("epsilon")
            print(f"[server] round {server_round}: aggregated fit metrics "
                  f"{ {k: round(v, 4) for k, v in aggregated[1].items() if isinstance(v, (int, float))} }",
                  flush=True)
            if eps is not None:
                print(f"[server] round {server_round}: mean client epsilon = {eps:.4f}", flush=True)
        return aggregated

    def aggregate_evaluate(self, server_round, results, failures):
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)
        if metrics:
            # A metrics-logging failure must never abort a long training run.
            try:
                log_round(
                    server_round, self.source,
                    metrics.get("accuracy"), metrics.get("precision"),
                    metrics.get("recall"), metrics.get("f1"), metrics.get("auc"),
                    epsilon=metrics.get("epsilon", self.last_fit_metrics.get("epsilon")),
                    results_path=self.results_path or DEFAULT_CSV,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[server] WARNING: could not log round {server_round}: {exc}", flush=True)
            print(f"[server] round {server_round}: logged {self.source} row "
                  f"(acc={metrics.get('accuracy', float('nan')):.4f} "
                  f"recall={metrics.get('recall', float('nan')):.4f} "
                  f"auc={metrics.get('auc', float('nan')):.4f})", flush=True)
        return loss, metrics


def build_strategy(source: str, results_path=None) -> LoggingFedAvg:
    """FedAvg with the two callbacks the plan requires."""
    return LoggingFedAvg(
        source=source,
        results_path=results_path,
        fraction_fit=1.0,            # both nodes train every round
        fraction_evaluate=1.0,       # both nodes evaluate every round
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,     # server blocks until BOTH nodes are connected
        evaluate_metrics_aggregation_fn=weighted_average,   # weighted acc/recall/AUC
        fit_metrics_aggregation_fn=weighted_average,        # carries epsilon in Phase 3
        initial_parameters=ndarrays_to_parameters(
            [v.detach().cpu().numpy() for v in build_model().state_dict().values()]
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="FedGuard Flower orchestrator")
    parser.add_argument("--num_rounds", "--num-rounds", dest="num_rounds", type=int, default=10)
    parser.add_argument("--server_address", "--server-address", dest="server_address",
                        default="0.0.0.0:8080",
                        help="bind address; use 0.0.0.0:8080 to accept remote cloud VMs")
    parser.add_argument("--source", default="federated",
                        help="results.csv source tag for this run "
                             "(federated / federated_dp / federated_nodp)")
    parser.add_argument("--results_path", default=None, help="override results CSV path")
    parser.add_argument("--append", action="store_true",
                        help="append to existing rows for this source instead of replacing them")
    args = parser.parse_args()

    if not args.append:
        reset_results_for_source(args.source, args.results_path or DEFAULT_CSV)

    strategy = build_strategy(args.source, args.results_path)

    print("=" * 70)
    print(f"PHASE 2 - FedAvg orchestrator  |  rounds={args.num_rounds}  "
          f"bind={args.server_address}  source={args.source}")
    print("=" * 70, flush=True)

    history = fl.server.start_server(
        server_address=args.server_address,
        config=ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
        client_manager=fl.server.SimpleClientManager(),
    )

    print("\n[server] run complete. Round-by-round history:")
    for metric, values in history.metrics_centralized.items():
        if values:
            first, last = values[0], values[-1]
            print(f"  {metric:<10} round {first[0]:>2}: {first[1]:.4f}  ->  "
                  f"round {last[0]:>2}: {last[1]:.4f}")


if __name__ == "__main__":
    main()
