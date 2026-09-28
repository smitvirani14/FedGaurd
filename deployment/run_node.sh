#!/bin/bash
# FedGuard - start one cloud node's Flower client.
#
# Usage:
#   ./run_node.sh <node1|node2> <orchestrator_host> [port] [--use-dp|--no-dp]
#
# Examples:
#   ./run_node.sh node2 localhost            # node co-located with the orchestrator
#   ./run_node.sh node1 8.231.92.55 8080     # remote cross-cloud node
#   ./run_node.sh node1 8.231.92.55 8080 --use-dp
#
# This script is cloud-agnostic: the exact same file runs on the AWS VM and on
# the GCP VM. Only the partition name and orchestrator address differ.
set -euo pipefail

PARTITION="${1:-}"
HOST="${2:-}"
PORT="${3:-8080}"
EXTRA_ARGS=("${@:4}")

if [[ -z "$PARTITION" || -z "$HOST" ]]; then
  echo "usage: $0 <node1|node2> <orchestrator_host> [port] [--use-dp|--no-dp]" >&2
  exit 1
fi
if [[ "$PARTITION" != "node1" && "$PARTITION" != "node2" ]]; then
  echo "error: partition must be node1 or node2 (got '$PARTITION')" >&2
  exit 1
fi

# Prefer a local virtualenv if one exists, then the user's home venv,
# otherwise fall back to python3.
if [[ -x ".venv/bin/python" ]]; then
  PYTHON=".venv/bin/python"
elif [[ -x "$HOME/.venv/bin/python" ]]; then
  PYTHON="$HOME/.venv/bin/python"
else
  PYTHON="python3"
fi

DATA_FILE="data/${PARTITION}_partition.csv"
if [[ ! -f "$DATA_FILE" && ! -f "${PARTITION}_partition.csv" ]]; then
  echo "error: $DATA_FILE not found - copy the node's partition to this VM first" >&2
  exit 1
fi

SERVER_ADDRESS="${HOST}:${PORT}"
echo "[run_node] partition=${PARTITION} server=${SERVER_ADDRESS} dp=${*:-off}"
exec "$PYTHON" src/fl_client.py --partition "$PARTITION" \
  --server_address "$SERVER_ADDRESS" "${EXTRA_ARGS[@]}"
