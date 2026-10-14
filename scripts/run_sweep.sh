#!/usr/bin/env bash
# Run the full sweep: solo reference for every adapter, then every co-served
# configuration. Writes results/sweep.csv (append-only) and results/table.md.
set -euo pipefail

CONFIG="${1:-configs/default.yaml}"

if [ ! -d artifacts/adapters ]; then
  echo "no adapters found in artifacts/adapters/"
  echo "run scripts/train_zoo.sh first, or use --simulate for a dry run:"
  echo "  python -m src.measure --config ${CONFIG} --simulate --adapters 4"
  exit 1
fi

echo "=== sweep starting ==="
python -m src.measure --config "${CONFIG}"

echo
echo "=== results ==="
echo "raw rows:  results/sweep.csv"
echo "summary:   results/table.md"
echo
echo "Read worst_ids before mean_ids. The worst adapter is the one that"
echo "loses a customer."
