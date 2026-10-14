#!/usr/bin/env bash
# The two-minute scaffolding check. No GPU, no trained adapters, no dataset
# downloads. Run this after any change to src/ before you spend GPU hours.
set -euo pipefail

CONFIG="${1:-configs/default.yaml}"

echo "=== import check ==="
python -c "
from src import adapters, metrics, policies, serving, measure
print('all modules import cleanly')
"

echo
echo "=== zoo construction ==="
python -c "
import yaml
from src.adapters import build_zoo, working_set_bytes
cfg = yaml.safe_load(open('${CONFIG}'))
zoo = build_zoo(cfg)
total = working_set_bytes(zoo, 1536, 28)
print(f'{len(zoo)} adapters, working set = {total/1e6:.1f} MB')
for s in zoo[:4]:
    print(f'  {s.adapter_id:<28} r={s.rank:<3} {s.family:<14} {s.metric}')
"

echo
echo "=== simulator sweep (resource channel only) ==="
python -m src.measure --config "${CONFIG}" --simulate --adapters 4

echo
echo "smoke test passed"
