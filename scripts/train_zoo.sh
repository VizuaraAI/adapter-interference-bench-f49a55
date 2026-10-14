#!/usr/bin/env bash
# Train every adapter in the zoo against the shared frozen base.
#
# Start small: set zoo.num_adapters to 4 and train.max_train_samples to 500
# in configs/default.yaml to get something working today, then scale up once
# the pipeline is trustworthy.
set -euo pipefail

CONFIG="${1:-configs/default.yaml}"

echo "training the adapter zoo defined in ${CONFIG}"
echo "base model: $(python -c "
import yaml, sys
print(yaml.safe_load(open('${CONFIG}'))['base_model']['name'])
")"

python -m src.train_adapter --config "${CONFIG}" --all

echo
echo "adapters saved under artifacts/adapters/"
ls -1 artifacts/adapters/ 2>/dev/null || echo "(nothing found: did training fail?)"
