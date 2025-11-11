#!/usr/bin/env bash
set -euo pipefail
python -m alchemi.cli pretrain_mae --config configs/train.mae.yaml
