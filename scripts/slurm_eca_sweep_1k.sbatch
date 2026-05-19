#!/bin/bash
#SBATCH --job-name=eca-sweep-1k
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/eca-sweep-1k-%j.out
#SBATCH --error=logs/eca-sweep-1k-%j.err

set -euo pipefail
cd "$HOME/enum"

.venv/bin/python scripts/launch_eca_rule_sweep.py \
  --output-root outputs/eca_rule_sweep_1k \
  --steps 1000 \
  --checkpoint-every 1000 \
  --gpus 8 \
  --python .venv/bin/python \
  --script rule110_tiny_transformer.py \
  --batch-size 1024 \
  --dtype bf16
