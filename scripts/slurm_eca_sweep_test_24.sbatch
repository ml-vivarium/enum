#!/bin/bash
#SBATCH --job-name=eca-test-24
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH --mem=64G
#SBATCH --output=logs/eca-test-24-%j.out
#SBATCH --error=logs/eca-test-24-%j.err

set -euo pipefail
cd "$HOME/enum"

.venv/bin/python scripts/launch_eca_rule_sweep.py \
  --rules 0,1,2,3,4,5,6,7,8,9,10,11,14,15,18,19,22,23,24,25,26,27,28,29 \
  --output-root outputs/eca_rule_sweep_test_24_100 \
  --steps 100 \
  --checkpoint-every 100 \
  --gpus 8 \
  --python .venv/bin/python \
  --script rule110_tiny_transformer.py \
  --batch-size 1024 \
  --dtype bf16
