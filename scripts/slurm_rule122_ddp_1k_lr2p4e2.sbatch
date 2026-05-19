#!/bin/bash
#SBATCH --job-name=rule122-ddp-1k-lr2p4e2
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH --mem=64G
#SBATCH --output=logs/rule122-ddp-1k-lr2p4e2-%j.out
#SBATCH --error=logs/rule122-ddp-1k-lr2p4e2-%j.err

set -euo pipefail
cd "$HOME/enum"

OUT=outputs/rule122_ddp_1k_lr2p4e2
mkdir -p "$OUT/checkpoints" "$OUT/attention"

.venv/bin/python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=8 \
  rule110_tiny_transformer.py \
  --task lm \
  --lm-direction reverse \
  --mask-row-prefix \
  --position-encoding grid \
  --rule 122 \
  --width 32 \
  --frames 32 \
  --layers 4 \
  --d-model 64 \
  --heads 4 \
  --batch-size 1024 \
  --steps 1000 \
  --checkpoint-every 1000 \
  --checkpoint-dir "$OUT/checkpoints" \
  --checkpoint-out "$OUT/checkpoints/final.pt" \
  --best-checkpoint-out "$OUT/checkpoints/best.pt" \
  --loss-curve-image "$OUT/loss.png" \
  --eval-every 100 \
  --eval-batches 4 \
  --lr 0.024 \
  --dtype bf16

.venv/bin/python rule110_tiny_transformer.py \
  --task lm \
  --lm-direction reverse \
  --mask-row-prefix \
  --position-encoding grid \
  --rule 122 \
  --width 32 \
  --frames 32 \
  --layers 4 \
  --d-model 64 \
  --heads 4 \
  --batch-size 1024 \
  --steps 0 \
  --checkpoint-in "$OUT/checkpoints/best.pt" \
  --attention-image "$OUT/attention/attention.png" \
  --attention-targets 16:16 \
  --attention-all-heads \
  --attention-order natural \
  --attention-overlay \
  --evolution-image "$OUT/attention/evolution.png" \
  --dtype bf16
