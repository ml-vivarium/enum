# Transformer CA Experiments

Tiny PyTorch transformer experiments for elementary cellular automata.

The main script is:

```bash
rule110_tiny_transformer.py
```

Despite the filename, the script supports arbitrary elementary rules with `--rule`
or `--compare-rules`.

## Setup

Use `uv`:

```bash
uv sync
```

Then run commands with:

```bash
uv run python rule110_tiny_transformer.py ...
```

The script uses PyTorch and automatically selects CUDA when available:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

## Forward LM

Flatten a generated spacetime diagram and train a causal LM to predict the next
cell token. Grid-aware positions are enabled by default:

```text
token embedding
+ source frame/cell embedding
+ target frame/cell embedding
```

Example:

```bash
uv run python rule110_tiny_transformer.py \
  --task lm \
  --lm-direction forward \
  --position-encoding grid \
  --rule 110 \
  --steps 1000 \
  --batch-size 64 \
  --width 32 \
  --frames 8 \
  --eval-every 100 \
  --eval-batches 4 \
  --lr 0.001
```

The random initial frame is masked from the loss.

## Reverse LM

Generate forward evolution, reverse the frame order, then train the LM to predict
earlier states from later states:

```bash
uv run python rule110_tiny_transformer.py \
  --task lm \
  --lm-direction reverse \
  --position-encoding grid \
  --rule 110 \
  --steps 1000 \
  --batch-size 64 \
  --width 32 \
  --frames 8 \
  --eval-every 100 \
  --eval-batches 4 \
  --lr 0.001
```

This vanilla reverse LM leaks same-row teacher-forced prefixes. It measures:

```text
P(row_t[i] | future rows, row_t[0:i])
```

## Row-Prefix-Masked Reverse LM

Use `--mask-row-prefix` to prevent a target cell from attending to true cells in
the same row:

```bash
uv run python rule110_tiny_transformer.py \
  --task lm \
  --lm-direction reverse \
  --mask-row-prefix \
  --position-encoding grid \
  --compare-rules 30,110 \
  --steps 1000 \
  --batch-size 64 \
  --width 32 \
  --frames 8 \
  --eval-every 50 \
  --eval-batches 4 \
  --lr 0.001 \
  --loss-curve-image rule30_rule110_reverse_loss_rowmasked.png
```

This measures something closer to:

```text
P(row_t[i] | future rows only)
```

CPU results from the initial run:

```text
Rule 30:  eval_loss=0.3509 eval_acc=0.820
Rule 110: eval_loss=0.3610 eval_acc=0.802
```

The generated curve is:

```text
rule30_rule110_reverse_loss_rowmasked.png
```

## Reconstruction Images

For reverse LM runs, the script can produce autoregressive reconstruction images:

```bash
uv run python rule110_tiny_transformer.py \
  --task lm \
  --lm-direction reverse \
  --position-encoding grid \
  --rule 110 \
  --steps 1000 \
  --batch-size 64 \
  --width 32 \
  --frames 8 \
  --eval-every 200 \
  --eval-batches 4 \
  --lr 0.001 \
  --reconstruction-image rule110_reverse_reconstruction.png \
  --reconstruction-samples 4 \
  --image-scale 12
```

Each image contains:

```text
top:    true forward evolution
middle: autoregressive reverse reconstruction
bottom: error mask, red = wrong cell
```

## GPU Rerun Suggestions

Start by repeating the CPU runs exactly on GPU, then scale one axis at a time:

```text
width:  32 -> 64 -> 128
frames: 8  -> 16 -> 32
layers: 2  -> 4
d_model: 64 -> 128
```

Useful first GPU command:

```bash
uv run python rule110_tiny_transformer.py \
  --task lm \
  --lm-direction reverse \
  --mask-row-prefix \
  --position-encoding grid \
  --compare-rules 30,110 \
  --steps 5000 \
  --batch-size 512 \
  --width 64 \
  --frames 16 \
  --layers 4 \
  --d-model 128 \
  --heads 4 \
  --eval-every 100 \
  --eval-batches 8 \
  --lr 0.001 \
  --loss-curve-image rule30_rule110_reverse_loss_rowmasked_gpu.png
```

The current code uses learned absolute frame/cell embeddings, so results are tied
to the configured width and frame count.
