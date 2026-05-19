# Transformer CA Experiment Log

## 2026-05-18

### Setup

- Repo branch: `codex/jax-ca-core`
- Main script: `rule110_tiny_transformer.py`
- GPU host: `kboguta-dev-a100-mjtpu`
- GCP project: `midjourney-tpu`
- GPU: NVIDIA A100-SXM4-40GB
- Driver: 580.126.20
- PyTorch on GPU: 2.12.0+cu130

### Task

Row-prefix-masked reverse LM for elementary cellular automata.

For Rule 110:

```text
Generate row0 -> row1 -> ... -> row7
Reverse sequence to row7 -> row6 -> ... -> row0
Predict previous rows from future rows only
Mask same-row teacher-forced prefix
```

### Throughput Checks

Matched CPU/GPU benchmark:

```bash
uv run python rule110_tiny_transformer.py \
  --task lm \
  --lm-direction reverse \
  --mask-row-prefix \
  --position-encoding grid \
  --rule 110 \
  --steps 500 \
  --batch-size 64 \
  --width 32 \
  --frames 8 \
  --eval-every 500 \
  --eval-batches 1 \
  --lr 0.001
```

Results:

```text
Local CPU:
  batch_size=64
  elapsed=63.2s
  steps/sec=7.9

A100 GPU:
  batch_size=64
  elapsed=4.9s
  steps/sec=102.0
  samples/sec=6,531
```

GPU batch-size check:

```text
batch_size=64:
  samples/sec ~= 6,531

batch_size=1024:
  samples/sec ~= 17,239 before quick fixes
  samples/sec ~= 17,675 after cached masks/rule-table
  samples/sec ~= 26,506 with bf16
```

Generator-only benchmark on A100:

```text
batch_size=64:
  generator samples/sec ~= 67,704

batch_size=1024:
  generator samples/sec ~= 1,094,286
```

Conclusion: current runs are model forward/backward-bound, not CA data-generation-bound.

### Rule 110 GPU Training Curve

Command:

```bash
uv run python rule110_tiny_transformer.py \
  --task lm \
  --lm-direction reverse \
  --mask-row-prefix \
  --position-encoding grid \
  --rule 110 \
  --steps 5000 \
  --batch-size 1024 \
  --width 32 \
  --frames 8 \
  --eval-every 100 \
  --eval-batches 4 \
  --lr 0.001 \
  --dtype bf16 \
  --loss-curve-image rule110_reverse_rowmasked_bs1024_bf16_loss.png
```

Final:

```text
step=5000
train_loss=0.2357
eval_loss=0.2338
eval_acc=0.884
train_elapsed=188.3s
train_steps_per_sec=26.55
train_samples_per_sec=27,187
```

Artifact:

```text
rule110_reverse_rowmasked_bs1024_bf16_loss.png
```

### Rule 110 Learning Rate Sweep

Fixed settings:

```text
rule=110
task=reverse row-prefix-masked LM
batch_size=1024
width=32
frames=8
layers=2
d_model=64
heads=4
dtype=bf16
steps=5000
eval_every=100
eval_batches=4
```

Commands varied only `--lr` and `--loss-curve-image`.

Results:

```text
lr=3e-4:
  eval_loss=0.2678
  eval_acc=0.864
  train_steps_per_sec=26.52
  train_samples_per_sec=27,161
  artifact=rule110_reverse_rowmasked_bs1024_bf16_lr3e4_loss.png

lr=1e-3:
  eval_loss=0.2330
  eval_acc=0.885
  train_steps_per_sec=26.56
  train_samples_per_sec=27,194
  artifact=rule110_reverse_rowmasked_bs1024_bf16_lr1e3_loss.png

lr=3e-3:
  eval_loss=0.2091
  eval_acc=0.898
  train_steps_per_sec=26.57
  train_samples_per_sec=27,203
  artifact=rule110_reverse_rowmasked_bs1024_bf16_lr3e3_loss.png
```

Takeaway: `lr=3e-3` was the best of this initial sweep and was still stable at
5000 steps. Next LR candidates: `6e-3`, `1e-2`, and possibly a schedule around
`3e-3`.

### Rule 110 Higher Learning Rate Probe

Same fixed settings as the previous LR sweep.

Results:

```text
lr=6e-3:
  eval_loss=0.2035
  eval_acc=0.900
  train_steps_per_sec=26.56
  train_samples_per_sec=27,198
  artifact=rule110_reverse_rowmasked_bs1024_bf16_lr6e3_loss.png

lr=1e-2:
  eval_loss=0.1868
  eval_acc=0.908
  train_steps_per_sec=26.56
  train_samples_per_sec=27,199
  artifact=rule110_reverse_rowmasked_bs1024_bf16_lr1e2_loss.png
```

Takeaway: `lr=1e-2` was still stable and is the best so far at 5000 steps.
Next LR candidates: `2e-2`, `3e-2`. If those destabilize, use `1e-2` as the
fixed LR or try a schedule with warmup/decay.

### Rule 110 LR Break-Point Probe

Same fixed settings as the previous LR sweep.

Results:

```text
lr=2e-2:
  eval_loss=0.2809
  eval_acc=0.851
  train_steps_per_sec=26.53
  train_samples_per_sec=27,167
  artifact=rule110_reverse_rowmasked_bs1024_bf16_lr2e2_loss.png

lr=3e-2:
  killed at step 1200
  eval_loss at step 1200=0.6537
  eval_acc at step 1200=0.594
```

Takeaway: `2e-2` is too hot and noisy, and `3e-2` is clearly bad. Current best
fixed LR remains `1e-2`.

### Rule 110 Width Probe at Fixed LR

Fixed settings:

```text
task=lm
lm_direction=reverse
mask_row_prefix=True
position_encoding=grid
rule=110
steps=5000
batch_size=1024
width=32
frames=8
layers=2
heads=4
lr=1e-2
dtype=bf16
```

Results:

```text
d_model=32:
  parameters=28,290
  eval_loss=0.2632
  eval_acc=0.865
  train_steps_per_sec=33.46
  train_samples_per_sec=34,259
  artifact=rule110_reverse_rowmasked_bs1024_bf16_lr1e2_dm32_loss.png

d_model=64:
  parameters=106,338
  eval_loss=0.1868
  eval_acc=0.908
  train_steps_per_sec=26.56
  train_samples_per_sec=27,199
  artifact=rule110_reverse_rowmasked_bs1024_bf16_lr1e2_loss.png

d_model=128:
  parameters=408,066
  killed during early run at lr=1e-2
  step_500_eval_loss=0.6090
```

Takeaway: `d_model=32` is faster but clearly worse. `d_model=128` at the same
fixed LR looked too hot and was abandoned rather than treated as a fair capacity
comparison. The current best width under this fixed-LR setup remains
`d_model=64`.

### Rule 110 Depth Probe at Fixed LR

Fixed settings are the same as the width probe, except `d_model=64` and varying
`layers`.

Results:

```text
layers=1:
  parameters=55,618
  eval_loss=0.2307
  eval_acc=0.886
  train_steps_per_sec=42.11
  train_samples_per_sec=43,122
  artifact=rule110_reverse_rowmasked_bs1024_bf16_lr1e2_l1_loss.png

layers=2:
  parameters=106,338
  eval_loss=0.1868
  eval_acc=0.908
  train_steps_per_sec=26.56
  train_samples_per_sec=27,199
  artifact=rule110_reverse_rowmasked_bs1024_bf16_lr1e2_loss.png

layers=4:
  parameters=205,954
  eval_loss=0.2433
  eval_acc=0.874
  train_steps_per_sec=15.25
  train_samples_per_sec=15,613
  artifact=rule110_reverse_rowmasked_bs1024_bf16_lr1e2_l4_loss.png
```

Takeaway: `layers=1` is efficient and learns a lot, but underfits relative to
the 2-layer baseline. `layers=4` initially improved quickly but had large
instability spikes at fixed `lr=1e-2`, so it likely needs a lower LR and/or
warmup schedule before capacity comparisons are meaningful. Current best clean
run remains `layers=2, d_model=64, lr=1e-2`.

### Rule 110 Bigger Models at Lower LR

Fixed settings:

```text
task=lm
lm_direction=reverse
mask_row_prefix=True
position_encoding=grid
rule=110
steps=5000
batch_size=1024
width=32
frames=8
heads=4
lr=3e-3
dtype=bf16
```

Results:

```text
layers=2, d_model=128:
  parameters=408,066
  eval_loss=0.2124
  eval_acc=0.895
  train_steps_per_sec=19.33
  train_samples_per_sec=19,793
  artifact=rule110_reverse_rowmasked_bs1024_bf16_lr3e3_dm128_loss.png

layers=4, d_model=64:
  parameters=205,954
  eval_loss=0.1729
  eval_acc=0.915
  train_steps_per_sec=15.26
  train_samples_per_sec=15,624
  artifact=rule110_reverse_rowmasked_bs1024_bf16_lr3e3_l4_loss.png
```

Takeaway: lowering LR fixes the instability for the deeper model. The
4-layer/64-wide model at `lr=3e-3` is the best result so far, beating the
previous best `layers=2, d_model=64, lr=1e-2` run (`eval_loss=0.1868`,
`eval_acc=0.908`). The wider `d_model=128` run is stable at `3e-3`, but does
not beat the smaller baseline within 5000 steps.

### Rule 110 32x32 Shape Run

Changed the flattened grid from the earlier `32x8` setup to a square
`32x32` setup:

```text
task=lm
lm_direction=reverse
mask_row_prefix=True
position_encoding=grid
rule=110
steps=5000
batch_size=1024
width=32
frames=32
layers=4
d_model=64
heads=4
lr=3e-3
dtype=bf16
context_len=1023
```

Batch-size probe before the run:

```text
bs=128:  train_samples_per_sec=1,522
bs=256:  train_samples_per_sec=1,854
bs=512:  train_samples_per_sec=2,116
bs=1024: train_samples_per_sec=2,244
```

Completed result:

```text
parameters=209,026
eval_loss=0.0928
eval_acc=0.957
train_steps_per_sec=2.33
train_samples_per_sec=2,387
artifact=rule110_reverse_rowmasked_32x32_bs1024_bf16_lr3e3_l4d64_loss.png
```

Live GPU check during the run showed `100%` GPU utilization and about
`13.9GB / 40GB` memory used, so this dense-attention setup was compute-bound
rather than memory-bound at `bs=1024`.

Takeaway: the `32x32` square task trained cleanly and reached much better
per-cell accuracy than the `32x8` runs, though whole-grid reconstruction quality
still needs to be measured separately.

### Rule 110 32x32 Checkpointed Attention Smoke Run

Purpose: rerun the `32x32` setup briefly with checkpoint and attention export
enabled so prediction attention patterns can be inspected.

Settings:

```text
task=lm
lm_direction=reverse
mask_row_prefix=True
position_encoding=grid
rule=110
steps=1000
batch_size=1024
width=32
frames=32
layers=4
d_model=64
heads=4
lr=3e-3
dtype=bf16
eval_batches=4
context_len=1023
```

Result:

```text
parameters=209,026
eval_loss=0.2019
eval_acc=0.905
train_steps_per_sec=2.33
train_samples_per_sec=2,384
checkpoint=rule110_reverse_rowmasked_32x32_bs1024_bf16_lr3e3_l4d64_1k.pt
loss_curve=rule110_reverse_rowmasked_32x32_bs1024_bf16_lr3e3_l4d64_1k_loss.png
attention_targets=1:16,8:16,16:16,31:16
```

Attention artifacts:

```text
rule110_reverse_rowmasked_32x32_bs1024_bf16_lr3e3_l4d64_1k_attention_f001_c016.png
rule110_reverse_rowmasked_32x32_bs1024_bf16_lr3e3_l4d64_1k_attention_f008_c016.png
rule110_reverse_rowmasked_32x32_bs1024_bf16_lr3e3_l4d64_1k_attention_f016_c016.png
rule110_reverse_rowmasked_32x32_bs1024_bf16_lr3e3_l4d64_1k_attention_f031_c016.png
```

### Rule 30 32x32 Checkpointed 1k Run

Mirrors the Rule 110 checkpointed 1k setup, changing only the CA rule.

Settings:

```text
task=lm
lm_direction=reverse
mask_row_prefix=True
position_encoding=grid
rule=30
steps=1000
batch_size=1024
width=32
frames=32
layers=4
d_model=64
heads=4
lr=3e-3
dtype=bf16
eval_batches=4
context_len=1023
```

Result:

```text
parameters=209,026
eval_loss=0.3129
eval_acc=0.842
train_steps_per_sec=2.33
train_samples_per_sec=2,384
checkpoint=rule30_reverse_rowmasked_32x32_bs1024_bf16_lr3e3_l4d64_1k.pt
loss_curve=rule30_reverse_rowmasked_32x32_bs1024_bf16_lr3e3_l4d64_1k_loss.png
```

Takeaway: Rule 30 reverse prediction is much harder than Rule 110 at the same
step budget and model size. The learning curve was more stage-like: near chance
through step 200, a jump around step 300, a plateau around 500-600, then another
jump by 700-900.

### Rule 30 32x32 10k Run

Same fixed settings as the Rule 30 1k run, extended to 10k steps. This run was
launched detached on the A100 so it could continue after local disconnect.

Result:

```text
parameters=209,026
eval_loss=0.1116
eval_acc=0.944
train_steps_per_sec=2.33
train_samples_per_sec=2,388
checkpoint=rule30_reverse_rowmasked_32x32_bs1024_bf16_lr3e3_l4d64_10k.pt
loss_curve=rule30_reverse_rowmasked_32x32_bs1024_bf16_lr3e3_l4d64_10k_loss.png
log=rule30_reverse_rowmasked_32x32_bs1024_bf16_lr3e3_l4d64_10k.log
```

Notable instability events:

```text
step=3500: eval_loss=0.3264 eval_acc=0.828
step=4600: eval_loss=0.6807 eval_acc=0.585
step=6000: eval_loss=0.2549 eval_acc=0.868
step=7000: eval_loss=0.3486 eval_acc=0.800
```

Takeaway: despite repeated instability/reset events, the model recovered and
finished at the best observed Rule 30 performance so far.

### Rule 30 10k Per-Frame Accuracy

Teacher-forced inference over 10,000 random initial conditions using the final
Rule 30 10k checkpoint. Accuracy is grouped by natural evolution frame. In the
reverse-LM setup, frame 31 is the revealed/final row and is not a prediction
target, so it has count 0.

Artifacts:

```text
csv=rule30_reverse_rowmasked_32x32_bf16_lr3e3_l4d64_10k_frame_accuracy_10kics.csv
image=rule30_reverse_rowmasked_32x32_bf16_lr3e3_l4d64_10k_frame_accuracy_10kics.png
```

Summary:

```text
samples=10,000
cells_per_scored_frame=320,000
overall_scored_accuracy=0.9441
best_frame=22 accuracy=0.9464
worst_scored_frame=30 accuracy=0.9312
frame_29_accuracy=0.9366
frame_31=revealed/not scored
```

Takeaway: accuracy is nearly flat around 94.3-94.6% from natural frames 0-28,
then drops for frames 29-30, the latest predicted rows adjacent to the revealed
final row.

### B200 1-GPU Throughput Benchmark

Date: 2026-05-19

Ran on Lambda B200 cluster via Slurm using one NVIDIA B200. Same basic 32x32
reverse-LM setup as the A100 Rule 30/Rule 110 runs, with bf16 and batch size
1024. Used the existing local venv directly because `uv run` was resolving a
CUDA 13.0 torch build incompatible with the node driver.

Config:

```text
hardware=NVIDIA B200
torch=2.11.0+cu128
device=cuda
rule=30
task=lm
lm_direction=reverse
mask_row_prefix=True
position_encoding=grid
width=32
frames=32
layers=4
d_model=64
heads=4
parameters=209,026
batch_size=1024
steps=500
dtype=bf16
compile=False
```

Throughput:

```text
step=100 train_steps_per_sec=4.47 train_samples_per_sec=4,577
step=200 train_steps_per_sec=4.90 train_samples_per_sec=5,013
step=300 train_steps_per_sec=5.06 train_samples_per_sec=5,177
step=400 train_steps_per_sec=5.14 train_samples_per_sec=5,263
step=500 train_steps_per_sec=5.19 train_samples_per_sec=5,316
```

Comparison to previous A100 32x32 baseline:

```text
A100 baseline ~=2.33 steps/sec ~=2,388 samples/sec
B200 step-500 =5.19 steps/sec =5,316 samples/sec
speedup ~=2.23x by samples/sec
```

Takeaway: for this tiny 209k-parameter 32x32 transformer workload, one B200 is
about 2.2x faster than the A100 baseline. This is almost certainly still a small
model / overhead-heavy regime, so larger models or multi-GPU runs may show a
different scaling profile.

## Handoff Addendum: Reverse CA Transformer Experiments

Date: 2026-05-19

This addendum records the experiment sequence that happened after the B200
single-GPU benchmark. See `HANDOFF.md` for the compact narrative and next-agent
context.

### Code State

Recent committed capabilities in `rule110_tiny_transformer.py`:

```text
DDP training via torch.distributed.run
checkpointing every N steps plus best/final checkpoints
attention overlays in natural or model order
per-frame teacher-forced accuracy
transition consistency eval
full autoregressive reverse eval
cached autoregressive reverse eval
one-row autoregressive reverse eval
row-masked soft consistency loss
```

Generated artifacts are intentionally ignored by git. Results should be captured
in markdown summaries or kept in run directories under `outputs/`.

### 88-Rule B200 Sweep

Run:

```text
job=192295
script=scripts/slurm_eca_sweep_1k.sbatch
output_root=/home/kboguta_midjourney_com/enum/outputs/eca_rule_sweep_1k
rules=88
steps=1000 per rule
checkpoint_every=1000
gpus=8 independent 1-GPU workers
batch_size=1024
dtype=bf16
elapsed=00:40:39
```

Each completed rule directory contains:

```text
loss.png
train.log
attention.log
checkpoints/best.pt
checkpoints/final.pt
checkpoints/ruleXXX_step001000.pt
attention/evolution.png
16 attention overlays for natural frame 16 cell 16
```

Sorted final 1k accuracies, lowest to highest:

```text
060 0.500 0.6914
122 0.646 0.5778
129 0.677 0.5420
126 0.710 0.4990
022 0.720 0.5017
090 0.742 0.3569
105 0.743 0.3581
150 0.758 0.3346
075 0.792 0.3030
146 0.803 0.3592
030 0.861 0.2683
110 0.863 0.2774
018 0.870 0.2597
041 0.889 0.2354
045 0.894 0.1683
147 0.916 0.1925
005 0.926 0.2585
054 0.926 0.1705
061 0.936 0.1533
073 0.943 0.1235
109 0.946 0.1158
026 0.948 0.1189
057 0.964 0.0974
091 0.964 0.0836
037 0.968 0.0761
058 0.970 0.1008
134 0.974 0.0693
025 0.975 0.0616
094 0.975 0.0617
152 0.976 0.0593
036 0.978 0.0355
062 0.978 0.0598
164 0.978 0.0534
104 0.979 0.0517
009 0.980 0.0485
074 0.980 0.0449
056 0.981 0.0472
024 0.982 0.0261
078 0.982 0.0496
000 0.984 0.0224
156 0.985 0.0391
006 0.986 0.0364
028 0.986 0.0375
033 0.986 0.0318
044 0.986 0.0303
108 0.986 0.0318
123 0.987 0.0286
008 0.988 0.0255
032 0.988 0.0262
040 0.988 0.0311
072 0.989 0.0239
128 0.989 0.0252
046 0.990 0.0187
130 0.990 0.0243
004 0.991 0.0142
050 0.991 0.0240
077 0.991 0.0267
132 0.991 0.0210
001 0.992 0.0139
002 0.992 0.0132
007 0.992 0.0233
014 0.992 0.0189
127 0.992 0.0137
160 0.992 0.0249
010 0.993 0.0117
011 0.993 0.0156
023 0.993 0.0230
027 0.993 0.0206
038 0.993 0.0139
043 0.993 0.0197
079 0.993 0.0165
034 0.994 0.0100
035 0.994 0.0169
039 0.994 0.0158
019 0.995 0.0107
055 0.995 0.0104
059 0.995 0.0128
162 0.995 0.0135
003 0.996 0.0120
029 0.996 0.0079
063 0.996 0.0081
076 0.996 0.0096
095 0.996 0.0081
138 0.996 0.0099
042 0.997 0.0059
015 1.000 0.0000
051 1.000 0.0000
170 1.000 0.0000
```

Interpretation notes:

```text
Rule 60 is additive: next[i] = left XOR center. Reverse has complement ambiguity, matching 0.5.
Rule 90 is additive but width-32 periodic evolution collapses to all zeros by frame 16.
Rule 90 expected rough accuracy = (16*0.5 + 15*1.0) / 31 = 0.742, matching the sweep.
```

### Rule 122 DDP Runs

All DDP runs used 8 B200 GPUs with per-GPU batch size 1024, so global batch size
was 8192.

Regular LR:

```text
job=192298
script=scripts/slurm_rule122_ddp_1k.sbatch
output_root=/home/kboguta_midjourney_com/enum/outputs/rule122_ddp_1k
steps=1000
lr=0.003
elapsed=00:04:01
eval_loss=0.6191
eval_acc=0.615
train_samples_per_sec~=43112
```

High LR:

```text
job=192300
script=scripts/slurm_rule122_ddp_1k_lr2p4e2.sbatch
output_root=/home/kboguta_midjourney_com/enum/outputs/rule122_ddp_1k_lr2p4e2
steps=1000
lr=0.024
elapsed=00:04:01
eval_loss=0.6554
eval_acc=0.589
train_samples_per_sec~=43084
```

Takeaway: DDP gives much higher sample throughput, but a fixed 1k optimizer-step
run with global batch 8192 did not improve Rule 122. We need LR/warmup tuning or
comparisons at fixed number of samples.

### Rule 30 Consistency Evaluation

Baseline model:

```text
checkpoint=rule30_reverse_rowmasked_32x32_bs1024_bf16_lr3e3_l4d64_10k.pt
task=reverse row-masked LM
```

Job:

```text
job=192302
output_root=/home/kboguta_midjourney_com/enum/outputs/rule30_10k_consistency_eval
local_copy=outputs/rule30_10k_consistency_eval
samples=10000
```

Results:

```text
direct_hidden_cell_accuracy=0.9441
next_cell_consistency=0.9535
whole_row_exact_consistency=0.4975
```

Main artifact:

```text
outputs/rule30_10k_consistency_eval/frame_accuracy_vs_transition_consistency.png
```

Takeaway: high per-cell accuracy does not imply exact predecessor-row validity.
The model's independent cell marginals often produce rows that are not exactly
consistent with the known next row.

### Rule 30 Unmasked Autoregressive Runs

Training run:

```text
job=192304
output_root=/home/kboguta_midjourney_com/enum/outputs/rule30_unmasked_ar_1k
task=reverse causal LM without --mask-row-prefix
steps=1000
eval_loss=0.2239
eval_acc=0.885
train_steps_per_sec~=5.62
train_samples_per_sec~=5754
```

The original uncached full autoregressive eval was too slow and was canceled.
KV-style caching was added after that.

Cached full autoregressive eval:

```text
job=192349
output_root=/home/kboguta_midjourney_com/enum/outputs/rule30_unmasked_ar_1k_eval10_cached
local_copy=outputs/rule30_unmasked_ar_1k_eval10_cached
samples=10
elapsed~=13s
overall_cell_accuracy=0.5205
next_cell_consistency=0.5298
whole_row_exact_consistency=0.0000
```

One-row autoregressive eval:

```text
job=192385
output_root=/home/kboguta_midjourney_com/enum/outputs/rule30_unmasked_one_row_ar_1k_eval100
local_copy=outputs/rule30_unmasked_one_row_ar_1k_eval100
samples=100
overall_cell_accuracy=0.7247
next_cell_consistency=0.8359
whole_row_exact_consistency=0.0055
```

Takeaway: same-row prefixes are useful under teacher forcing, but generated
prefixes compound errors. Autoregressive generation did not solve consistency.

### Rule 30 Row-Masked Consistency Loss

Run:

```text
job=192387
output_root=/home/kboguta_midjourney_com/enum/outputs/rule30_rowmasked_consistency_w1_1k
local_copy=outputs/rule30_rowmasked_consistency_w1_1k
steps=1000
consistency_loss_weight=1.0
```

Results:

```text
best_step=800
best_eval_loss=0.4309
best_eval_acc=0.803
final_eval_loss=0.4374
final_eval_acc=0.800
frame_accuracy_overall=0.7993
next_cell_consistency=0.8823
whole_row_exact_consistency=0.0475
```

Takeaway: explicit consistency loss improved exact row validity relative to
one-row AR, but `weight=1.0` degraded per-cell posterior accuracy sharply. Try
smaller weights next.

### Current Working Theory

Reverse CA prediction has at least three distinct targets:

```text
1. Estimate posterior cell marginals.
2. Produce a predecessor row that is dynamically valid.
3. Sample or choose among valid predecessor rows under the posterior.
```

The row-masked transformer is good at item 1. Exact row consistency exposes item
2. Attention maps may show features useful for item 1, but attention is not by
itself a causal claim.

The cleanest next theory experiment is exact enumeration at width 20:

```text
2^20 = 1,048,576 initial conditions
```

This should let us compute exact posterior marginals and valid predecessor
fibers, then compare model logits and attention-derived hypotheses to ground
truth.
