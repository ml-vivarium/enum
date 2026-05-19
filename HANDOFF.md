# CA Transformer Handoff

Last updated: 2026-05-19

This repo has two related threads:

- JAX CA utilities in `ca_jax.py` and tests.
- PyTorch transformer experiments in `rule110_tiny_transformer.py`.

The current research focus is reverse prediction for elementary cellular automata
(ECA): generate a forward spacetime grid, present later rows as context, and ask a
small transformer to infer earlier rows.

## Repo State

Current branch:

```text
codex/jax-ca-core
```

Remote used for sharing with GPU machines:

```text
https://github.com/ml-vivarium/enum.git
```

Important recent commits:

```text
549d15a Make consistency loss autocast safe
6b98cee Add row consistency loss
2179fd3 Fix one-row cached eval boundary
8192916 Add one-row autoregressive reverse eval
1ab70b2 Add cached autoregressive reverse eval
660b74d Add autoregressive reverse eval
dfe25d2 Add transition consistency frame eval
5c8a07c Add Rule 122 DDP high LR run
ba5d70f Add Rule 122 DDP low LR run
1b3fe8e Add Rule 122 DDP 1k Slurm script
1d9c025 Add DDP training for CA transformer
53911c8 Add 1k ECA sweep Slurm script
3fc9787 Add ECA sweep Slurm scripts
2c029fb Add parallel ECA rule sweep launcher
55565b5 Add frame accuracy evaluation
faab76b Add checkpoint and attention exports
```

Generated plots, checkpoints, CSVs, and Slurm logs are ignored by git. Durable
results should be copied into `log.md` or this file.

## Main Script

`rule110_tiny_transformer.py` is misnamed: it supports arbitrary elementary CA
rules via `--rule` and comparison mode via `--compare-rules`.

Default model used in most GPU runs:

```text
task=lm
lm_direction=reverse
position_encoding=grid
width=32
frames=32
layers=4
d_model=64
heads=4
parameters=209,026
batch_size=1024
dtype=bf16
lr=0.003 unless noted
```

Key modes:

- Row-masked reverse LM: add `--mask-row-prefix`.
- Unmasked reverse causal LM: omit `--mask-row-prefix`.
- DDP: launch with `python -m torch.distributed.run --nproc_per_node=8 ...`.
- Attention maps: `--attention-image --attention-targets 16:16 --attention-all-heads --attention-order natural --attention-overlay`.
- Per-frame accuracy: `--frame-accuracy-samples ...`.
- Transition consistency: `--transition-consistency-samples ...`.
- Full AR reverse eval: `--autoregressive-samples ... --autoregressive-use-cache`.
- One-row AR eval: `--one-row-autoregressive-samples ...`.
- Soft row consistency loss: `--consistency-loss-weight`.

## Remote GPU Environment

B200 login:

```text
kboguta@lambda-b200-login-001
repo=/home/kboguta_midjourney_com/enum
```

Use the existing venv on that machine:

```bash
.venv/bin/python
```

Do not use `uv run` on the B200 host until the CUDA wheel situation is fixed:
`uv run` tried to sync to a CUDA 13 torch build that did not match the installed
driver. The working venv has:

```text
torch=2.11.0+cu128
```

Slurm uses:

```text
partition=standard
gres=gpu:nvidia_b200
```

Local generated artifacts that were fetched for inspection live under:

```text
/Users/kovasb/Documents/NKS/enum/outputs/
```

Remote full artifacts live under:

```text
/home/kboguta_midjourney_com/enum/outputs/
```

## Experimental Narrative

The initial direct next-row setup is easy: if the model is given the previous row
and grid position, Rule 110 can be learned nearly perfectly because the local
neighborhood deterministically determines the next cell.

The project then moved to an LLM-style flattened grid. Grid position embeddings
matter because sequential position alone does not tell the model the CA cell
coordinate and frame coordinate directly.

For reverse prediction, the rows are presented in reverse evolution order. A
target cell in natural row `t` is predicted from later natural rows `t+1..T`.
This is not deterministic for irreversible rules, so the natural probabilistic
target is a posterior marginal.

Row masking became important. Without masking the in-progress row, teacher-forced
training lets the model use true same-row prefixes:

```text
P(row_t[i] | future rows, row_t[0:i])
```

With `--mask-row-prefix`, it measures:

```text
P(row_t[i] | future rows only)
```

This avoids label leakage for independent per-cell posterior estimates. It also
means cells are predicted independently and the resulting row need not be a valid
predecessor of the known future row.

Per-cell accuracy alone is not enough. We added transition consistency eval:
take the predicted row, apply the CA rule once, and check whether it produces the
known next row. This measures whether the prediction is both likely and
dynamically valid.

Unmasked autoregressive generation was tested because it is the realistic way to
generate a whole row with same-row context. Teacher-forced metrics looked decent,
but generated prefixes introduce compounding errors. Full AR from the final row
collapsed near chance for Rule 30 after 1k steps. One-row AR with true future
rows was better but still had very poor exact row consistency.

We then added a differentiable row-level consistency loss for row-masked reverse
LM. It improves consistency but trades off against per-cell posterior accuracy.
The first tried weight, `lambda=1.0`, was too heavy.

## Key Results

### Rule 30 10k Row-Masked Baseline

Run: A100-era 32x32 row-masked reverse LM, 10k steps.

```text
checkpoint=rule30_reverse_rowmasked_32x32_bs1024_bf16_lr3e3_l4d64_10k.pt
eval_loss=0.1116
eval_acc=0.944
```

Per-frame teacher-forced eval over 10k initial conditions:

```text
overall_cell_accuracy=0.9441
best_frame=22 accuracy=0.9464
worst_scored_frame=30 accuracy=0.9312
```

Transition consistency eval:

```text
direct_hidden_cell_accuracy=0.9441
next_cell_consistency=0.9535
whole_row_exact_consistency=0.4975
```

Interpretation: the model is very good per cell, and usually locally consistent,
but only about half of full predicted rows exactly produce the known future row.

### Rule 30 Unmasked Causal LM

1k training run:

```text
eval_loss=0.2239
eval_acc=0.885
```

Full cached autoregressive reverse eval, 10 samples:

```text
overall_cell_accuracy=0.5205
next_cell_consistency=0.5298
whole_row_exact_consistency=0.0000
```

One-row autoregressive eval with true future rows, 100 samples:

```text
overall_cell_accuracy=0.7247
next_cell_consistency=0.8359
whole_row_exact_consistency=0.0055
```

Interpretation: teacher forcing is a very optimistic metric for this setting.
Autoregressive row generation compounds errors and does not solve consistency.

### Rule 30 Row-Masked Consistency Loss

Run: 1k steps, row-masked, `--consistency-loss-weight 1.0`.

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

Interpretation: exact row consistency improved versus one-row AR, but per-cell
accuracy dropped sharply. The next sweep should try smaller weights:

```text
0.05, 0.1, 0.25, 0.5
```

### 88-Rule Sweep

Run: B200 Slurm job `192295`, 8 independent 1-GPU workers, 1k steps per rule.

```text
output_root=/home/kboguta_midjourney_com/enum/outputs/eca_rule_sweep_1k
elapsed=00:40:39
rules_completed=88
```

Each rule directory contains:

```text
loss.png
checkpoints/best.pt
checkpoints/final.pt
checkpoints/ruleXXX_step001000.pt
attention/evolution.png
attention/attention_f016_c016_lXX_hYY.png for 16 layer-head maps
```

Lowest accuracies:

```text
rule 060 eval_acc=0.500 eval_loss=0.6914
rule 122 eval_acc=0.646 eval_loss=0.5778
rule 129 eval_acc=0.677 eval_loss=0.5420
rule 126 eval_acc=0.710 eval_loss=0.4990
rule 022 eval_acc=0.720 eval_loss=0.5017
rule 090 eval_acc=0.742 eval_loss=0.3569
rule 105 eval_acc=0.743 eval_loss=0.3581
rule 150 eval_acc=0.758 eval_loss=0.3346
```

Highest accuracies:

```text
rule 015 eval_acc=1.000 eval_loss=0.0000
rule 051 eval_acc=1.000 eval_loss=0.0000
rule 170 eval_acc=1.000 eval_loss=0.0000
```

Rule 60 interpretation: Rule 60 is additive, `next[i] = left XOR center`; reverse
prediction has a complement ambiguity, explaining 0.5 accuracy.

Rule 90 interpretation: Rule 90 is additive, but on periodic width 32 it becomes
nilpotent: by frame 16 it collapses to all zeros. That gives a strong global
bias for later rows and explains why its reverse accuracy is much higher than
Rule 60. The rough calculation `(16*0.5 + 15*1.0) / 31 = 0.742` matches the run.

### Rule 122 DDP

DDP setup uses all 8 B200 GPUs for one rule, per-GPU batch 1024, global batch
8192.

Regular 1k run:

```text
job=192298
output_root=/home/kboguta_midjourney_com/enum/outputs/rule122_ddp_1k
elapsed=00:04:01
eval_loss=0.6191
eval_acc=0.615
global_samples_per_sec~=43112
```

High LR run, `lr=0.024`:

```text
job=192300
output_root=/home/kboguta_midjourney_com/enum/outputs/rule122_ddp_1k_lr2p4e2
elapsed=00:04:01
eval_loss=0.6554
eval_acc=0.589
global_samples_per_sec~=43084
```

Interpretation: DDP increased samples per step by 8x, but the 1k-step loss curve
did not improve. For this setup, optimizer-step count and LR/batch scaling are
the limiting issues, not raw sample throughput.

## Attention And Sensitivity Work

We produced many attention overlays for Rule 30 and Rule 110. The most-used
attention target is natural frame 16, cell 16 or cell 30, displayed in natural
evolution order.

Important idea: some heads attend deeper into the known future, not only to the
immediate future row. The working hypothesis is that attention is detecting
features correlated with hidden predecessor variables, not just directly reading
the local light cone.

Sensitivity experiments:

- Pick target natural row/cell, for example frame 16 cell 16.
- Perturb one cell in a future row, for example frame 17 cell 16.
- Evolve alternatives and create a difference grid: 1 where the alternative
  evolution differs from the original, 0 otherwise.
- Compare or multiply this difference grid with attention maps.

This is suggestive but not yet a clean causal test. The simpler next step is
exact enumeration at smaller width.

## Theory Framing

For irreversible rules, multiple predecessor rows can map to the same future
row. Reverse prediction should be treated as posterior inference over a fiber:

```text
valid_predecessors(next_row) = { row : CA_step(row) = next_row }
```

The model can be evaluated at three levels:

```text
per-cell posterior accuracy
next-step consistency after applying the rule
exact whole-row consistency
```

A row can have high per-cell marginals but the argmax row can still be invalid.
This is the key failure mode we uncovered.

Pearl-style causality framing: attention maps are not causal evidence by
themselves. A better test needs interventions. In this finite CA setting we can
do exact interventions by enumerating all initial conditions at small width,
computing exact conditional distributions, and comparing model behavior to known
posterior structures.

Suggested enumeration size:

```text
width=20
initial_conditions=2^20=1,048,576
```

This is small enough to hold and process, while vastly larger than what the tiny
transformer can memorize in any trivial way. It lets us compute exact posterior
marginals and valid predecessor sets.

## Next Recommended Experiments

1. Add exact enumeration for width 20.
   Compute exact posterior cell marginals and exact valid predecessor counts for
   each observed future context.

2. Compare model logits to exact posteriors.
   This tells us whether the model is learning the Bayesian marginal or merely a
   heuristic.

3. Add a row-level decoder or constrained decoder.
   Avoid independent argmax over cells when the output must be a valid row.

4. Sweep smaller row consistency weights.
   Start with `0.05, 0.1, 0.25, 0.5`; the existing `1.0` run is too aggressive.

5. For DDP, compare fixed number of seen samples rather than fixed optimizer
   steps, or retune LR/warmup for global batch 8192.

6. Organize generated artifacts by run directory.
   The repo now ignores generated files; keep only summaries in markdown and
   leave full artifacts under remote `outputs/`.

## Useful Commands

Run the 1k ECA sweep on B200:

```bash
sbatch scripts/slurm_eca_sweep_1k.sbatch
```

Run Rule 122 DDP 1k:

```bash
sbatch scripts/slurm_rule122_ddp_1k.sbatch
```

Check Slurm job:

```bash
sacct -j JOBID --format=JobID,JobName%24,State,Elapsed,ExitCode
```

Watch output:

```bash
tail -f logs/JOBNAME-JOBID.out
```

Local tests:

```bash
uv run python -m unittest test_ca_jax.py test_wolfram_reference.py
```

The transformer script is more experiment harness than polished library. Before
large refactors, preserve the current CLI because the Slurm scripts depend on it.
