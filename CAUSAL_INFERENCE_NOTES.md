# Causal Inference Notes For Reverse CA Attention

Last updated: 2026-05-19

These notes sharpen the question:

```text
Are the attention maps causally meaningful, or are they just correlated visual
patterns?
```

The short answer is: attention alone is not causal evidence. But cellular
automata give us an unusually clean finite laboratory where we can define
interventions exactly and compare the model to ground truth posteriors.

## Core Setup

We train a transformer on a 32x32 elementary cellular automaton spacetime grid.
For reverse prediction, the model sees later natural rows and predicts earlier
natural rows.

Notation:

```text
X_t[i] = CA cell i at natural evolution frame t
F      = deterministic forward CA transition
X_{t+1} = F(X_t)
Y      = visible future context, usually rows X_{t+1}, ..., X_T
Z      = hidden predecessor row or hidden local variable inside X_t
A_h    = attention map for head h while predicting a target cell
L      = model logits for a target cell or target row
```

For irreversible rules, many predecessor rows may produce the same future row:

```text
fiber(Y) = { X_t : F(X_t) is consistent with the visible future context Y }
```

So reverse prediction is not simply "recover the true past." It is posterior
inference over a set of possible pasts.

## What We Have Actually Measured

The row-masked reverse LM estimates per-cell marginals:

```text
P(X_t[i] | X_{t+1:T})
```

This avoids same-row label leakage, but independent argmax predictions can form
a row that is not a valid predecessor.

The consistency eval asks:

```text
Does F(predicted row) equal the known next row?
```

This produced the key split:

```text
Rule 30 10k row-masked model:
per-cell accuracy              ~= 0.9441
next-cell consistency          ~= 0.9535
whole-row exact consistency    ~= 0.4975
```

Conclusion: per-cell Bayesian-looking accuracy and whole-row causal/dynamical
validity are different objectives.

## Why Attention Is Not Enough

An attention map can highlight cells that are useful for prediction, but that
does not by itself establish causality.

Possible interpretations of a bright attention cell:

```text
1. It is causally upstream of the target hidden cell.
2. It is a proxy for a shared hidden variable.
3. It is correlated because of deterministic CA constraints.
4. It is part of a learned computational routine, not a semantic variable.
5. It is visually striking but behaviorally irrelevant.
```

Pearl-style causal language requires interventions. We need to ask what changes
when we actively alter a variable, not just what co-occurs in observational
samples.

## Variables And Graph Intuition

For target cell `X_t[i]`, a useful local graph is:

```text
X_t neighborhood  -> X_{t+1} local cells -> X_{t+2} wider cells -> ...
       |                                                    |
       +---------------- hidden constraints ----------------+
```

In reverse prediction, the model observes descendants of the hidden target. It
may also observe descendants of nearby hidden variables. These can carry shared
information about the hidden predecessor row.

The important hypothesis is not:

```text
The attended cells directly cause the target cell.
```

The sharper hypothesis is:

```text
The attended cells are measurements of hidden variables that constrain the
posterior over the target cell or target row.
```

This is a hidden-variable detector hypothesis.

## Exact Enumeration Experiment

Use a width small enough to enumerate all initial conditions:

```text
width=20
initial_conditions=2^20=1,048,576
frames=20 or 32
```

For every initial condition, compute the full CA evolution. This gives an exact
finite population, not a sample approximation.

For each target `(t, i)` and visible context definition `Y`, compute:

```text
P_exact(X_t[i] = 1 | Y)
P_exact(X_t = row | Y)
valid_predecessor_count(Y)
```

Then compare:

```text
model probability          vs exact posterior marginal
model chosen row           vs valid predecessor set
attention-sensitive cells  vs exact information-bearing cells
```

## Interventions We Can Define Exactly

### 1. Context Cell Intervention

Choose a visible future cell `Y_j`, flip it, and recompute the exact posterior:

```text
delta_j = P_exact(X_t[i] = 1 | do(Y_j = 1)) -
          P_exact(X_t[i] = 1 | do(Y_j = 0))
```

Caveat: arbitrary `do(Y_j = v)` may create impossible future contexts that no
valid CA trajectory could produce. This is still a useful model-intervention
test, but it is not always an in-distribution causal query.

Use it for:

```text
Does changing the visible cell change the model's target prediction?
Does the model's sensitivity match exact posterior sensitivity where defined?
```

### 2. Antecedent Row Intervention

Choose a candidate hidden row cell `X_t[k]`, flip it, evolve forward, and compare
the resulting future grid to the original future grid.

This is the sensitivity-map experiment we already started:

```text
D_k[u, v] = 1 if future cell (u, v) differs after flipping X_t[k], else 0
```

For a target cell `X_t[i]`, compare attention `A_h` to the intervention effect
field `D_k`.

This tells us:

```text
Which future cells would carry evidence about hidden variable X_t[k]?
Does a head attend to those cells while predicting X_t[i]?
```

### 3. Fiber Intervention

For a fixed observed future context `Y`, enumerate all valid predecessor rows.
Then alter the distribution over predecessor rows directly:

```text
P(X_t | Y) -> P'(X_t | Y)
```

This is not a physical CA intervention; it is an intervention on the posterior
population. It is useful for asking whether the model tracks:

```text
cell marginals
row validity
specific modes in the predecessor fiber
```

## Estimands

Recommended quantities to compute:

```text
posterior_marginal_error =
  mean_abs(model_p(X_t[i]=1 | Y) - P_exact(X_t[i]=1 | Y))

posterior_log_loss =
  cross_entropy(P_exact marginal, model probability)

valid_row_rate =
  fraction of model-produced rows where F(row) == known next row

fiber_mode_accuracy =
  whether model row equals the most likely predecessor row under exact posterior

attention_intervention_alignment =
  correlation or rank overlap between attention A_h and intervention effect D

model_intervention_alignment =
  correlation between exact posterior delta_j and model logit delta_j
```

The most important distinction:

```text
attention_intervention_alignment asks what the head looks at.
model_intervention_alignment asks what actually changes the model output.
```

The second is stronger evidence.

## Minimal Practical Test

Start with one rule, one trained checkpoint, and one target.

Suggested first target:

```text
rule=30
width=20
target=(t=10, i=10)
visible_context=rows t+1..T
```

Procedure:

```text
1. Enumerate all 2^20 initial conditions.
2. Group trajectories by visible future context Y.
3. For each group, compute exact P(X_t[i]=1 | Y).
4. Run the model on those same Y contexts.
5. Compare model probability to exact posterior.
6. For visible cells in Y, intervene/flip and measure model logit changes.
7. Compare model logit changes to exact posterior changes where contexts are valid.
8. Compare attention maps to both exact deltas and model logit deltas.
```

Pass criteria for "hidden-variable detector" language:

```text
1. The head attends to cells with high intervention effect on the hidden variable.
2. Ablating or masking those attended cells changes the model prediction.
3. The direction of logit change agrees with exact posterior change.
4. The pattern replicates across many contexts, not just one pretty example.
```

If only criterion 1 holds, the attention pattern is interesting but not yet
causal evidence.

## Attention Ablations

A direct model-side causal test:

```text
Run prediction normally.
Run prediction while masking or zeroing a specific head's attention to selected cells.
Measure target logit delta.
```

Variants:

```text
mask top-k attention cells
mask top-k exact-intervention cells
mask random cells with matched distance/frame distribution
mask entire head
```

Controls are essential. A bright attention pattern is only meaningful if it beats
matched random or geometry-matched baselines.

## Decision Table

```text
Observation                                      Interpretation
------------------------------------------------ -----------------------------
High attention, low output delta                 visual/computational artifact
Low attention, high output delta                 information is carried elsewhere
High attention, high output delta                candidate causal model feature
High alignment with exact posterior deltas       strong hidden-variable evidence
High per-cell acc, low valid-row rate            marginal model, invalid joint row
High valid-row rate, worse marginal calibration  constrained decoder behavior
```

## Main Warnings

Do not claim:

```text
The attention map is the causal graph.
```

Do claim, if supported:

```text
This head attends to cells whose interventions strongly change the exact
posterior over the hidden target, and ablating those cells changes the model's
logit in the same direction.
```

That is the bridge from attention visualization to causal evidence.

## Next Implementation Step

Build an enumeration module for width 20:

```text
enumerate all initial rows
evolve each under a selected ECA rule
pack visible contexts into integer keys
aggregate exact posterior counts
compute valid predecessor fibers
export CSV/NPZ summaries for model comparison
```

The first deliverable should be a single plot comparing:

```text
x-axis: exact P(X_t[i]=1 | Y)
y-axis: model P(X_t[i]=1 | Y)
```

Then add intervention/alignment plots after the exact-posterior baseline is
working.
