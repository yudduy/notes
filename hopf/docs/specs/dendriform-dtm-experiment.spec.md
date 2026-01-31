# Specification: Coproduct Consistency Regularization for Compositional Generalization in sDTM

> Use `/duy-workflow:execute docs/specs/dendriform-dtm-experiment.spec.md` to implement.
>
> **Revision history**: v1 targeted dendriform axiom losses. Adversarial review (Prof. Voronova) identified three fatal flaws: (1) dendriform ≺/≻ ≠ car/cdr/cons — different algebras, (2) Marcolli uses coproduct/coideals, not dendriform ops, (3) tree reversal task not in sDTM repo. v2 targets the **coproduct** — the actual algebraic structure Marcolli formalizes — which IS expressible via car/cdr/cons.

## Goal
Test whether enforcing **coassociativity of tree decomposition** — the central algebraic property of the Hopf algebra structure Marcolli-Chomsky proved language syntax satisfies — improves compositional generalization in sDTM, with a random baseline control to isolate algebraic signal from regularization effect.

## Context
- **Paper**: Workshop paper targeting NeurIPS/ICML workshops (CompGen, MathAI, NeuroAI)
- **Core claim**: Marcolli-Chomsky-Berwick (2023, arXiv:2305.18278; MIT Press 2025) proved linguistic Merge generates a Loday-Ronco Hopf algebra. The **coproduct** Δ of this algebra encodes how syntactic structures decompose into sub-constituents. Coassociativity (Δ⊗id)∘Δ = (id⊗Δ)∘Δ means decomposition order doesn't matter — you get the same sub-constituents regardless of which piece you break apart first. We test whether enforcing this as a regularization loss improves sDTM's compositional generalization on structural OOD splits.
- **Why coproduct, not dendriform**: Marcolli's formalization uses the coproduct and coideals. The dendriform structure (≺, ≻) is a mathematical property of the Loday-Ronco algebra but is NOT what Marcolli invokes, and cannot be expressed via car/cdr/cons. The coproduct CAN be expressed via car/cdr (tree decomposition = iterated subtree extraction).
- **Outcome either way**: If L_coprod > L_random > baseline → algebraic structure matters (positive). If L_random >= L_coprod → algebra irrelevant (publishable negative — first empirical test of Hopf algebra constraints in NNs).

## Requirements

### REQ-1: Fork and reproduce sDTM baseline
- Clone `psoulos/sdtm` (NeurIPS 2024)
- Reproduce **Active↔Logical** task results as the primary benchmark (this IS in the repo, has structural OOD splits, and sDTM reports results on it)
- Identify which structural OOD splits show imperfect performance — these are the targets for improvement
- Run a pilot (1 seed, 1 task) to measure **wall-clock training time** — needed before committing to 60 runs
- **Acceptance**: Reproduced sDTM's reported accuracy on Active↔Logical (within 2% of paper). Training time per run measured.

### REQ-2: Implement coproduct consistency loss (L_coprod)
The Hopf algebra coproduct on binary trees decomposes a tree into sub-constituents. Coassociativity means the order of decomposition doesn't matter. We enforce this on sDTM's intermediate tree states.

**Decomposition via car/cdr**: For a tree T in sDTM's memory:
- **Single cut**: `(car(T), cdr(T))` — decompose into left and right subtrees
- **Double cut, left-first**: `(car(car(T)), cdr(car(T)), cdr(T))` — decompose left subtree further
- **Double cut, right-first**: `(car(T), car(cdr(T)), cdr(cdr(T)))` — decompose right subtree further

**Coassociativity loss**: The model's encoder f should produce consistent representations regardless of decomposition order. For any tree T with depth >= 2:

```
# Decompose left subtree first: get 3 pieces
left_first = f(car(car(T))) + f(cdr(car(T))) + f(cdr(T))

# Decompose right subtree first: get 3 pieces
right_first = f(car(T)) + f(car(cdr(T))) + f(cdr(cdr(T)))

# The SET of atomic pieces should produce equivalent aggregate representations
# (using sum as a permutation-invariant aggregation)
L_coassoc = ||left_first - right_first||^2
```

Note: This is a simplification. Full coassociativity operates over ALL admissible cuts, but we approximate by sampling decomposition paths. For trees of depth d, we sample K=3 random decomposition orders per tree.

**Reconstruction identity** (auxiliary):
```
L_recon = ||cons(car(T), cdr(T)) - T||^2
```
This tests whether sDTM's operations are self-consistent. May already be ~0 in a well-trained model.

**Combined loss**:
```
L_coprod = L_coassoc + alpha * L_recon
```

- Sample trees T from the model's intermediate memory states at each computation step
- Only compute L_coassoc for trees with depth >= 2 (need at least two levels to decompose both ways)
- **Acceptance**: L_coprod computes without error, L_coassoc is non-zero at initialization, produces non-zero gradients, and changes during training

### REQ-3: Implement random consistency loss (L_random — control)
- For each tree T, compute two random projections of the flattened SCT state using fixed random matrices W1, W2 (initialized once, frozen)
- L_random = ||W1 @ flatten(T) - W2 @ flatten(T)||^2
- This has matched compute structure (two forward passes through fixed transforms, one MSE) but zero algebraic content
- **Acceptance**: L_random has approximately matched compute overhead (+/- 30% wall-clock) and produces non-zero gradients

### REQ-4: Implement tree reconstruction loss (L_recon_only — ablation)
- L_recon_only = ||cons(car(T), cdr(T)) - T||^2
- This tests whether the simpler "reconstruction" regularization (without coassociativity) suffices
- Cheaper than L_coprod — serves as intermediate ablation point
- **Acceptance**: Computes without error

### REQ-5: Training configuration
- **Four** experimental conditions:
  1. **Baseline**: sDTM with no auxiliary loss (original training)
  2. **L_coprod**: sDTM + λ * L_coprod (coassociativity — the Hopf algebra property)
  3. **L_recon_only**: sDTM + λ * L_recon_only (reconstruction only — simpler constraint)
  4. **L_random**: sDTM + λ * L_random (random — controls for regularization effect)
- Lambda sweep: {0.01, 0.1, 1.0} — 3 values (reduced from 4 to keep total runs manageable)
- 5 random seeds per condition per lambda = 4 conditions × 3 lambdas × 5 seeds = **60 total runs**
- Use sDTM's existing entropy regularization warmup pattern (linear warmup over 10k steps) for the auxiliary losses
- **Primary benchmark**: Active↔Logical (in sDTM repo, has structural OOD splits)
- **Acceptance**: All 60 runs complete without OOM on single consumer GPU (<=24GB VRAM). Adjust batch size if needed.

### REQ-6: Evaluation metrics
- **Primary metric**: Structural OOD accuracy on Active↔Logical
- **Secondary metrics**:
  - In-distribution accuracy
  - Lexical OOD accuracy
  - L_coassoc value at convergence (how close to coassociative is the trained model?)
  - L_recon value at convergence (how self-consistent are the tree operations?)
- Report: mean ± std across 5 seeds for best lambda per condition
- Statistical test: paired t-test (or Wilcoxon signed-rank if non-normal) between conditions at best lambda
- **Acceptance**: Results table with all metrics, p-values, and effect sizes (Cohen's d)

### REQ-7: Experiment runner and logging
- Single entry point: `python run_experiment.py --condition {baseline,coprod,recon,random} --seed {0-4} --lambda {value} --task active_logical`
- Wandb logging of: primary loss, auxiliary loss, L_coassoc, L_recon, OOD accuracy per epoch
- Save best checkpoint per run (by validation accuracy)
- Results aggregation script: `python aggregate_results.py` → summary table (LaTeX-formatted)
- **Acceptance**: Aggregation script produces clean table with means, stds, p-values

### REQ-8: Pilot run (MUST complete before full sweep)
- Run 1 seed of each condition at lambda=0.1 on Active↔Logical
- Measure: wall-clock time per run, VRAM usage, convergence behavior
- Verify: L_coprod is non-trivial (not ~0 from start), gradients flow, no NaN
- Estimate total experiment time from pilot
- **Acceptance**: Pilot completes, measurements recorded, go/no-go decision documented

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Algebraic structure | **Coproduct coassociativity** (NOT dendriform) | Coproduct is what Marcolli actually uses; expressible via car/cdr/cons; dendriform is incoherent for this architecture (see revision history) |
| Base model | sDTM (psoulos/sdtm) | 70x fewer params, 1.8GB VRAM, has auxiliary loss patterns |
| Implementation path | Loss-based (soft enforcement) | Simpler, directly comparable to control, sufficient for workshop paper |
| Primary benchmark | **Active↔Logical** (not tree reversal) | Actually in the sDTM repo with structural OOD splits. Tree reversal is from a different paper with no public code. |
| Control condition | Random consistency loss | Matched compute structure, zero algebraic content |
| Additional ablation | Reconstruction-only loss | Separates "coassociativity helps" from "any tree regularization helps" |
| Seeds | 5 per condition | Standard for comp. gen. papers |
| Lambda grid | {0.01, 0.1, 1.0} | 3 values × 4 conditions × 5 seeds = 60 runs (feasible) |
| Venue target | Workshop (NeurIPS/ICML) | 4-6 pages, negative results accepted |

## Completion Criteria
- [ ] REQ-1: sDTM baseline reproduced on Active↔Logical
- [ ] REQ-2: L_coprod implemented with non-zero, non-trivial gradients
- [ ] REQ-3: L_random implemented with matched compute
- [ ] REQ-4: L_recon_only implemented
- [ ] REQ-5: All 60 runs complete
- [ ] REQ-6: Results table with statistical tests
- [ ] REQ-7: Experiment runner + aggregation scripts
- [ ] REQ-8: Pilot run completed with go/no-go decision
- [ ] Build + lint clean

## Edge Cases

| Case | Expected Behavior |
|------|-------------------|
| L_coassoc ≈ 0 from start | sDTM already satisfies coassociativity. Interesting finding — report it. Means the tree operations are already algebraically consistent. |
| L_coassoc diverges / NaN | Clip auxiliary loss gradients. If persists, reduce lambda. Log NaN occurrences. |
| Tree depth < 2 in memory | Skip L_coassoc for this tree (can't do double decomposition). Only apply to trees with sufficient depth. |
| All conditions equal | Clean negative result. Report: algebraic structure adds nothing beyond baseline. |
| L_coprod = L_random > baseline | Regularization helps, but algebra is irrelevant. Important finding. |
| L_coprod > L_random > L_recon > baseline | Full algebraic structure (coassociativity) matters beyond simple reconstruction. Strongest positive result. |
| L_recon > L_coprod | Simple reconstruction beats coassociativity. Simpler structure suffices. |
| OOM | Reduce batch size. sDTM 1.8GB baseline + ~1GB auxiliary → should fit 8GB. |
| Pilot shows >6hr per run | Scale down: 3 seeds instead of 5, or reduce lambda grid to {0.1, 1.0}. |

## Technical Context

### Key Files (to create/modify)

**Fork from**: `https://github.com/psoulos/sdtm`

| File | Purpose |
|------|---------|
| `sdtm/losses.py` (NEW) | L_coprod (L_coassoc + L_recon), L_random, L_recon_only implementations |
| `sdtm/trainer.py` (MODIFY) | Inject auxiliary losses into training loop (follow existing entropy pattern) |
| `sdtm/config.py` (MODIFY) | Add --lambda_aux, --condition, --seed CLI args |
| `run_experiment.py` (NEW) | Single entry point for all experiment runs |
| `run_pilot.sh` (NEW) | Shell script to run pilot (4 conditions × 1 seed × 1 lambda) |
| `run_sweep.sh` (NEW) | Shell script to run full sweep (60 runs, sequential) |
| `aggregate_results.py` (NEW) | Read wandb/logs, produce LaTeX results table + p-values |

### Coproduct Loss Implementation Detail

```python
def coproduct_consistency_loss(trees, encoder_fn, car_fn, cdr_fn):
    """
    Coassociativity: decomposing left-first vs right-first should give
    equivalent aggregate representations.

    For tree T with depth >= 2:
      Left-first:  f(car(car(T))) + f(cdr(car(T))) + f(cdr(T))
      Right-first: f(car(T)) + f(car(cdr(T))) + f(cdr(cdr(T)))

    Loss = ||left_first - right_first||^2
    """
    loss = 0.0
    count = 0
    for T in trees:
        if tree_depth(T) < 2:
            continue
        # Left-first decomposition
        left_sub = car_fn(T)
        right_sub = cdr_fn(T)
        lf = encoder_fn(car_fn(left_sub)) + encoder_fn(cdr_fn(left_sub)) + encoder_fn(right_sub)
        # Right-first decomposition
        rf = encoder_fn(left_sub) + encoder_fn(car_fn(right_sub)) + encoder_fn(cdr_fn(right_sub))
        loss += (lf - rf).pow(2).sum()
        count += 1
    return loss / max(count, 1)
```

### Patterns to Follow
- sDTM's existing entropy regularization in `trainer.py`:
  ```python
  batch_loss += current_entropy_coef * (
      batch_entropies['cons_arg1'].mean() +
      batch_entropies['cons_arg2'].mean()
  )
  ```
- Linear warmup over 10k steps for auxiliary loss coefficient
- Wandb logging (sDTM already uses wandb)
- SCT tensor format: sparse coordinate trees with Gorn addresses

### Dependencies
- PyTorch >= 1.13
- torch_geometric (sDTM dependency)
- wandb (logging)
- scipy (statistical tests in aggregate_results.py)
- numpy, matplotlib (standard)

### Hardware
- Single consumer GPU (8-24GB VRAM)
- sDTM baseline: 1.8GB VRAM
- With auxiliary losses: estimated 2.5-3.5GB VRAM
- Training time: unknown until pilot (REQ-8)
- Total experiment: 60 runs (sequential unless multiple GPUs available)

## Paper Outline (for reference)

```
Title: Does Hopf Algebra Structure Help Compositional Generalization?
       Testing Coproduct Consistency on Differentiable Tree Machines

1. Introduction
   - Marcolli-Chomsky-Berwick (2023/2025): language Merge = Hopf algebra
   - The coproduct encodes constituent decomposability
   - Coassociativity = decomposition order invariance
   - Question: does enforcing this as a neural loss improve comp. gen.?

2. Background
   - Loday-Ronco Hopf algebra and the coproduct on binary trees
   - Sparse Differentiable Tree Machines (sDTM)
   - Compositional generalization and structural OOD

3. Method
   - L_coprod: coproduct coassociativity loss on sDTM tree operations
   - L_recon: reconstruction identity (ablation)
   - L_random: random consistency control (matched compute)
   - Experimental design: 4 conditions × 3 lambdas × 5 seeds

4. Results
   - Table: structural OOD accuracy across conditions
   - Statistical comparison with effect sizes
   - Lambda sensitivity
   - Analysis: how coassociative is the baseline model?

5. Discussion
   - If positive: first evidence Hopf algebra structure helps NNs
   - If negative: Marcolli-Chomsky formalization doesn't provide
     actionable inductive bias for current architectures
   - Relationship to necessary conditions for comp. gen. (arXiv:2505.02627)

6. Related Work
   - Marcolli-Chomsky-Berwick (2023, 2025 MIT Press)
   - DTM/sDTM (Soulos et al., 2023, 2024)
   - Algebraic constraints for comp. gen. (arXiv:2601.18858)
   - Coverage principle and algebraic invariance (arXiv:2505.20278)

7. Conclusion
```

## Verification Notes (from adversarial review)

**Reviewed by**: Prof. Voronova framework (adversarial + domain skeptic)
**Issues fixed in v2**:
1. Replaced dendriform ≺/≻ (wrong algebra) with coproduct coassociativity (correct — what Marcolli actually uses)
2. Replaced tree reversal (not in sDTM repo) with Active↔Logical (in repo, has OOD splits)
3. Added L_recon_only as intermediate ablation (separates "any tree regularization" from "coassociativity specifically")
4. Added mandatory pilot run (REQ-8) before committing to 60-run sweep
5. Loss formulas verified: car/cdr/cons CAN express tree decomposition (unlike dendriform grafting)

**Remaining risks**:
- L_coassoc may be trivially ~0 if sDTM already satisfies coassociativity
- Structural OOD splits in Active↔Logical may already be near-perfect for sDTM (need to check)
- Training time unknown until pilot
- Sum-based aggregation in L_coassoc loses ordering information — may need alternative aggregation
