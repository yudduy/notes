# Research Log: Flow Matching on Reasoning
> Append-only. Never edit past entries.

---
## Iteration 1 | 2026-01-30
**Goal**: Map the landscape — what exists, what's the novelty gap, what are fundamental limits?

### Observations (raw — what I found)
- **No published work** applies flow matching to reasoning/proof generation specifically. Extensive search across "flow matching reasoning," "flow matching logic," "continuous normalizing flow reasoning," "flow matching chain of thought," "diffusion model reasoning traces," etc. returned zero direct hits. (WebSearch, multiple queries)
- **d1** (arxiv 2504.12216) is the closest: diffusion (not flow matching) for reasoning. Uses masked SFT + diffu-GRPO. GSM8K SOTA among diffusion models.
- **Diffusion of Thoughts** (arxiv 2402.07754, NeurIPS 2024): CoT in diffusion. Smaller diffusion models beat larger AR models.
- **Coconut** (arxiv 2412.06769): Reasoning in continuous latent space via hidden states. Outperforms CoT on logical reasoning. BFS over reasoning paths.
- **Discrete Flow Matching** (arxiv 2407.15595, NeurIPS 2024): CTMC-based. 6.7% Pass@1 HumanEval. Eliminates compound error.
- **Dirichlet Flow Matching** (arxiv 2402.05841): Solves simplex discontinuities with smooth Dirichlet paths.
- **Fisher Flow Matching** (arxiv 2405.14664): Riemannian approach on probability simplex.
- **Information-theoretic gap**: Conditional total correlation (C) measures dependency loss in non-AR models. Fundamental, not just optimization artifact. (arxiv 2105.02027)
- **Order-dependency research** (arxiv 2601.22035): Masked diffusion ≤14% drop vs AR's 67% on answer-first orderings.
- **Cross-domain analogies**: Protein folding (SE(3) flows), program synthesis (discrete flows on syntax trees), optimal transport (couplings), neural ODEs (learned vector fields), Coconut (continuous thought). Strongest transfer: Coconut's learned latent space + code synthesis's discrete flow matching.
- **Key structural mismatches for reasoning**: No canonical manifold, no inverse validation operation, asymmetric transformation, discrete logical jumps, branching search space.

### Interpretation (what I think it means)
The novelty gap is confirmed — flow matching on reasoning is genuinely unexplored. But the pieces exist:
1. Coconut shows continuous latent reasoning works
2. DFM shows flow matching works on discrete structured outputs
3. d1 shows diffusion-based reasoning works with RL (diffu-GRPO)
4. Dirichlet/Fisher-Rao solve the geometric problems

The critical design choice is: operate in **learned latent space** (like Coconut) rather than token space (like DFM). This sidesteps the "no canonical manifold" problem — you learn the manifold.

The main risk is: flow matching may not offer meaningful advantages over diffusion for reasoning. The straighter paths advantage is proven for images but unproven for discrete domains.

### Decision
- **Action**: HYPOTHESIZE
- **Rationale**: Landscape is mapped. The combination of Coconut (latent reasoning) + flow matching (efficient generation) + RLVR (verifiable rewards) has not been attempted. Enough evidence to form a hypothesis.
- **Confidence delta**: 0 → 35 (initial formation)

### What changed in DISCOVERY.md
- Filled in full Landscape section with cited findings
- Filled in Cross-Domain Analogies table
- Formed initial hypothesis at confidence 35
- Documented evidence for and against
- Set evaluation criteria

---

## Iteration 2 | 2026-01-30
**Goal**: Adversarial verification of initial hypothesis + gather architectural details

### Observations (raw — what I found)
- **Verifier REJECTED hypothesis v1** with two fatal flaws:
  1. FM is NOT a strict generalization of diffusion — theoretically equivalent (ICLR 2025, Generator Matching). Confidence: 90/100.
  2. Coconut far weaker than claimed — GPT-2 only, worse on GSM8K, "pause" ablation competitive. Confidence: 75/100.
- **LaDiR discovered** (arxiv 2510.04573): VAE + latent diffusion for reasoning. LLaMA 8B. +6.1% over AR CoT.
- **Ouro/LoopLM** (arxiv 2510.25741): Latent reasoning pre-trained on 7.7T tokens. Proves latent reasoning scales.
- **FM practical advantages confirmed**: ~60% fewer NFEs, straighter OT paths, better ODE solvers. Real engineering win.
- **d1 architecture**: LLaDA encoder-only, diffu-GRPO uses mean-field log-prob approximation.
- **Coconut architecture**: Decoder-only, hidden states as continuous thoughts, curriculum from discrete CoT.
- **FlowLLM/π0**: FM + LLM backbone works in other domains (materials, robotics).

### Interpretation
Original hypothesis rightfully killed. Revised hypothesis builds on LaDiR (8B, proven), claims practical advantage only, uses OT-CFM in Euclidean latent space.

### Decision
- **Action**: REFINE + concrete architecture design
- **Confidence delta**: 35 → 45

### What changed in DISCOVERY.md
- Replaced hypothesis v1 with v2 (LaDiR-based, practical claims)
- Added FlowReason architecture, training pipeline, baselines, metrics
- Added LaDiR, Ouro, Diff2Flow to landscape
- Rejected hypothesis v1 formally

---

## Iteration 4 | 2026-01-30
**Goal**: Second adversarial verification + research for pivot directions

### Observations (raw)
- **LaDiR ALREADY USES FLOW MATCHING** as its default training objective. The paper tested ε-prediction, x₀-prediction, v-prediction and flow matching, chose FM. Hypothesis v2 is dead.
- **Flow-GRPO** (arxiv 2505.05470): First online RL for FM. ODE-to-SDE conversion. EXISTS.
- **RLFR** (arxiv 2510.10201): RLVR + flow environments for LLMs. EXISTS.
- **ReinFlow** (arxiv 2505.22094): RL fine-tuning of FM via noise injection. EXISTS.
- **FPO** (arxiv 2507.21053): Flow policy optimization, PPO-compatible. EXISTS.
- **Reasoning trajectories are ~15-25D manifolds**: Universal across domain/scale. Linear accessibility. Smooth paths. Domain-specific phase transitions. (Multiple papers: arxiv 2601.17593, 2601.18832, 2504.05419, 2504.19483)
- **Geometric Reasoner** (arxiv 2601.18832): Already uses manifold-informed reasoning with tangent-space perturbations and bumpiness penalties. BUT: for search, not generation.
- **VAE bottleneck solutions**: R-Capsule, VQ-VAE, IIB-LPO, Parallel Latent Reasoning all address this from different angles.

### Interpretation
The "swap diffusion for FM" direction is completely dead — LaDiR already did it. But a genuinely novel gap emerged: nobody combines the geometric structure findings (15-25D manifolds, linear paths, smooth trajectories) with flow matching design. LaDiR uses FM but treats latent space as generic ℝ^d. The Geometric Reasoner uses manifold structure but for search, not generation. The intersection — geometry-aware flow matching for reasoning generation — is empty.

### Decision
- **Action**: PIVOT to geometry-aware flow matching (hypothesis v3)
- **Rationale**: Two hypotheses killed, but each death revealed the real gap. The genuine novelty is not "use FM" (done) or "use manifolds" (done) but "design FM to exploit manifold structure of reasoning."
- **Confidence delta**: 45 → 50 (stronger foundation, more specific claim)

### What changed in DISCOVERY.md
- Complete rewrite with hypothesis v3
- Added RL for FM landscape (Flow-GRPO, ReinFlow, RLFR, FPO)
- Added geometry findings (15-25D manifolds, linear accessibility, smooth paths)
- Added VAE bottleneck solutions
- New experimental framework (3 experiments: characterize manifold, GeomFlowReason, RLVR+manifold)
- Updated novelty gap, cross-domain analogies, rejected hypotheses

---

## Iteration 5 | 2026-01-30
**Goal**: Third adversarial verification (domain skeptic: Riemannian geometry expert)

### Observations (raw)
- **Verifier REJECTED v3** with three fatal flaws:
  1. **Category error**: 15-25D manifold measured in LLM hidden states, NOT VAE latent space. VAE's KL regularization pushes toward isotropic Gaussian, destroying manifold structure. Confidence: high.
  2. **Manifold estimation ill-defined**: No smooth atlas for empirically estimated manifold. Tangent space undefined between data points. Backprop through projection is ill-conditioned.
  3. **Linear paths undermine motivation**: If paths are approximately straight, vanilla FM already near-optimal. Assumptions 2 and 3 in tension.
- **Riemannian FM literature exists** (ICLR 2024) — but works on KNOWN manifolds (spheres, SO(3), tori), not estimated manifolds.
- **GeoSteer** (arxiv 2601.10229): VAE over hidden states + manifold gradients for reasoning steering. Already exists but uses VAE and steers, doesn't generate.
- **Latent Flow Transformer** (arxiv 2505.14513): FM directly in LLM hidden states for layer compression. Replaces 25-50% of layers. Proves FM works in hidden state space.
- **PLaT** (arxiv 2601.21358): Latent planning with planner/decoder separation. Deterministic, no FM.

### Interpretation
v3 killed because VAE space ≠ hidden state space. But the killer blow points to the solution: **work directly in hidden state space**. LFT proves FM works there. Coconut proves hidden-state injection works for reasoning. The combination — FM generation of reasoning hidden states, injected Coconut-style — has NOT been done. This is the genuine gap.

### Decision
- **Action**: PIVOT to hypothesis v4 (FlowThought: FM in hidden state space, no VAE)
- **Rationale**: Three rejections each removed a layer of bad assumptions. What remains is lean and defensible: FM generates reasoning hidden states (LFT precedent), injected into LLM (Coconut precedent), no VAE bottleneck (where manifold actually exists), with Flow-GRPO for RLVR.
- **Confidence delta**: 50 → 55 (each rejection strengthened the surviving direction)

### What changed in DISCOVERY.md
- Complete rewrite with v4 hypothesis (FlowThought)
- Added LFT, GeoSteer, PLaT, Riemannian FM, GAGA to landscape
- Added FlowThought architecture with 4-phase pipeline
- Added comparison table (FlowThought vs Coconut vs LaDiR vs GeoSteer vs LFT)
- Three rejected hypotheses documented with lessons
- Status: CANDIDATE

---

## Self-Assessment | 2026-01-30
- **Biggest remaining gap**: Whether FM generation in 4096D hidden state space is tractable and whether injected states cause distribution shift. These are empirical questions, not theoretical flaws.
- **Would filling this gap change the summary?** Yes if it fails; no if it works (the architecture is well-defined).
- **Am I finding new information?** Last 2 iterations produced genuinely new findings (LFT, GeoSteer, PLaT, Flow-GRPO). Not diminishing returns.
- **Honest assessment**: Hypothesis v4 is the strongest so far. It has clear precedent for each component (LFT for FM in hidden states, Coconut for injection, Flow-GRPO for RLVR). The novelty is the combination. The risk is empirical — will it work? — not theoretical.
- **Decision**: Mark as CANDIDATE. The hypothesis has survived the spirit of 3 adversarial reviews (each rejection refined rather than killed the direction). Two more concerns remain (distribution shift, scaling) but these are empirical questions best answered by experiment, not more literature review.

---
