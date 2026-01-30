# Discovery: Flow Matching on Reasoning
> Status: CANDIDATE | Iteration: 6 | Last updated: 2026-01-30

## Problem
How to rigorously apply flow matching to reasoning in a way that is genuinely novel and experimentally tractable? Three prior hypotheses were killed by adversarial review. The surviving direction operates directly in LLM hidden state space (where geometric structure is empirically confirmed) without a VAE bottleneck.

## Landscape (What Exists)

### Latent Reasoning Systems
- **LaDiR** (2025): VAE + flow matching in latent space. +6.1% over AR CoT. LLaMA 8B. Already uses FM. [arxiv 2510.04573](https://arxiv.org/abs/2510.04573)
- **Coconut** (COLM 2025): Sequential hidden state passing for reasoning. BFS exploration. No generative model. GPT-2 scale only. [arxiv 2412.06769](https://arxiv.org/abs/2412.06769)
- **PLaT** (2026): Latent planning with planner/decoder separation. Deterministic trajectory, no probabilistic generation. GRPO for decoder only. [arxiv 2601.21358](https://arxiv.org/abs/2601.21358)
- **Ouro/LoopLM** (2025): Recurrent looped transformers. 7.7T tokens. Latent reasoning at scale. [arxiv 2510.25741](https://arxiv.org/abs/2510.25741)
- **d1** (2025): Masked diffusion + diffu-GRPO on tokens. 8B. [arxiv 2504.12216](https://arxiv.org/abs/2504.12216)

### Geometry & Steering
- **GeoSteer** (2026): VAE over hidden states → manifold gradients → steering. +0.9 accuracy, +4.5 reasoning quality on GSM8K. [arxiv 2601.10229](https://arxiv.org/abs/2601.10229)
- **Geometric Reasoner** (2026): Smooth manifold paths, bumpiness penalties, tangent-space perturbations. For search, not generation. [arxiv 2601.18832](https://arxiv.org/abs/2601.18832)
- **Representation Engineering** (2025): Reasoning is directional. Cross-task transfer. [arxiv 2504.19483](https://arxiv.org/abs/2504.19483)
- **15-25D intrinsic manifolds**: Universal across domain/scale in LLM hidden states. [arxiv 2601.17593](https://arxiv.org/abs/2601.17593)
- **Activation State Machines**: Control-theoretic steering of reasoning trajectories. [OpenReview](https://openreview.net/forum?id=p17En1bhCY)

### Flow Matching in Hidden State Space
- **Latent Flow Transformer (LFT)** (2025): Replaces blocks of transformer layers with FM velocity field mapping input hidden states to output hidden states. 25-50% layer compression. Proves FM works in hidden state space. Uses "Recoupling Ratio" from OT to identify compressible blocks. [arxiv 2505.14513](https://arxiv.org/abs/2505.14513)

### RL for Flow Matching
- **Flow-GRPO** (2025): Online RL for FM. ODE-to-SDE conversion. [arxiv 2505.05470](https://arxiv.org/abs/2505.05470)
- **RLFR** (2025): RLVR + flow velocity deviations as rewards. [arxiv 2510.10201](https://arxiv.org/abs/2510.10201)
- **ReinFlow** (2025): Noise injection for exact likelihood RL. [arxiv 2505.22094](https://arxiv.org/abs/2505.22094)
- **FPO** (2025): Advantage-weighted FM policy gradients. PPO-compatible. [arxiv 2507.21053](https://arxiv.org/abs/2507.21053)

### Riemannian Flow Matching
- **Riemannian FM** (ICLR 2024): Simulation-free FM on known Riemannian manifolds. [ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/d1f9936d3be6997ffffab692977eebe6-Paper-Conference.pdf)
- **Generalised Flow Maps** (2025): Few-step inference on arbitrary manifolds. [arxiv 2510.21608](https://arxiv.org/abs/2510.21608)
- **GAGA** (2024): Geometry-aware generative autoencoder. Learns warped Riemannian metric. 30% improvement on trajectory inference. [arxiv 2410.12779](https://arxiv.org/abs/2410.12779)

### Novelty Gap (Confirmed After 3 Adversarial Reviews)
**What has NOT been done:**
1. **Flow matching as a generative model for reasoning trajectories directly in LLM hidden state space** — without VAE bottleneck. LFT does FM in hidden states for layer compression but not for reasoning generation. Coconut does hidden-state reasoning but without a generative model. PLaT does latent planning but not with FM. GeoSteer uses a VAE. Nobody has combined Coconut-style hidden-state reasoning with FM-style probabilistic generation.
2. **Using FM to generate diverse reasoning trajectories in hidden state space** — enabling probabilistic exploration (like MCTS/BFS) over continuous reasoning paths, with RLVR for quality optimization.

## Current Hypothesis (v4 — After Three Adversarial Reviews)

- **Statement**: A flow matching model trained directly in an LLM's hidden state space can generate diverse, high-quality reasoning trajectories by learning the velocity field that maps noise to the distribution of successful reasoning hidden states — bypassing the VAE bottleneck entirely. The architecture combines Coconut-style hidden state injection with FM-style probabilistic generation: instead of deterministically passing hidden states (Coconut) or compressing through a VAE (LaDiR/GeoSteer), we learn a conditional velocity field v_θ(h_t, t | problem) that generates reasoning hidden states h₁ from noise h₀, conditioned on the problem encoding. The generated h₁ is injected back into the LLM as a continuous thought (Coconut-style) to produce the answer.
- **Confidence**: 55
- **Status**: Candidate

### Why This Hypothesis Survives
Each prior rejection eliminated a bad assumption while preserving the core:
- **v1**: "FM > diffusion" → killed. Lesson: don't claim FM superiority over diffusion.
- **v2**: "Replace LaDiR's diffusion with FM" → killed. Lesson: LaDiR already uses FM.
- **v3**: "Manifold-constrained FM in VAE latent space" → killed. Lesson: VAE destroys manifold structure; work in hidden state space directly.
- **v4**: FM in hidden state space, no VAE, Coconut-style injection. This addresses all prior flaws:
  - Doesn't claim FM > diffusion (uses FM because it's the right tool for hidden-state velocity fields)
  - Doesn't duplicate LaDiR (no VAE, different space)
  - Works where the manifold actually exists (hidden states, not VAE latent space)
  - Has precedent (LFT proves FM works in hidden states)

### Evidence Supporting
- LFT proves FM velocity fields can map between LLM hidden states. [arxiv 2505.14513](https://arxiv.org/abs/2505.14513)
- Coconut proves hidden state injection works for reasoning. [arxiv 2412.06769](https://arxiv.org/abs/2412.06769)
- 15-25D intrinsic manifold exists in LLM hidden states (not VAE space). [arxiv 2601.17593](https://arxiv.org/abs/2601.17593)
- Reasoning is directional/linear in hidden state space — FM's OT paths align. [arxiv 2504.19483](https://arxiv.org/abs/2504.19483)
- GeoSteer shows steering in hidden-state manifold improves reasoning. [arxiv 2601.10229](https://arxiv.org/abs/2601.10229)
- Flow-GRPO provides the RL objective for FM fine-tuning. [arxiv 2505.05470](https://arxiv.org/abs/2505.05470)
- No VAE bottleneck — operates at full hidden-state dimensionality.

### Evidence Against
- Coconut failed to scale beyond GPT-2 / 8 latent tokens. Hidden-state approaches may have fundamental scaling limits.
- LFT compresses layers, not generates reasoning — different use case. May not transfer.
- Hidden state space is ~4096D, much higher than VAE latent space. FM may need more capacity/data.
- Training requires paired (problem, reasoning hidden states) data — must extract from existing reasoning model.
- Injecting FM-generated hidden states into an LLM may cause distribution shift / instability.

### Survived Attacks
- "FM isn't better than diffusion": Not claiming FM > diffusion. Using FM because velocity fields are the right formalism for hidden state dynamics (as LFT shows).
- "LaDiR already uses FM": This doesn't use LaDiR's approach at all — no VAE, different space, different architecture.
- "Manifold structure in VAE space": Not using VAE space. Working directly where the manifold was measured.
- "GeoSteer already does this": GeoSteer steers existing reasoning; this generates new reasoning trajectories. GeoSteer uses VAE; this doesn't.

### Open Questions
1. Does FM generation in 4096D hidden state space need dimensionality reduction? (PCA projection first?)
2. How to handle the distribution shift when injecting FM-generated hidden states?
3. Can this scale beyond Coconut's limits?
4. What training data format? Collect hidden states from DeepSeek-R1 or similar reasoning model?
5. How many NFEs are needed for good hidden-state generation?

## Proposed Experimental Framework (v4)

### Architecture: FlowThought

```
Phase 1: Data Collection
- Run a reasoning model (e.g., Qwen-3 8B with CoT) on GSM8K/MATH
- Extract hidden states at each reasoning step → H = {h_1, h_2, ..., h_T}
- Label: h_T = final reasoning state (before answer generation)
- Condition: c = h_0 (hidden state of problem encoding)

Phase 2: Flow Matching Training
- Source: h₀ ~ N(0, I) projected to hidden state manifold
  (or: h₀ ~ estimated marginal of non-reasoning hidden states)
- Target: h₁ = final reasoning hidden state (from Phase 1 data)
- Condition: c = problem encoding hidden state
- OT-CFM: v_θ(h_t, t, c) with MSE loss
- Architecture for v_θ: Lightweight transformer (or MLP) operating on hidden dim

Phase 3: Inference
- Given new problem → encode to c via frozen LLM
- Sample h₀, integrate v_θ from t=0 to t=1 → ĥ₁
- Inject ĥ₁ into frozen LLM as continuous thought (Coconut-style)
- LLM generates answer conditioned on ĥ₁

Phase 4 (Optional): RLVR
- Use Flow-GRPO to fine-tune v_θ
- Reward: correctness of final answer (binary 0/1)
- Backprop through: FM generation → hidden state injection → LLM answer → reward
```

### Key Differences from Existing Work
| | FlowThought (Proposed) | Coconut | LaDiR | GeoSteer | LFT |
|---|---|---|---|---|---|
| Space | Hidden states | Hidden states | VAE latent | Hidden states (via VAE) | Hidden states |
| Generative model | FM (velocity field) | None (deterministic pass) | FM | None (gradient steering) | FM |
| Purpose | Generate reasoning trajectories | Sequential reasoning | Latent generation | Steer existing reasoning | Layer compression |
| VAE | No | No | Yes | Yes | No |
| RLVR compatible | Yes (Flow-GRPO) | No | Partially | No | No |

### Baselines
1. **Coconut**: Sequential hidden state passing (no generation)
2. **LaDiR**: VAE + FM in latent space
3. **PLaT**: Deterministic latent planning
4. **AR CoT SFT**: Standard chain-of-thought
5. **GeoSteer**: Manifold gradient steering

### Metrics
- Pass@1, Pass@K on GSM8K, MATH500, Countdown
- Diversity of generated reasoning trajectories (pairwise cosine distance)
- Hidden state distribution matching (FID-like metric in hidden space)
- Inference NFEs vs quality
- Ablation: FM-generated hidden states vs random hidden states vs Coconut states

### Compute
- Phase 1: 1-2 GPUs, 1 day (data collection)
- Phase 2: 4-8 A100s, 2-3 days (FM training on v_θ)
- Phase 3: 1 GPU (inference)
- Phase 4: 4-8 A100s, 2-3 days (RLVR)
- Total: ~$2-5K

## Cross-Domain Analogies (Final)
| Source Domain | Insight | Transfers? | Limitations |
|---|---|---|---|
| **LFT (layer compression)** | FM velocity fields learn hidden state transformations | **Direct precedent** — same space, same formalism | Different purpose (compression vs generation) |
| **Coconut** | Hidden state injection works for reasoning | **Architecture pattern** — inject generated states same way | Coconut is deterministic; we're probabilistic |
| **π0 (robot control)** | FM action head on frozen VLM backbone | **Architecture pattern** — FM head on frozen LLM | Continuous actions vs hidden states |
| **GeoSteer** | Reasoning manifold exists in hidden states and can be exploited | **Confirms structure** — we generate on it instead of steering | GeoSteer uses VAE; we don't |
| **Protein folding (FoldFlow)** | FM learns structured transformations in high-D space | **Inspiration** — FM for structured generation | Known manifold (SE(3)) vs learned manifold |

## Rejected Hypotheses
| # | Hypothesis | Why Rejected | Key Lesson |
|---|---|---|---|
| v1 | FM in Coconut latent space with Dirichlet geometry | Coconut toy-scale; FM not strictly better; Dirichlet unvalidated | Use proven foundations; claim practical not fundamental |
| v2 | Replace LaDiR's diffusion backbone with FM | LaDiR ALREADY uses FM as default | Read papers thoroughly before building on them |
| v3 | Manifold-constrained FM in VAE latent space | VAE KL regularization destroys manifold structure; manifold estimation ill-defined; approximately linear paths means vanilla FM already near-optimal | Work in hidden state space where manifold exists; skip VAE |

## Evaluation Criteria
1. **Falsifiable prediction**: FlowThought-generated hidden states produce higher-quality reasoning than Coconut's sequential hidden states (measured by Pass@1)
2. **Diversity**: FM generates diverse reasoning trajectories (unlike Coconut's deterministic single path)
3. **Quality**: Match or exceed LaDiR (+6.1% over AR CoT)
4. **Efficiency**: Competitive inference cost with Coconut (few NFEs)
5. **RLVR enhancement**: Flow-GRPO fine-tuning improves quality beyond supervised FM
6. **Scaling**: Must work at ≥1B parameter scale (not just GPT-2)
