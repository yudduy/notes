# Research Log: Hopf Algebra Language Models
> Append-only. Never edit past entries.

---
## Iteration 0 | 2026-01-30
**Goal**: Initialize discovery. Set up documents.

### Observations (raw)
- User has had extensive conversation with Gemini covering FEP, dissipative structures, KPZ universality, integrable probability, and Marcolli-Chomsky Hopf algebra formalization of Merge
- Prior research (my earlier analysis) found: Hopf algebra NNs are nonexistent, recursive Transformers exist (ReCAT, Relaxed Recursive), Lambeq/QNLP is real but doesn't scale, RG-NN mapping is theoretical
- The core question: can Marcolli's math become architecture?

### Interpretation
The gap between algebraic linguistics and neural architecture is the central problem. Need to map the landscape deeply before hypothesizing.

### Decision
- **Action**: RESEARCH MORE — map the landscape
- **Rationale**: Cannot hypothesize without knowing what's been tried and what's impossible
- **Confidence delta**: N/A (starting)

### What changed in DISCOVERY.md
- Initialized all sections with problem statement and evaluation criteria

---

## Iteration 1 | 2026-01-30
**Goal**: Map the full landscape via three parallel research agents. Form initial hypothesis.

### Observations (raw)
- **Agent 1 (Marcolli/Hopf/Tree)**: Marcolli-Chomsky 2023 paper is real, has 3 follow-ups through 2025 (MIT Press book, wavelet function spaces paper, old/new minimalism comparison). ZERO neural implementations exist. DTM (ICML 2023) achieves 100% on SCAN/COGS via differentiable tree ops. NSR (ICLR 2024) also 100%. Treeformer adds CKY to Transformers. ReCAT (ICLR 2024) does differentiable CKY with linear space. Permutation equivariance helps compositionality (ICLR 2020). — Sources: arXiv:2305.18278, arXiv:2306.00751, arXiv:2210.01603, arXiv:2207.06960
- **Agent 2 (DisCoCat/QNLP)**: Lambeq doesn't scale (<100 sentences, 74-87% accuracy). DisCoCirc improves composability but still tiny. CRITICAL FINDING: arXiv:2501.02931 (Jan 2025) formalizes self-attention as parametric endofunctor in Para(Vect). DisCoCLIP (2025) combines CCG tensor networks with CLIP. — Sources: arXiv:2504.09909, arXiv:2501.02931
- **Agent 3 (Limits/Failures)**: Tree-RNNs died due to GPU mismatch, not theoretical impossibility. No fundamental impossibility. Exponential gradient growth with tree depth. Tensor networks wrong entanglement for text. Compositional gen. REQUIRES structural alignment. Grammar bias provides "surprisingly little benefit" in most tasks. — Sources: arXiv:2207.02098, arXiv:2505.02627, PhysRevA.111.032409

### Interpretation
1. Marcolli math is proven but completely unimplemented
2. DTM/NSR ALREADY solve compositional gen. on synthetic benchmarks with 100% accuracy
3. Real question: does HOPF STRUCTURE add value beyond just using trees?
4. Main threat: DTM gets 100% without Hopf. What would Hopf add?
5. Main opportunity: DTM doesn't scale to real-world data. Perhaps Hopf constraints help scaling.

### Decision
- **Action**: HYPOTHESIZE — formed Hypothesis 1 (Coproduct Regularization)
- **Rationale**: Landscape mapped. Hypothesis targets specific gap.
- **Confidence delta**: 0 → 35

### What changed in DISCOVERY.md
- Filled Landscape, formed Hypothesis 1, added analogies, updated eval criteria

---

## Iteration 2 | 2026-01-30
**Goal**: Stress-test Hypothesis 1 via adversarial verifier. Deep-dive DTM architecture.

### Observations (raw)
- **Verifier 1 (Adversarial)**: VERDICT: REJECT. Fatal flaws: (1) DTM already 100% on proposed benchmarks — no headroom, (2) coassociativity may be trivially satisfied or vacuous for neural encoders, (3) Marcolli's proof is descriptive not prescriptive — map≠territory. Weaknesses: O(2^n) coproduct, no differentiable Loday-Ronco implementation exists, improvement may come from trees not Hopf. Critical counterexample: linguistically invalid decompositions ("cat sat") forced to have similar representations would HURT performance. Recommended: test axioms incrementally, start simple. — Sources: arXiv:2306.00751, arXiv:2210.01603, arXiv:2311.00268, arXiv:2601.18858
- **DTM Deep-Dive**: DTM uses car/cdr/cons on Tensor Product Representations. Blending (weighted soft mixtures) is essential — Gumbel-Softmax causes "complete breakdown." DTM enforces NO algebraic properties: no reconstruction identity, no associativity, no consistency checks. sDTM (NeurIPS 2024) uses sparse coordinate trees with bit-shifting, 70x fewer params. CRITICAL: arXiv:2407.02060 shows DTM has "fake OOD" — fails on novel tree reversal tasks. DTM's 100% stems from invariance to specific test cases, not true compositional understanding. — Sources: arXiv:2306.00751, arXiv:2412.14076, arXiv:2407.02060

### Interpretation
1. Hypothesis 1 is dead — can't improve on 100% on SCAN/COGS, and full coproduct is intractable
2. BUT: DTM's "fake OOD" is a real vulnerability. Algebraic constraints COULD fix this.
3. The verifier's counterexample (invalid decompositions) is strong but applies to FULL coproduct — sampled valid decompositions could avoid this
4. The verifier's recommended approach (incremental axiom testing) is better than our original
5. Key insight: DTM's blending creates invalid trees. Algebraic constraints = "type checker" ensuring tree validity
6. Homomorphism regularization (arXiv:2601.18858) already shows simpler algebraic constraints help — we should climb the ladder

### Decision
- **Action**: PIVOT — reject H1, form H2 (Algebraic Ladder)
- **Rationale**: Verifier identified fatal flaws in H1. H2 addresses all of them: targets failure cases, incremental, tractable.
- **Confidence delta**: 35 → 45 (better hypothesis, but untested)

### What changed in DISCOVERY.md
- Rejected H1, formed H2 (Algebraic Ladder)
- Added DTM's fake OOD to Known Limits
- Added homomorphism regularization to Current SOTA
- Revised Open Gaps to reflect incremental testing opportunity
- Updated evaluation criteria to target DTM failure cases

---

## Iteration 3 | 2026-01-30
**Goal**: Second-round adversarial verification (NLP skeptic). Research associativity in language.

### Observations (raw)
- **NLP Skeptic Verifier**: Key attacks: (1) Associativity is meaningless for natural language syntax — grammar combination provably non-associative [Springer s10849-009-9081-1]. (2) Reconstruction identity is circular — just autoencoding, not Hopf. (3) RASP Transformers get near-perfect COGS structural gen. [arXiv:2504.15349]. Compositional Program Generation gets perfect with 14-22 examples [arXiv:2309.16467]. (4) Stacked constraints give diminishing returns [NeurIPS 2018]. (5) KILLER TEST: replace algebraic constraints with random auxiliary losses of equal compute cost — if random ≥ algebraic, the algebra is irrelevant.
- **Associativity Research**: CRITICAL FINDING — Language parse trees are NON-ASSOCIATIVE. BUT: Loday-Ronco has DENDRIFORM structure where individual grafting ops ≺, ≻ are non-associative while their sum * = ≺ + ≻ IS associative. The linguistically meaningful property is COASSOCIATIVITY of the coproduct (decomposition), not associativity of the product. Merge operates on a free non-associative commutative magma. — Sources: nLab dendriform algebra, Cornell Aguiar/Sottile, arXiv:2305.18278

### Interpretation
1. H2's Level 2 (associativity loss) is WRONG — language is non-associative. Must replace with dendriform axiom.
2. The skeptic's random baseline demand is devastating but fair — MUST include this control.
3. Scaling Transformers (RASP, 32-layer) are a real competitor — algebra may be unnecessary.
4. Confidence should DROP because the competitive landscape is stronger than I initially thought.
5. The dendriform structure is the CORRECT algebraic formulation — this is actually a refinement, not a weakness.
6. Even at low confidence, this is worth pursuing as first-ever neural Hopf implementation — publishable as negative result.

### Decision
- **Action**: REFINE — reject H2's associativity, form H3 (Dendriform Regularization with Random Baseline Control)
- **Rationale**: Correct the algebra (dendriform, not associative), add random baseline control, lower confidence to honest 30.
- **Confidence delta**: 45 → 30 (more honest; stronger competitive landscape)

### What changed in DISCOVERY.md
- Rejected H2, formed H3
- Corrected algebra: dendriform ≺, ≻ instead of naive associativity
- Added RASP and CPG to competitive landscape
- Added random baseline control as required ablation
- Lowered confidence to 30 with explicit reasons
- Updated Open Gaps to reflect dendriform gap

---

## Iteration 4 | 2026-01-30
**Goal**: Determine implementability — are dendriform axioms trivial in vector spaces? Does code exist?

### Observations (raw)
- **SageMath has dendriform algebra**: `sage.combinat.free_dendriform_algebra` — production-ready implementation of free dendriform algebras over planar binary trees. Operations: `prec` (≺), `succ` (≻), associative product *. BUT: Trac #25452 indicates Loday-Ronco Hopf algebra methods incomplete. — Source: [SageMath docs](https://doc.sagemath.org/html/en/reference/algebras/sage/combinat/free_dendriform_algebra.html)
- **Dendriform.jl exists**: Julia package for dendriform tree operations — Source: [GitHub chakravala/Dendriform.jl](https://github.com/chakravala/Dendriform.jl)
- **CRITICAL: Dendriform axioms are NON-TRIVIAL**: Classification shows exactly 10 non-isomorphic dendriform structures on 2D complex space. Not every associative algebra admits dendriform decomposition — requires specific construction (Rota-Baxter operator). — Source: ResearchGate classification paper
- **Single dendriform operation is O(n)** for trees of size n. Full coproduct remains O(2^n).
- **ZERO neural/ML implementations exist**. Search for "dendriform neural" returns only "dendritic neuron" results.
- **Rota-Baxter connection**: Given associative algebra A with Rota-Baxter operator R, define x ≺ y = xR(y), x ≻ y = R(x)y. This gives dendriform structure. Key: R must satisfy R(x)R(y) = R(R(x)y + xR(y) + λxy). — Source: [Rota-Baxter algebra Wikipedia](https://en.wikipedia.org/wiki/Rota%E2%80%93Baxter_algebra)

### Interpretation
1. Dendriform axioms are genuinely non-trivial — good, they'd impose real constraints
2. SageMath code exists for validation — can compare neural outputs against exact symbolic computation
3. O(n) per operation is tractable for sentences (<100 tokens)
4. The Rota-Baxter construction suggests an ALTERNATIVE approach: instead of enforcing axioms as losses, parameterize the operations via a learnable Rota-Baxter operator R. Then axioms are AUTOMATICALLY satisfied by construction.
5. This is genuinely unexplored territory — first neural implementation would be novel regardless of outcome

### Decision
- **Action**: REFINE H3 — add Rota-Baxter parameterization as alternative to loss-based enforcement
- **Rationale**: Loss-based enforcement may drift during training. Architectural enforcement (via Rota-Baxter) guarantees axiom satisfaction.
- **Confidence delta**: 30 → 40 (implementability confirmed, two implementation paths available)

### What changed in DISCOVERY.md
- Added SageMath/Julia implementations to landscape
- Added Rota-Baxter alternative to H3
- Confirmed non-triviality of dendriform axioms
- Raised confidence 30→40 based on implementability

---

## Iteration 5 | 2026-01-30
**Goal**: Final novelty check. Assess completion.

### Observations (raw)
- WebSearch "Rota-Baxter operator neural network machine learning 2024 2025" → ZERO results. No prior work.
- WebSearch "Rota-Baxter deep learning neural differentiable arxiv" → ZERO results. Confirmed novel.
- All search results return neural operators for PDEs (DeepONet), differentiable architecture search (DARTS), etc. — completely unrelated.

### Interpretation
1. Rota-Baxter operators in neural networks is genuinely unexplored territory
2. The hypothesis (H3) has survived: 2 adversarial verifier rounds, 1 major algebra correction (associativity → dendriform), 1 implementability confirmation, 1 novelty check
3. Confidence sits at 40 — moderate. The hypothesis is well-formed, novel, and implementable, but faces strong competition from scaling Transformers and has never been tested
4. The experiment design (L_dendri vs L_random vs baseline on DTM failure cases) is clean regardless of outcome
5. This is ready for marking as CANDIDATE

### Decision
- **Action**: Mark H3 as CANDIDATE. Discovery complete.
- **Rationale**: Hypothesis survived required scrutiny (2 adversarial rounds, different framings), novelty confirmed, implementability confirmed. The remaining open question (does L_dendri > L_random?) can only be answered by running the experiment.
- **Confidence delta**: 40 (unchanged — no new evidence for/against)

### What changed in DISCOVERY.md
- Final status update to CANDIDATE
- Summary of what was explored and what was learned

---
