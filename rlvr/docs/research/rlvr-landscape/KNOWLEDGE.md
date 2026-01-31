# Research: RLVR Landscape — From Verifiable Rewards to Discovery & AGI
> Started: 2026-01-30

## Summary
RLVR (Reinforcement Learning with Verifiable Rewards) has emerged as the dominant post-training paradigm for reasoning LLMs, replacing RLHF for tasks with objective correctness signals. The landscape spans three interacting paradigms: (1) **weight-updating RLVR** (DeepSeek-R1, OpenAI o1/o3) using GRPO/PPO with verifiable rewards, (2) **frozen-LLM search** (AlphaEvolve, FunSearch, ShinkaEvolve) using evolutionary/MCTS algorithms with LLMs as static generators, and (3) **in-context RL** where LLMs adapt at test-time through context injection without weight updates. The user's two key papers — DeepSearch and TTT-Discover — represent frontier work bridging RLVR with MCTS (DeepSearch) and test-time RL for scientific discovery (TTT-Discover). Flow matching on reasoning is a nascent but theoretically promising direction, with discrete flow matching already working for code generation. The strategic question is whether frozen-LLM + search (rapid iteration, no training cost) or RLVR weight updates (deeper internalization) is more effective for discovery — evidence suggests they are complementary, with frozen search for exploration and RLVR for consolidation.

## Core Concepts
- **RLVR**: RL training using deterministic verifiable rewards (math correctness, code execution, logic) instead of learned human preference models. Binary 1/0 signals. Resistant to reward hacking. (DeepSeek-R1, 2025; [arxiv 2501.12948](https://arxiv.org/abs/2501.12948))
- **GRPO (Group Relative Policy Optimization)**: RL algorithm eliminating critic model by comparing groups of responses relative to each other. ~50% compute reduction vs PPO. Foundation of DeepSeek-R1. ([arxiv 2402.03300](https://arxiv.org/abs/2402.03300))
- **RLHF**: RL from Human Feedback — predecessor paradigm using learned reward models. Vulnerable to reward hacking at scale, subjective, expensive. Being supplanted by RLVR for reasoning tasks.
- **Test-Time Scaling (TTS)**: Expanding compute at inference rather than training. Includes longer chain-of-thought, MCTS search, in-context RL. ([testtimescaling.github.io](https://testtimescaling.github.io/))
- **In-Context RL (ICRL)**: LLMs learning from reward feedback within context window, no weight updates. Multi-round prompting where responses + scalar rewards accumulate in context. ([arxiv 2506.06303](https://arxiv.org/abs/2506.06303))
- **Algorithm Distillation (AD)**: Training transformers to predict actions from RL learning histories, distilling the learning algorithm itself into a network. (Laskin et al., ICLR; [arxiv 2210.14215](https://arxiv.org/abs/2210.14215))
- **Flow Matching**: Simulation-free training of continuous normalizing flows by regressing vector fields u_θ(t,x) between source and target distributions. (Lipman et al.; [arxiv 2210.02747](https://arxiv.org/abs/2210.02747))
- **Discrete Flow Matching**: Extension of flow matching to discrete domains (language, code). Non-autoregressive generation. 6.7% Pass@1 on HumanEval at 1.7B params. (NeurIPS 2024; [arxiv 2407.15595](https://arxiv.org/abs/2407.15595))
- **Differentiable Reasoning**: Continuous relaxations of symbolic logic enabling gradient-based optimization of logical programs. Includes SATNet, DLM. ([arxiv 1905.12149](https://arxiv.org/abs/1905.12149), [arxiv 2102.11529](https://arxiv.org/abs/2102.11529))
- **Verifiable Process Reward Models (VPRM)**: Assessing correctness of intermediate reasoning steps, not just final answers. ([arxiv 2601.17223](https://arxiv.org/abs/2601.17223))

## Key Literature

### User's Two Papers
- **DeepSearch: Overcome the Bottleneck of RLVR via MCTS** (Wu et al., 2025): Integrates MCTS into RLVR training loop (not just inference). Global frontier selection, entropy-based guidance for confident mistakes, Tree-GRPO objective. 1.5B model achieves 62.95% avg on math benchmarks with 5.7x fewer GPU hours than baselines. Key insight: exploration bottleneck in RLVR solved by algorithmic innovation, not more compute. [arxiv 2509.25454](https://arxiv.org/abs/2509.25454)
- **Learning to Discover at Test Time (TTT-Discover)** (Yuksekgonul et al., 2026): Test-time RL for scientific discovery. Entropic objective prioritizes finding single exceptional solutions over average performance. PUCT state selection (AlphaZero-inspired). Results: improved Erdős minimum overlap bound, 50% faster GPU kernels, 1st place on AtCoder competitions, improved single-cell denoising. ~$500/problem with open-source 120B model. [arxiv 2601.16175](https://arxiv.org/abs/2601.16175)

### RLVR Foundations
- **DeepSeek-R1** (2025): First major RLVR success. GRPO + binary verifiable rewards. AIME 2024: 15.6%→77.9%. [arxiv 2501.12948](https://arxiv.org/abs/2501.12948)
- **OpenAI o1/o3**: "Thinking before answering" — extended CoT + RL with verifiable rewards. o3 adds tool use via RL, multimodal reasoning. [openai.com](https://openai.com/index/learning-to-reason-with-llms/)
- **RLVR Incentivizes Correct Reasoning** (2025): Theoretical framework showing RLVR incentivizes correct reasoning even with answer-only rewards. [arxiv 2506.14245](https://arxiv.org/abs/2506.14245)
- **RLVR with Noisy Verifiers** (2025): Handles imperfect verification. [arxiv 2510.00915](https://arxiv.org/abs/2510.00915)
- **RLVRR: From Verifiable Dot to Reward Chain** (ICLR 2026): Extends RLVR to open-ended generation via reference-based rewards. [arxiv 2601.18533](https://arxiv.org/abs/2601.18533)
- **TACLer: Tailored Curriculum RL** (2026): Curriculum RL reducing compute >50%, inference tokens >42%, improving accuracy >9%. [arxiv 2601.21711](https://arxiv.org/abs/2601.21711)
- **GRPO Dynamics & Success Amplification** (2025): Analysis of GRPO algorithm effectiveness. [arxiv 2503.06639](https://arxiv.org/abs/2503.06639)

### Frozen LLM + Search
- **AlphaEvolve** (DeepMind, 2025): Evolutionary coding agent with Gemini. Dual LLM strategy (Flash+Pro). Recovered 0.7% of Google's compute, 23% speedup in matrix multiplication, first improvement over Strassen in 56 years. [arxiv 2506.13131](https://arxiv.org/abs/2506.13131)
- **FunSearch** (DeepMind, Nature): Evolutionary search for functions. LLM generates candidate code, evaluator filters, best solutions seed next generation. Discovered largest cap sets in 20 years. [Nature](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/funsearch-making-new-discoveries-in-mathematical-sciences-using-large-language-models/Mathematical-discoveries-from-program-search-with-large-language-models.pdf)
- **ShinkaEvolve** (Sakana AI, 2025): Orders of magnitude more sample-efficient than AlphaEvolve/FunSearch. SOTA circle packing in 150 samples. Parent sampling + novelty rejection + dynamic LLM prioritization. Apache 2.0. [arxiv 2509.19349](https://arxiv.org/abs/2509.19349); [GitHub](https://github.com/SakanaAI/ShinkaEvolve)

### MCTS + LLMs
- **MC-DML** (ICLR 2025): LLM as initial policy, learns from failure trajectories during planning.
- **ReKG-MCTS** (ACL Findings 2025): MCTS for knowledge graph reasoning with LLM guidance. [ACL](https://aclanthology.org/2025.findings-acl.484/)
- **MCTS Boosts Reasoning via Iterative Preference Learning**: Step-level reward signals from MCTS. [arxiv 2405.00451](https://arxiv.org/abs/2405.00451)
- **LLM-AHD**: MCTS for LLM-based automated heuristic design. [arxiv 2501.08603](https://arxiv.org/abs/2501.08603)

### In-Context RL
- **Reward Is Enough** (2025): ICRL prompting framework. Works on Game of 24, creative writing, ScienceWorld, AIME. Even works with self-generated rewards. [arxiv 2506.06303](https://arxiv.org/abs/2506.06303)
- **Survey of In-Context RL** (Feb 2025): Comprehensive survey. [arxiv 2502.07978](https://arxiv.org/abs/2502.07978)
- **TTRL: Test-Time Reinforcement Learning** (2025): [arxiv 2504.16084](https://arxiv.org/abs/2504.16084)

### Flow Matching + Reasoning
- **Discrete Flow Matching** (NeurIPS 2024): Flow matching for language/code. [arxiv 2407.15595](https://arxiv.org/abs/2407.15595)
- **FlowSeq** (EACL 2024): Flow matching for conditional text generation. Resolves slow sampling and noise schedule issues.
- **Controllable Flow Matching for RL**: Trajectory generation with hierarchical planning. [arxiv 2511.06816](https://arxiv.org/abs/2511.06816)

### Neuro-Symbolic
- **SATNet**: Differentiable MAXSAT solver embeddable in deep architectures. [arxiv 1905.12149](https://arxiv.org/abs/1905.12149)
- **DLM (Differentiable Logic Machines)**: Continuous relaxation of first-order logic programs. [arxiv 2102.11529](https://arxiv.org/abs/2102.11529)
- **Neural Theorem Provers**: Differentiable backward-chaining provers. [arxiv 1705.11040](https://arxiv.org/abs/1705.11040)

### World Modeling
- **Ha & Schmidhuber "World Models"** (2018): Generative RNNs build compressed spatial/temporal representations; policies train inside hallucinated environments. [arxiv 1803.10122](https://arxiv.org/abs/1803.10122); [worldmodels.github.io](https://worldmodels.github.io/)
- **DreamerV3** (Nature 2025): Single config outperforms specialized methods across 150+ tasks. First to collect diamonds in Minecraft from scratch. [arxiv 2301.04104](https://arxiv.org/abs/2301.04104)
- **RLVR-World** (Tsinghua, 2025): Directly bridges RLVR and world modeling — uses task-specific verifiable rewards to train world models. +30.7% accuracy on text games, +15.1% F1 on web prediction, +9.2% LPIPS on robot manipulation. [thuml.github.io/RLVR-World](https://thuml.github.io/RLVR-World/)
- **WorldLLM** (2025): Curiosity-driven theory-making for LLM world models. Natural language hypotheses + Bayesian refinement + RL-based curiosity. [arxiv 2506.06725](https://arxiv.org/abs/2506.06725)
- **RAP: Reasoning via Planning** (EMNLP 2023): LLM as both reasoning agent AND world model + MCTS. LLaMA-33B surpasses GPT-4 CoT by 33% on plan generation. [arxiv 2305.14992](https://arxiv.org/abs/2305.14992)
- **LLMs as World Simulators** (2024): GPT-4 tested as text-based world simulator — "still unreliable without further innovations." Fails on state transitions requiring arithmetic/common-sense reasoning. [arxiv 2406.06485](https://arxiv.org/abs/2406.06485)
- **World Models Survey** (ACM Computing Surveys 2025): 49-page systematic review. Two functions: understanding mechanisms vs predicting future states. [arxiv 2411.14499](https://arxiv.org/abs/2411.14499); [GitHub](https://github.com/tsinghua-fib-lab/World-Model)

## Methods & Techniques
- **GRPO**: Group comparison of responses, no critic model. Used in DeepSeek-R1. ~50% compute savings vs PPO.
- **Tree-GRPO**: DeepSearch's extension — MCTS-guided GRPO with asymmetric Q-value backup and soft clipping.
- **Entropic Objective**: TTT-Discover's log E[e^(β(s)R(s,a))] — optimizes for exceptional single solutions, not average performance.
- **PUCT (Predictor + UCT)**: AlphaZero-style state selection balancing exploitation (max child reward), guidance (prior ranking), exploration (visit counts). Used in TTT-Discover.
- **Evolutionary Search with LLMs**: LLM generates code candidates → evaluator scores → best seed next generation. Used in AlphaEvolve, FunSearch, ShinkaEvolve.
- **Global Frontier Selection**: DeepSearch's tree exploration prioritizing nodes by composite scoring (quality potential + uncertainty bonus + depth incentive) across entire tree.
- **Entropy-Based Guidance**: When no correct solution exists, select lowest-entropy incorrect trajectory (most confident mistake) for targeted learning.
- **Conditional Flow Matching (CFM)**: Construct probability paths p_t between source p₀ and target p₁, regress vector field u_θ(t,x). Simulation-free training.
- **Differentiable Logic**: Continuous relaxations of discrete logical operations (MAXSAT → SDP, first-order logic → weighted predicates) enabling gradient-based optimization.

## Implementations
- **ShinkaEvolve**: Python, Apache 2.0. [GitHub](https://github.com/SakanaAI/ShinkaEvolve) — Most accessible frozen-LLM evolutionary search framework.
- **FunSearch**: Python. [GitHub](https://github.com/google-deepmind/funsearch) — DeepMind's evolutionary code search.
- **OpenEvolve**: Open-source AlphaEvolve reimplementation. [HuggingFace](https://huggingface.co/blog/codelion/openevolve)
- **LLM-MCTS**: [GitHub](https://github.com/1989Ryan/llm-mcts) — MCTS + LLM integration.

## Open Gaps
- **RLVR beyond binary rewards**: Extending to domains without clean correct/incorrect signals (creative writing, open-ended science). Partially addressed by RLVRR (ICLR 2026).
- **Flow matching on reasoning**: No direct work on learning vector fields over logical reasoning trajectories. Discrete flow matching works for code but hasn't been applied to reasoning chains specifically.
- **Neuro-symbolic + flow matching integration**: Theoretical alignment (both use continuous relaxations) but no published work combining them.
- **In-context RL robustness**: Works empirically but theoretical understanding of when/why it fails is limited.
- **Frozen search vs weight updates for discovery**: No systematic comparison. TTT-Discover (frozen + test-time RL) and DeepSearch (weight updates with MCTS) represent different points but weren't compared head-to-head.
- **World modeling from RLVR**: RLVR-World shows verifiable rewards CAN improve world models (+30% accuracy), but current RLVR reasoning models still lack robust causal world understanding. LLMs develop implicit but brittle world models (OthelloGPT evidence). Gap between "solves math" and "understands physics" persists.
- **Scaling in-context RL**: Context window limits constrain how much learning history can be injected. No clear solution beyond longer contexts.

## Connections
- [[RLVR]] -- extends → [[RLHF]]: Replaces learned reward models with deterministic verifiers, solving reward hacking
- [[GRPO]] -- enables → [[RLVR]]: Efficient RL algorithm that makes RLVR practical (no critic model)
- [[DeepSearch]] -- combines → [[RLVR]] + [[MCTS]]: Uses MCTS for exploration during RLVR training, solving exploration bottleneck
- [[TTT-Discover]] -- combines → [[In-Context RL]] + [[PUCT]]: Test-time weight updates with tree search for discovery
- [[AlphaEvolve/FunSearch/ShinkaEvolve]] -- implements → [[Frozen LLM + Search]]: LLM as static generator in evolutionary loop
- [[In-Context RL]] -- alternative to → [[RLVR]]: Adaptation via context instead of weights; faster iteration but potentially less robust
- [[Flow Matching]] -- could extend to → [[Reasoning]]: Vector fields over reasoning trajectories (unexplored)
- [[Discrete Flow Matching]] -- bridges → [[Flow Matching]] + [[Language]]: Makes FM work for discrete tokens
- [[Differentiable Reasoning]] -- shares math with → [[Flow Matching]]: Both use continuous relaxations of discrete structures
- [[DeepSearch]] -- directly relevant to → [[User's CS224N project]]: MCTS + frozen/trained LLM for mathematical discovery
- [[TTT-Discover]] -- directly relevant to → [[User's CS224N project]]: Test-time adaptation for scientific discovery, uses PUCT like AlphaZero
- [[ShinkaEvolve]] -- most practical for → [[User's frozen LLM experiments]]: Open-source, sample-efficient, Apache 2.0
- [[VPRM]] -- extends → [[RLVR]]: Process-level rewards for intermediate reasoning steps
- [[RLVR-World]] -- bridges → [[RLVR]] + [[World Models]]: Verifiable rewards directly improve world model accuracy
- [[RAP]] -- combines → [[LLM as World Model]] + [[MCTS]]: LLM serves dual role as agent and world simulator
- [[World Models]] -- foundation for → [[AGI]]: Causal understanding of state transitions, not just pattern matching
- [[DreamerV3]] -- demonstrates → [[Learning inside world models]]: Policies trained entirely in imagination transfer to real environments

## Research Questions (User-Specific) — Analysis

### Q1: Frozen LLM + search vs RLVR weight updates for discovery?
**Evidence says: Complementary, not either/or.**
- Frozen search (AlphaEvolve, FunSearch, ShinkaEvolve) excels at **exploration** — rapidly generating diverse candidates without training cost. ShinkaEvolve achieves SOTA in 150 evals.
- RLVR weight updates excel at **internalization** — the model permanently learns reasoning patterns. DeepSeek-R1 went from 15.6%→77.9% on AIME.
- TTT-Discover bridges both: frozen model + test-time weight updates per problem (~$500/problem).
- **Recommendation**: For your CS224N project, frozen LLM + search is the more tractable direction (no training infrastructure needed). ShinkaEvolve is open-source and sample-efficient. But RLVR is more powerful for building lasting capabilities.

### Q2: In-context RL vs weight-based RL?
**Evidence says: In-context RL works but has limits.**
- "Reward Is Enough" (2025) shows ICRL works on AIME, Game of 24, ScienceWorld — even with self-generated rewards.
- But: context window limits constrain learning history. Weight updates have unbounded capacity.
- Algorithm Distillation (Laskin et al.) shows you can distill RL algorithms into context-processing, achieving better sample efficiency than the source algorithm.
- **For discovery**: In-context RL gives rapid iteration but may plateau on harder problems. Weight updates (RLVR/TTT-Discover) can push further.

### Q3: Flow matching on reasoning — feasibility?
**Evidence says: Theoretically promising, practically unexplored.**
- Discrete flow matching already works for code generation (NeurIPS 2024).
- Controllable flow matching for RL trajectory generation exists (arxiv 2511.06816).
- Differentiable reasoning (SATNet, DLM) shares the same mathematical foundation (continuous relaxations of discrete structures).
- **The gap**: Nobody has explicitly trained a flow matching model where the vector field represents logical reasoning steps. The pieces exist but haven't been assembled.
- **Feasibility**: Medium. Would require (1) defining a continuous space where reasoning trajectories live, (2) collecting paired (conjecture, proof) data as source/target distributions, (3) training CFM to interpolate. Main challenge: what is the right continuous representation of logical steps?

### Q4: Neuro-symbolic flow matching → autoregressive decoder?
**Evidence says: Not explored, but pieces exist.**
- Differentiable logic (DLM, SATNet) provides continuous relaxations of symbolic reasoning.
- Flow matching provides the generative framework for learning transformations between distributions.
- Decoding continuous representations into autoregressive text is standard (latent diffusion → text decoders exist).
- **This would be novel research** — no published work combines all three.

### Q5: Path from RLVR toward world modeling / AGI?
**Evidence says: RLVR is a necessary but insufficient step, but RLVR-World is promising.**
- RLVR-World (Tsinghua 2025) directly shows verifiable rewards improve world models by 15-30% across language and vision domains.
- DeepSeek-R1's emergent reasoning suggests RLVR implicitly incentivizes world modeling as a side effect of correctness optimization.
- But: LLMs develop brittle, implicit world models (Melanie Mitchell: "map-level" not "simulator-level"). OthelloGPT develops board representations but they're fragile.
- RAP framework shows LLM reasoning IS planning with a world model — MCTS + LLM-as-world-model beats GPT-4 CoT by 33%.
- TTT-Discover adapts to novel problems at test-time, showing generalization beyond memorization.
- **The gap**: Current world models are implicit and unreliable. Explicit world modeling (DreamerV3-style) combined with RLVR could be the path — train inside imagined worlds using verifiable rewards.
- **Speculative synthesis**: Flow matching could learn the "world model transformation" — mapping from current state to future state via learned vector fields, then using RLVR to verify predictions against reality.

## Strategic Recommendation for User

Given your CS224N project (frozen model + search algorithms) and interests:

**Highest-ROI direction: Frozen LLM + ShinkaEvolve-style search + MCTS for discovery**
1. Use ShinkaEvolve (open-source, sample-efficient) as your evolutionary search backbone
2. Add MCTS-guided exploration (inspired by DeepSearch's global frontier selection)
3. Experiment with in-context RL for memory/prompt updating (inspired by "Reward Is Enough")
4. This is the most tractable direction — no training infrastructure, open-source tools exist

**Medium-term pivot if results are promising: TTT-Discover style test-time RL**
- If frozen search plateaus, add lightweight test-time weight updates per problem
- This is what TTT-Discover does and it achieved breakthrough results

**Speculative long-term: Flow matching on reasoning**
- Novel research direction. Nobody has done it. Could be a strong thesis contribution.
- Start by framing reasoning trajectories as points in a continuous space
- Use existing discrete flow matching infrastructure as starting point
- Risk: may not work; reward: would be genuinely novel

**What to avoid:**
- Full RLVR from scratch (needs massive compute, unclear advantage over using existing RLVR-trained models)
- Pure neuro-symbolic approaches (too disconnected from LLM scaling trends)
- Reconstructing the full training stack (too large a project for CS224N timeline)

## Progress
- [x] Read user's two papers (2509.25454, 2601.16175)
- [x] Core RLVR foundations (DeepSeek-R1, GRPO, OpenAI o1)
- [x] In-context RL / in-context reinforcement learning
- [x] Search algorithms on frozen LLMs (AlphaEvolve, FunSearch, Shinka)
- [x] Flow matching foundations + reasoning applications
- [x] MCTS + LLMs landscape
- [x] Neuro-symbolic approaches to reasoning
- [x] World modeling via RL
- [x] Connections & synthesis
- [x] Strategic recommendation for user
