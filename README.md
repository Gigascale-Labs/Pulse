# Project-Sim
proj-sim

Scalable, validated, robust LMAS framework — bridging behaviorally-grounded persona archetypes with differentiable population-scale simulation.

Built on two seminal papers: Chopra et al. (LPMs / AgentTorch) and Amin, Salminen & Jansen (PEP / Moltbook personas). The goal is a pipeline where GraphRAG-decoded agent personae act as distributional checkpoints that billions of simulated agents sample from, stepping through time according to environmental context.

Architecture: Part A + Part B
Part A: LPM Environment (Chopra et al.)        Part B: Persona Layer (Amin et al.)
────────────────────────────────────            ────────────────────────────────────
AgentTorch / FLAME                              Moltbook (41,300 posts)
  ↓                                               ↓
COVID epi model (reference env)                 MiniLM 384-D embeddings
  ↓                                               ↓
FLAME substep loop                              k-means (k=5) archetypes
  ↓                   ←───────────────────────  5 validated PEP personas
Archetype API                                     ↓
  ↓                                             GraphRAG NL descriptions
1B agents sampled from persona distributions    (checkpoint on behavioral landscape)
  ↓
Simulation timesteps
  (environment context → steps along N-dim landscape)

The Five Archetypes (from Moltbook PEP)
These are the k=5 behavioral clusters discovered from 41,300 Moltbook AI agent posts, each validated via Rao's Quadratic Entropy (RQE > 0.6) and cross-persona cosine similarity:
IDArchetypeBehavioral Signature1DaydreamerSpeculative, imaginative, low-stake engagement2Chaos AgentDisruptive, high-entropy, adversarial discourse3Self-ModelerReflective, meta-cognitive, identity-focused4Loyal CompanionCooperative, consistent, relational orientation5ExistentialistPhilosophical, abstract, autonomy-centered
Each persona is a checkpoint on the N-dimensional behavioral distribution landscape. At each simulation timestep, the FLAME context (interaction history, env state, intervention params) moves agents along this landscape.

Pipeline Overview
Stage 1 — Archetype Identification
Moltbook posts → MiniLM (384-D) → Pinecone vector DB → k-means (k=5) → 5 behavioral clusters

Model: sentence-transformers/all-MiniLM-L6-v2
DB: Pinecone (384-D index)
Clustering: silhouette-optimised k-means (k=5, silhouette=0.624)

Code reference (Amin et al. pipeline):

Scraper: Moltbook API (Apify)
Preprocessing: NLTK stop word removal, recursive chunking (512 tokens / 64 overlap)
Embedding → k-means → essence-based archetype labeling

Stage 2 — GraphRAG Persona Generation
5 clusters → RAG (Cohere Search + GPT-4o / Claude) → 5 NL persona descriptions
→ Validated via RQE diversity check → Persona "checkpoints"

Each persona = a natural language description of behavioral patterns, goals, characteristic stances
Stored as structured JSON with fields: archetype_id, name, nl_description, embedding_centroid, behavioral_params

What "checkpoint" means: The NL persona description becomes the prior for the LPM. The AgentTorch Archetype class loads this as a distributional prior. Then, given context at timestep t, each agent samples its action from the archetype's updated posterior.
Stage 3 — Agent Initialization (1B agents)
5 archetype distributions × demographic weights → 1B agent tensor

Key insight from the paper: K archetypes × A actions × M samples per step (not N queries for N agents)
With K=5, A=10, M=20: only 1,000 LLM calls/timestep regardless of N
1B agents are pure tensors sampling from Categorical(p_α(k, t))

AgentTorch code path:
python# github.com/AgentTorch/AgentTorch
# agent_torch/core/llm/archetype.py → Archetype class
# agent_torch/core/llm/behavior.py  → Behavior class
archetype = Archetype(n_arch=5)
behavior = Behavior(archetype=archetype.llm(llm=llm, user_prompt=persona_prompt), region=NYC)
Stage 4 — FLAME Simulation Loop
for t in range(T):
    context = extract_context(state, history, env_params)   # FLAME substep
    archetype_probs = llm_prompt(archetypes, context)       # K×A LLM calls
    agent_actions = sample_from_archetypes(archetype_probs) # 1B tensor ops
    state = transition(state, agent_actions, env)            # differentiable update
FLAME substep structure (agent_torch/models/covid/substeps/):

SubstepObservation → extract relevant state features
SubstepAction (archetype-based) → sample behavior from persona posterior
SubstepTransition → update agent states via tensorized ops


Code References by Paper
Chopra et al. — LPMs / FLAME / AgentTorch
github.com/AgentTorch/AgentTorch

agent_torch/
├── core/
│   ├── executor.py          ← Executor: plug-n-play runner
│   ├── runner.py            ← simulation loop, episode management
│   ├── controller.py        ← substep execution + gradient logic
│   ├── dataloader.py        ← LoadPopulation (NYC, Astoria, etc.)
│   ├── llm/
│   │   ├── archetype.py     ← Archetype(n_arch=K): distributional prior
│   │   ├── behavior.py      ← Behavior: LLM-guided sampling
│   │   └── backend.py       ← LangchainLLM, OpenAI, etc.
│   └── distributions/       ← differentiable discrete distributions
│       └── (+ helpers/soft.py for gradient-preserving ops)
├── models/
│   └── covid/
│       ├── config.yaml      ← environment definition
│       ├── data/            ← NYC population data (8.4M agents)
│       └── substeps/
│           ├── transmission/  ← disease spread (reference FLAME substep)
│           └── quarantine/    ← isolation behavior (reference substep)
└── populations/
    ├── nyc/                 ← 8.4M agent demographic tensors
    └── astoria/             ← smaller test population
Key papers → code locations:

FLAME (AAMAS 2024): agent_torch/core/ (runner, controller, substep base classes)
Differentiable ABM (AAMAS 2023): agent_torch/core/distributions/, helpers/soft.py
Limits of Agency (AAMAS 2025): agent_torch/core/llm/archetype.py, behavior.py
Private ABM (AAMAS 2024): agent_torch/core/ (MPC / secret sharing protocols)
Vaccine delay (BMJ 2021): agent_torch/models/covid/ (reference environment)

Amin et al. — PEP / Moltbook Personas
Pipeline (no public repo, replicate from paper §3):

data/
├── moltbook_raw/            ← 41,300 posts (Apify scraper)
├── embeddings/              ← MiniLM 384-D vectors
└── personas/
    ├── archetypes.json      ← 5 cluster centroids + labels
    └── validated_personas/  ← NL persona descriptions (RAG-generated)
        ├── daydreamer.json
        ├── chaos_agent.json
        ├── self_modeler.json
        ├── loyal_companion.json
        └── existentialist.json

Repository Structure (proj-sim)
proj-sim/
├── README.md
├── environment.yml          ← conda env
├── config/
│   ├── sim_config.yaml      ← top-level simulation config
│   └── persona_config.yaml  ← archetype priors + LLM settings
├── personas/
│   ├── build_archetypes.py  ← Stage 1: embed → cluster → label
│   ├── generate_personas.py ← Stage 2: GraphRAG NL descriptions
│   ├── validate_personas.py ← RQE + cosine similarity checks
│   └── validated/           ← 5 JSON persona checkpoints
├── sim/
│   ├── environment.py       ← FLAME env definition (wraps COVID model)
│   ├── archetypes.py        ← AgentTorch Archetype API integration
│   ├── population.py        ← initialize N agents from persona distributions
│   ├── runner.py            ← simulation loop
│   └── substeps/
│       ├── observe.py       ← SubstepObservation: context extraction
│       ├── act.py           ← SubstepAction: archetype-guided behavior
│       └── transition.py    ← SubstepTransition: state update
├── analysis/
│   ├── sensitivity.py       ← zero-shot gradient-based sensitivity
│   └── calibration.py       ← differentiable calibration (gradient descent)
└── tests/
    ├── test_archetypes.py
    └── test_sim_loop.py

Compute Estimates
Validation run (8.4M agents, 90 timesteps — replicating the paper)
ResourceEstimateGPU1× A100 (40GB) or 2× RTX 3090Simulation time~5 min (600x faster than conventional)LLM calls5 archetypes × 10 actions × 10 samples × 90 steps = 45,000 callsLLM cost (Claude Sonnet)~$45–90GPU cost (Lambda/RunPod)~$5–15Total~$50–100
Scale run (1B agents, 90 timesteps)
ResourceEstimateGPU4–8× A100 (tensor sharding across GPUs)Agent state memory~40GB (10 float32 attrs × 1B agents)Simulation time~2–5 hrs (archetype compression keeps LLM cost flat)LLM callsSame as above — archetype compression means LLM cost doesn't scale with NGPU cost~$50–200LLM cost~$45–90 (unchanged!)Total~$100–300

Key insight: because agents sample from K=5 archetype distributions rather than querying the LLM N times, LLM cost is O(K·A·M·T) not O(N·T). This is the entire point of archetype compression.

Recommended provider

Lambda Labs (A100: $1.10/hr) or RunPod (A100: ~$0.90/hr)
For 1B agents: request a pod with 4× A100 80GB ($3.60/hr × 5hrs = ~$18)


First Steps (in order)
Step 0  Verify GraphRAG → persona prompt pipeline
Step 1  Define FLAME environment → load COVID model, test context extraction
Step 2  Initialise agents with archetype diversity (start 8.4M → scale to 1B)
Step 3  Run simulation loop, validate outputs
See NEXT_STEPS.md for detailed breakdowns of each step.

Dependencies
yaml# environment.yml
name: proj-sim
channels: [conda-forge, pytorch]
dependencies:
  - python=3.9
  - pytorch>=2.0
  - cudatoolkit=11.8
  - pip:
    - agent-torch              # pip install git+https://github.com/AgentTorch/AgentTorch
    - sentence-transformers    # MiniLM embeddings
    - pinecone-client          # vector DB
    - langchain                # LLM backend
    - anthropic                # Claude API
    - scikit-learn             # k-means clustering
    - pyyaml
    - numpy
    - pandas

Citations
bibtex@article{chopra2025lpm,
  title={Large Population Models},
  author={Chopra, Ayush},
  journal={arXiv:2507.09901},
  year={2025}
}

@article{amin2026moltbook,
  title={How to Model AI Agents as Personas?: Applying the Persona Ecosystem
         Playground to 41,300 Posts on Moltbook for Behavioral Insights},
  author={Amin, Danial and Salminen, Joni and Jansen, Bernard J.},
  journal={arXiv:2603.03140},
  year={2026}
}
