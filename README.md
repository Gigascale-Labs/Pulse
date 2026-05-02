# Project-Sim
# proj-sim

> **Scalable, validated, robust LMAS framework** — bridging behaviorally-grounded persona archetypes with differentiable population-scale simulation.

Built on two seminal papers: Chopra et al. (LPMs / AgentTorch) and Amin, Salminen & Jansen (PEP / Moltbook personas). The goal is a pipeline where GraphRAG-decoded agent personae act as distributional checkpoints that billions of simulated agents sample from, stepping through time according to environmental context.

---

## Architecture: Part A + Part B

```
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
```

---


## Pipeline Overview

### Stage 1 — Archetype Identification
```
Moltbook posts → MiniLM (384-D) → Pinecone vector DB → k-means (k=5) → 5 behavioral clusters
```

### Stage 2 — GraphRAG Persona Generation
```
5 clusters → RAG (Cohere Search + GPT-4o / Claude) → 5 NL persona descriptions
→ Validated via RQE diversity check → Persona "checkpoints"
```

**What "checkpoint" means:** The NL persona description becomes the prior for the LPM. The AgentTorch `Archetype` class loads this as a distributional prior. Then, given context at timestep `t`, each agent samples its action from the archetype's updated posterior.

### Stage 3 — Agent Initialization (1B agents)

### Stage 4 — FLAME Simulation Loop


## Citations

```bibtex
@article{chopra2025lpm,
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
```}
}

@article{amin2026moltbook,
  title={How to Model AI Agents as Personas?: Applying the Persona Ecosystem
         Playground to 41,300 Posts on Moltbook for Behavioral Insights},
  author={Amin, Danial and Salminen, Joni and Jansen, Bernard J.},
  journal={arXiv:2603.03140},
  year={2026}
}
