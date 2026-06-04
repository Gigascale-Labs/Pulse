# proj-sim

> **A scalable, validated, versatile MAS framework** — extracting behaviorally-grounded persona archetypes from real agent social data and deploying them in population-scale simulation.

---

## Architecture

```
Moltbook (3.1M posts)                    MiroFish simulation
  ↓                                            ↓
MiniLM 384-D embeddings             GraphRAG → entity graph
  ↓                                            ↓
k-means archetypes          →       NL persona descriptions
  ↓                                            ↓
representative posts        →       agent personas + memories
                                             ↓
                                    OASIS simulation engine
                                    (dual Twitter/Reddit env)
```

The persona extraction pipeline (left) feeds directly into MiroFish (right) as seed material. Instead of MiroFish's default news/policy seed, we use data-driven archetypes grounded in real Moltbook agent behaviour.

---

## Current Progress

### Stage 1 — Embedding
```
SimulaMet/moltbook-observatory-archive (3.1M posts)
  → text cleaning
  → MiniLM-L6-v2 384-D on NVIDIA RTX PRO 4000 Blackwell
  → moltbook_embeddings.npy (3105136, 384)
```

### Stage 2 — Clustering
```
moltbook_embeddings.npy
  → silhouette sweep k=3..8 on 250k sample
  → MiniBatchKMeans at optimal k
  → cluster_labels.npy, cluster_centroids.npy
```

### Stage 3 — Representative Post Retrieval
```
cluster centroids → cosine similarity → top-N posts per cluster
  → cluster_k_posts.txt
```

### Stage 4 — MiroFish Persona Generation
```
cluster_k_posts.txt → MiroFish GraphRAG pipeline
  → entity graph per cluster
  → NL persona descriptions
  → OASIS agent simulation
```

---

## Further Directions
- UMAP (384→50D) before clustering to improve silhouette scores
- K-medoids for real-post medoid anchoring
- Validation against Jiang et al. topic taxonomy and Stable Personas dual-assessment framework

---

## References

- Amin, D., Salminen, J., Jansen, B.J. (2026). *How to Model AI Agents as Personas?* arXiv:2603.03140
- Jiang, Y. et al. (2026). *Humans welcome to observe: A First Look at the Agent Social Network Moltbook.* arXiv:2602.10127
- Guo, H. (2025). *MiroFish: A Simple and Universal Swarm Intelligence Engine.* github.com/666ghj/MiroFish
- Liu, Z. et al. (2024). *OASIS: Open Agent Social Interaction Simulations with One Million Agents.* arXiv:2411.11581
