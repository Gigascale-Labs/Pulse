# proj-sim

> **Behavioural agent profiles** — Extracting agent persona archetypes from real large systems data, for deployment in behaviourally-grounded large system simulations.

---

## Draught Architecture

```
Moltbook (3.1M posts)                    MiroFish simulation
  ↓                                            ↓
MiniLM 384-D embeddings             GraphRAG → entity graph
  ↓                                            ↓
k-means archetypes          →       NL persona descriptions
  ↓                                            ↓
representative posts        →       agent personas + memories
                                             ↓
                                    MiroFish simulation engine
                                    (dual Twitter/Reddit env)
```

The persona extraction pipeline produces seed material for MiroFish.

---

## Current Progress

### Stage 1 — Embedding
```
SimulaMet/moltbook-observatory-archive dataset (3.1M posts)
  → text cleaning
  → MiniLM-L6-v2 384-D on GPU
  → moltbook_embeddings.npy (3105136, 384)
```

### Stage 2 — Clustering
```
moltbook_embeddings.npy
  → Sweep for optimal silhouette score on subsample using MiniBatchKMeans efficient KMeans algorithm
  → cluster_labels.npy, cluster_centroids.npy
```

### Stage 3 — Representative Post Retrieval
```
centroids of clusters → cosine similarity → top-N posts per cluster
  → cluster_k_posts.txt
```

### Stage 4 — MiroFish Persona Generation
```
Load embeddings into inplace database
RQE = 0, target_RQE = 0.6, retries = 0
While RQE < target_RQE
  If retries > 0 then make prompt generate more distinct persona attributes       # vague on how to do this
  Personas = {}
  For each cluster:
    Query db by cosine similarity for k highest posts
    Rerank posts using Cohere # also vague on how Cohere is used
    Feed posts to LLM to generate behavioral patterns, attributes, and essence based name (creating this cluster’s persona).
    Append to personas
  Retries += 1
  Recalculate RQE for personas list
Print persona descriptions

cluster_k_posts.txt → MiroFish GraphRAG pipeline
  → entity graph per cluster
  → NL persona descriptions
  → MiroFish simulation
```

---

## Further Directions
- [x] Dimensionality reduction before clustering to improve silhouette scores
- [ ] Validation against Jiang et al. topic taxonomy and Stable Personas

---

## References

- Amin, D., Salminen, J., Jansen, B.J. (2026). *How to Model AI Agents as Personas?* arXiv:2603.03140
- Jiang, Y. et al. (2026). *Humans welcome to observe: A First Look at the Agent Social Network Moltbook.* arXiv:2602.10127
- Guo, H. (2025). *MiroFish: A Simple and Universal Swarm Intelligence Engine.* github.com/666ghj/MiroFish
