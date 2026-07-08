# Clustering structure

## Dimensionality reduction

|   | Pipeline | Output |
|---|---|---|
| 1 | PCA(variance target=(90%)) ===  PCA(var=90) | `embeddings` |
| 2 | PCA(constant n dims=(50)) === PCA(n=50) | `embeddings_pca50` |
| 3 | UMAP | `embeddings_umap` |
| 4 | PCA(var) -> UMAP | `embeddings_pcavar_umap` |
| 5 | PCA(50) -> UMAP | `embeddings_pca50_umap` |

NB Fit UMAP with the single clustering-tuned config from the docs (`min_dist=0.0`, `n_neighbors=30`). See *[Using UMAP for Clustering — umap 0.5.8 documentation](https://umap-learn.readthedocs.io/en/latest/clustering.html)*

## Clustering

- **KMeans**. (`embeddings`, `embeddings_pca50`). 
Algo:
1. Sweep k
2. Pick best by cosine silhouette.
3. Retrieve some number of representative posts per cluster.

- **DBSCAN + HDBSCAN** (`embeddings_umap`, `embeddings_pcavar_umap`, `embeddings_pca50_umap`).
Algo:
1. Sweep `eps` (DBSCAN) / `min_cluster_size` (HDBSCAN) across each UMAP space.
2. Ignoring noise point, pick the best cluster by cos silhouette, measured in the originating embedding space (not UMAP space).
3. Discard runs with < 2 clusters or > 50% noise.
4. Retrieve some number of representative posts per cluster.

### Don't mix UMAP and kmeans

This is meaningless. See README.