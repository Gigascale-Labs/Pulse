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

- **Spherical KMeans**. (`embeddings`, `embeddings_pca50`).
Algo:
0. L2-normalise the PCA output onto the unit sphere (see below).
1. Sweep k
2. Pick best by cosine silhouette.
3. Retrieve some number of representative posts per cluster.

### Silhouette does not pick k here

Measured on the 100k subsample: the cosine silhouette curve is flat. PCA50 sits at 0.105-0.113
from k=12 out to k=300; PCA184 drifts from 0.069 up to 0.086 over the same range with no
interior peak. The spread across the whole plateau is close to the run-to-run wobble, so the
sweep's `argmax` mostly reports wherever the search range happened to stop.

Treat the sweep as a degeneracy check (watch largest-cluster % and smallest-cluster n, both of
which are well behaved now) and choose k from the representative-post dumps or from the number
of personae wanted downstream. Note also that PCA50 scores higher than PCA184 at every k; that
is largely the usual silhouette-in-fewer-dimensions effect, not evidence that PCA50 is the
better space — it retains 57.7% of variance against 90%.

### Why spherical, not plain, KMeans

Encoding produces unit-norm vectors, but PCA centres and rotates them, so length is no longer
constant. Plain KMeans would then use that length, and length in PCA space means "distance from
the data mean" — i.e. how atypical a post is, not what it is about. Clusters could split by
atypicality rather than topic, with no way to tell which happened.

Spherical KMeans removes this: vectors are re-normalised before fitting and each centroid is
projected back onto the unit sphere after every mean update, so only direction matters. It also
makes the fit agree with the rest of the pipeline, which selects representative posts and scores
personae by cosine similarity, and it settles the silhouette question: cosine is what the
algorithm optimises, so cosine is what we score with. (Euclidean tracks it closely on unit
vectors — Spearman 0.94 across the k sweep — but not exactly, since d = sqrt(2 - 2cos) is
non-linear, so the two can disagree on the argmax.)

cuML has no spherical variant, so the Lloyd loop is implemented directly on the GPU
(`spherical_kmeans` in the notebook), seeded per restart from a cuML k-means++ init.

If topic *and* intensity are both wanted, do it in two explicit stages: spherical KMeans for the
topic, then split within each cluster by distance from the centroid.

- **DBSCAN + HDBSCAN** (`embeddings_umap`, `embeddings_pcavar_umap`, `embeddings_pca50_umap`).
Algo:
1. Sweep `eps` (DBSCAN) / `min_cluster_size` (HDBSCAN) across each UMAP space.
2. Ignoring noise point, pick the best cluster by cos silhouette, measured in the originating embedding space (not UMAP space).
3. Discard runs with < 2 clusters or > 50% noise.
4. Retrieve some number of representative posts per cluster.

### Don't mix UMAP and kmeans

This is meaningless. See README.