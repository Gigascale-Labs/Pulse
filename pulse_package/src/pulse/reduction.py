"""
Purpose:
Create reduced embedding spaces from sampled embeddings.

This file should return a dictionary of named embedding spaces:
- raw
- pca_variance
- pca50
- umap_raw
- umap_pca_variance
- umap_pca50
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from pulse.config import PCAConfig, ReductionConfig, UMAPConfig
from pulse.contracts import ReductionResult, SampledPosts


ReductionBackend = Literal["cuml", "sklearn"]


def reduce_embeddings(
    sampled: SampledPosts,
    config: ReductionConfig,
    backend: ReductionBackend = "cuml",
) -> dict[str, ReductionResult]:
    """
    Create reduced embedding spaces for clustering.

    Args:
        sampled: Sampled posts and aligned embedding vectors.
        config: Dimensionality reduction settings.
        backend: Implementation backend used for reduction.

    Returns:
        Reduced embedding spaces keyed by name.
    """

    spaces: dict[str, ReductionResult] = {}

    raw = ReductionResult(
        name="raw",
        row_indices=sampled.row_indices.copy(),
        vectors=sampled.vectors.astype(np.float32, copy=False),
        metadata={
            "method": "none",
            "n_dimensions": int(sampled.vectors.shape[1]),
        },
    )
    spaces[raw.name] = raw

    pca_variance = run_pca(
        vectors=sampled.vectors,
        row_indices=sampled.row_indices,
        config=config.pca,
        backend=backend,
        name="pca_variance",
    )
    spaces[pca_variance.name] = pca_variance

    pca50 = run_fixed_pca(
        vectors=sampled.vectors,
        row_indices=sampled.row_indices,
        n_components=config.pca.pca50_components,
        backend=backend,
        name="pca50",
    )
    spaces[pca50.name] = pca50

    if config.umap.enabled:
        spaces["umap_raw"] = run_umap(
            vectors=raw.vectors,
            row_indices=raw.row_indices,
            config=config.umap,
            backend=backend,
            name="umap_raw",
            source_space="raw",
        )

        spaces["umap_pca_variance"] = run_umap(
            vectors=pca_variance.vectors,
            row_indices=pca_variance.row_indices,
            config=config.umap,
            backend=backend,
            name="umap_pca_variance",
            source_space="pca_variance",
        )

        spaces["umap_pca50"] = run_umap(
            vectors=pca50.vectors,
            row_indices=pca50.row_indices,
            config=config.umap,
            backend=backend,
            name="umap_pca50",
            source_space="pca50",
        )

    return spaces


def run_pca(
    vectors: np.ndarray,
    row_indices: np.ndarray,
    config: PCAConfig,
    backend: ReductionBackend,
    name: str,
) -> ReductionResult:
    """
    Run PCA using either explained variance or a fixed number of components.

    Args:
        vectors: Input embedding matrix.
        row_indices: Row positions aligned with the vectors.
        config: PCA settings.
        backend: Implementation backend.
        name: Name assigned to the reduced space.

    Returns:
        PCA reduction result.
    """

    if config.mode == "variance":
        return run_variance_pca(
            vectors=vectors,
            row_indices=row_indices,
            explained_variance=config.explained_variance,
            backend=backend,
            name=name,
        )

    return run_fixed_pca(
        vectors=vectors,
        row_indices=row_indices,
        n_components=config.n_components,
        backend=backend,
        name=name,
    )


def run_fixed_pca(
    vectors: np.ndarray,
    row_indices: np.ndarray,
    n_components: int,
    backend: ReductionBackend,
    name: str,
) -> ReductionResult:
    """
    Run PCA with a fixed number of output dimensions.

    Args:
        vectors: Input embedding matrix.
        row_indices: Row positions aligned with the vectors.
        n_components: Number of PCA dimensions.
        backend: Implementation backend.
        name: Name assigned to the reduced space.

    Returns:
        PCA reduction result.
    """

    n_components = _validate_n_components(vectors, n_components)

    if backend == "cuml":
        reduced, metadata = _run_cuml_pca(vectors, n_components)
    elif backend == "sklearn":
        reduced, metadata = _run_sklearn_pca(vectors, n_components)
    else:
        raise ValueError(f"Unsupported reduction backend: {backend}")

    return ReductionResult(
        name=name,
        row_indices=row_indices.copy(),
        vectors=reduced.astype(np.float32, copy=False),
        metadata={
            **metadata,
            "method": "pca",
            "mode": "components",
            "n_components": int(n_components),
            "backend": backend,
        },
    )


def run_variance_pca(
    vectors: np.ndarray,
    row_indices: np.ndarray,
    explained_variance: float,
    backend: ReductionBackend,
    name: str,
) -> ReductionResult:
    """
    Run PCA and keep enough dimensions to reach the target explained variance.

    Args:
        vectors: Input embedding matrix.
        row_indices: Row positions aligned with the vectors.
        explained_variance: Target cumulative explained variance.
        backend: Implementation backend.
        name: Name assigned to the reduced space.

    Returns:
        PCA reduction result.
    """

    max_components = min(vectors.shape[0], vectors.shape[1])

    if backend == "cuml":
        full_reduced, metadata = _run_cuml_pca(vectors, max_components)
    elif backend == "sklearn":
        full_reduced, metadata = _run_sklearn_pca(vectors, max_components)
    else:
        raise ValueError(f"Unsupported reduction backend: {backend}")

    ratios = np.asarray(metadata["explained_variance_ratio"], dtype=np.float64)
    cumulative = np.cumsum(ratios)
    n_components = int(np.searchsorted(cumulative, explained_variance) + 1)
    n_components = max(1, min(n_components, full_reduced.shape[1]))

    reduced = full_reduced[:, :n_components]

    return ReductionResult(
        name=name,
        row_indices=row_indices.copy(),
        vectors=reduced.astype(np.float32, copy=False),
        metadata={
            **metadata,
            "method": "pca",
            "mode": "variance",
            "target_explained_variance": float(explained_variance),
            "selected_components": int(n_components),
            "selected_explained_variance": float(cumulative[n_components - 1]),
            "backend": backend,
        },
    )


def run_umap(
    vectors: np.ndarray,
    row_indices: np.ndarray,
    config: UMAPConfig,
    backend: ReductionBackend,
    name: str,
    source_space: str,
) -> ReductionResult:
    """
    Run UMAP on an embedding space.

    Args:
        vectors: Input embedding matrix.
        row_indices: Row positions aligned with the vectors.
        config: UMAP settings.
        backend: Implementation backend.
        name: Name assigned to the reduced space.
        source_space: Name of the input embedding space.

    Returns:
        UMAP reduction result.
    """

    if backend == "cuml":
        reduced = _run_cuml_umap(vectors, config)
    elif backend == "sklearn":
        reduced = _run_cpu_umap(vectors, config)
    else:
        raise ValueError(f"Unsupported reduction backend: {backend}")

    return ReductionResult(
        name=name,
        row_indices=row_indices.copy(),
        vectors=reduced.astype(np.float32, copy=False),
        metadata={
            "method": "umap",
            "source_space": source_space,
            "n_components": config.n_components,
            "n_neighbors": config.n_neighbors,
            "min_dist": config.min_dist,
            "random_state": config.random_state,
            "metric": config.metric,
            "backend": backend,
        },
    )


def _run_cuml_pca(
    vectors: np.ndarray,
    n_components: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """
    Run PCA with cuML.
    """

    try:
        from cuml.decomposition import PCA
    except ImportError as exc:
        raise ImportError(
            "cuML PCA is unavailable. Install the CUDA dependencies or use "
            "backend='sklearn'."
        ) from exc

    model = PCA(n_components=n_components)
    reduced = model.fit_transform(vectors.astype(np.float32, copy=False))

    reduced_np = _to_numpy(reduced)
    ratios = _to_numpy(model.explained_variance_ratio_)

    return reduced_np, {
        "explained_variance_ratio": ratios.tolist(),
    }


def _run_sklearn_pca(
    vectors: np.ndarray,
    n_components: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """
    Run PCA with scikit-learn.
    """

    from sklearn.decomposition import PCA

    model = PCA(n_components=n_components, random_state=42)
    reduced = model.fit_transform(vectors)

    return reduced.astype(np.float32, copy=False), {
        "explained_variance_ratio": model.explained_variance_ratio_.tolist(),
    }


def _run_cuml_umap(
    vectors: np.ndarray,
    config: UMAPConfig,
) -> np.ndarray:
    """
    Run UMAP with cuML.
    """

    try:
        from cuml.manifold import UMAP
    except ImportError as exc:
        raise ImportError(
            "cuML UMAP is unavailable. Install the CUDA dependencies or use "
            "backend='sklearn'."
        ) from exc

    model = UMAP(
        n_components=config.n_components,
        n_neighbors=config.n_neighbors,
        min_dist=config.min_dist,
        random_state=config.random_state,
        metric=config.metric,
    )

    reduced = model.fit_transform(vectors.astype(np.float32, copy=False))
    return _to_numpy(reduced)


def _run_cpu_umap(
    vectors: np.ndarray,
    config: UMAPConfig,
) -> np.ndarray:
    """
    Run UMAP with the CPU umap-learn package.
    """

    try:
        import umap
    except ImportError as exc:
        raise ImportError(
            "CPU UMAP is unavailable. Install 'umap-learn' or disable UMAP."
        ) from exc

    model = umap.UMAP(
        n_components=config.n_components,
        n_neighbors=config.n_neighbors,
        min_dist=config.min_dist,
        random_state=config.random_state,
        metric=config.metric,
    )

    reduced = model.fit_transform(vectors)
    return reduced.astype(np.float32, copy=False)


def _validate_n_components(
    vectors: np.ndarray,
    n_components: int,
) -> int:
    """
    Validate a PCA component count against the input matrix.
    """

    if vectors.ndim != 2:
        raise ValueError(f"vectors must be a 2D array. Got shape {vectors.shape}.")

    if n_components <= 0:
        raise ValueError("n_components must be greater than zero.")

    max_components = min(vectors.shape[0], vectors.shape[1])

    if n_components > max_components:
        return max_components

    return n_components


def _to_numpy(value) -> np.ndarray:
    """
    Convert common GPU array objects to NumPy.
    """

    if hasattr(value, "get"):
        return value.get()

    if hasattr(value, "to_numpy"):
        return value.to_numpy()

    return np.asarray(value)