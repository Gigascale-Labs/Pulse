'''Run clustering over reduced embedding spaces and return ClusterResult objects.'''

from __future__ import annotations

from typing import Iterable, Literal

import numpy as np

from pulse.config import ClusteringConfig, DensityClusteringConfig, KMeansConfig
from pulse.contracts import ClusterResult, ReductionResult


ClusteringBackend = Literal["cuml", "sklearn"]


def cluster_reductions(
    reductions: dict[str, ReductionResult],
    config: ClusteringConfig,
) -> list[ClusterResult]:
    """
    Run configured clustering methods over reduced embedding spaces.

    Args:
        reductions: Reduced embedding spaces keyed by name.
        config: Clustering settings.

    Returns:
        Cluster results for all configured clustering runs.
    """

    results: list[ClusterResult] = []

    if config.kmeans.enabled:
        results.extend(
            run_kmeans_clustering(
                reductions=reductions,
                config=config.kmeans,
            )
        )

    if config.density.enabled:
        results.extend(
            run_density_clustering(
                reductions=reductions,
                config=config.density,
                backend=config.kmeans.backend,
            )
        )

    return results


def run_kmeans_clustering(
    reductions: dict[str, ReductionResult],
    config: KMeansConfig,
    space_names: Iterable[str] | None = None,
) -> list[ClusterResult]:
    """
    Run KMeans clustering over selected embedding spaces.

    Args:
        reductions: Reduced embedding spaces keyed by name.
        config: KMeans clustering settings.
        space_names: Optional subset of embedding spaces to cluster.

    Returns:
        Cluster results for each space and cluster size.
    """

    selected_spaces = _select_spaces(reductions, space_names)
    results: list[ClusterResult] = []

    for space_name, reduction in selected_spaces.items():
        for n_clusters in config.cluster_sizes:
            result = run_single_kmeans(
                reduction=reduction,
                n_clusters=n_clusters,
                config=config,
            )
            results.append(result)

    return results


def run_single_kmeans(
    reduction: ReductionResult,
    n_clusters: int,
    config: KMeansConfig,
) -> ClusterResult:
    """
    Run one KMeans clustering job for one embedding space.

    Args:
        reduction: Embedding space used for clustering.
        n_clusters: Number of clusters.
        config: KMeans settings.

    Returns:
        Cluster labels aligned with the reduction rows.
    """

    if n_clusters <= 1:
        raise ValueError("n_clusters must be greater than one.")

    if n_clusters > reduction.vectors.shape[0]:
        raise ValueError(
            "n_clusters cannot exceed the number of records. "
            f"Got {n_clusters} clusters for {reduction.vectors.shape[0]} records."
        )

    if config.backend == "cuml":
        labels, metadata = _run_cuml_kmeans(
            vectors=reduction.vectors,
            n_clusters=n_clusters,
            random_state=config.random_state,
            n_init=config.n_init,
        )
    elif config.backend == "sklearn":
        labels, metadata = _run_sklearn_kmeans(
            vectors=reduction.vectors,
            n_clusters=n_clusters,
            random_state=config.random_state,
            n_init=config.n_init,
        )
    else:
        raise ValueError(f"Unsupported KMeans backend: {config.backend}")

    return ClusterResult(
        method="kmeans",
        space_name=reduction.name,
        row_indices=reduction.row_indices.copy(),
        labels=labels.astype(np.int32, copy=False),
        backend=config.backend,
        metadata={
            **metadata,
            "n_clusters": int(n_clusters),
            "space_name": reduction.name,
            "space_metadata": reduction.metadata,
            "n_posts_per_cluster": config.n_posts_per_cluster,
        },
    )


def run_density_clustering(
    reductions: dict[str, ReductionResult],
    config: DensityClusteringConfig,
    backend: ClusteringBackend,
    space_names: Iterable[str] | None = None,
) -> list[ClusterResult]:
    """
    Run optional DBSCAN and HDBSCAN clustering experiments.

    Args:
        reductions: Reduced embedding spaces keyed by name.
        config: Density clustering settings.
        backend: Implementation backend.
        space_names: Optional subset of embedding spaces to cluster.

    Returns:
        Cluster results for configured density clustering runs.
    """

    selected_spaces = _select_spaces(reductions, space_names)
    results: list[ClusterResult] = []

    for space_name, reduction in selected_spaces.items():
        for eps in config.dbscan_eps_values:
            results.append(
                run_single_dbscan(
                    reduction=reduction,
                    eps=eps,
                    min_samples=config.dbscan_min_samples,
                    backend=backend,
                    n_posts_per_cluster=config.n_posts_per_cluster,
                )
            )

        for min_cluster_size in config.hdbscan_min_cluster_sizes:
            results.append(
                run_single_hdbscan(
                    reduction=reduction,
                    min_cluster_size=min_cluster_size,
                    backend=backend,
                    n_posts_per_cluster=config.n_posts_per_cluster,
                )
            )

    return results


def run_single_dbscan(
    reduction: ReductionResult,
    eps: float,
    min_samples: int,
    backend: ClusteringBackend,
    n_posts_per_cluster: int,
) -> ClusterResult:
    """
    Run one DBSCAN clustering job.

    Args:
        reduction: Embedding space used for clustering.
        eps: DBSCAN epsilon value.
        min_samples: Minimum samples value.
        backend: Implementation backend.
        n_posts_per_cluster: Number of representative posts exported per cluster.

    Returns:
        Cluster labels aligned with the reduction rows.
    """

    if backend == "cuml":
        labels, metadata = _run_cuml_dbscan(reduction.vectors, eps, min_samples)
    elif backend == "sklearn":
        labels, metadata = _run_sklearn_dbscan(reduction.vectors, eps, min_samples)
    else:
        raise ValueError(f"Unsupported density clustering backend: {backend}")

    return ClusterResult(
        method="dbscan",
        space_name=reduction.name,
        row_indices=reduction.row_indices.copy(),
        labels=labels.astype(np.int32, copy=False),
        backend=backend,
        metadata={
            **metadata,
            "eps": float(eps),
            "min_samples": int(min_samples),
            "space_name": reduction.name,
            "n_posts_per_cluster": n_posts_per_cluster,
        },
    )


def run_single_hdbscan(
    reduction: ReductionResult,
    min_cluster_size: int,
    backend: ClusteringBackend,
    n_posts_per_cluster: int,
) -> ClusterResult:
    """
    Run one HDBSCAN clustering job.

    Args:
        reduction: Embedding space used for clustering.
        min_cluster_size: Minimum cluster size.
        backend: Implementation backend.
        n_posts_per_cluster: Number of representative posts exported per cluster.

    Returns:
        Cluster labels aligned with the reduction rows.
    """

    if backend == "cuml":
        labels, metadata = _run_cuml_hdbscan(reduction.vectors, min_cluster_size)
    elif backend == "sklearn":
        labels, metadata = _run_cpu_hdbscan(reduction.vectors, min_cluster_size)
    else:
        raise ValueError(f"Unsupported density clustering backend: {backend}")

    return ClusterResult(
        method="hdbscan",
        space_name=reduction.name,
        row_indices=reduction.row_indices.copy(),
        labels=labels.astype(np.int32, copy=False),
        backend=backend,
        metadata={
            **metadata,
            "min_cluster_size": int(min_cluster_size),
            "space_name": reduction.name,
            "n_posts_per_cluster": n_posts_per_cluster,
        },
    )


def _run_cuml_kmeans(
    vectors: np.ndarray,
    n_clusters: int,
    random_state: int,
    n_init: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """
    Run KMeans with cuML.
    """

    try:
        from cuml.cluster import KMeans
    except ImportError as exc:
        raise ImportError(
            "cuML KMeans is unavailable. Install the CUDA dependencies or use "
            "backend='sklearn'."
        ) from exc

    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init,
    )

    labels = model.fit_predict(vectors.astype(np.float32, copy=False))

    return _to_numpy(labels), {
        "inertia": _safe_float(getattr(model, "inertia_", None)),
    }


def _run_sklearn_kmeans(
    vectors: np.ndarray,
    n_clusters: int,
    random_state: int,
    n_init: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """
    Run KMeans with scikit-learn.
    """

    from sklearn.cluster import KMeans

    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init,
    )

    labels = model.fit_predict(vectors)

    return labels.astype(np.int32, copy=False), {
        "inertia": float(model.inertia_),
    }


def _run_cuml_dbscan(
    vectors: np.ndarray,
    eps: float,
    min_samples: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """
    Run DBSCAN with cuML.
    """

    try:
        from cuml.cluster import DBSCAN
    except ImportError as exc:
        raise ImportError(
            "cuML DBSCAN is unavailable. Install the CUDA dependencies or disable "
            "density clustering."
        ) from exc

    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(vectors.astype(np.float32, copy=False))

    return _to_numpy(labels), {
        "n_noise": int(np.sum(_to_numpy(labels) == -1)),
    }


def _run_sklearn_dbscan(
    vectors: np.ndarray,
    eps: float,
    min_samples: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """
    Run DBSCAN with scikit-learn.
    """

    from sklearn.cluster import DBSCAN

    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(vectors)

    return labels.astype(np.int32, copy=False), {
        "n_noise": int(np.sum(labels == -1)),
    }


def _run_cuml_hdbscan(
    vectors: np.ndarray,
    min_cluster_size: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """
    Run HDBSCAN with cuML.
    """

    try:
        from cuml.cluster import HDBSCAN
    except ImportError as exc:
        raise ImportError(
            "cuML HDBSCAN is unavailable. Install the CUDA dependencies or disable "
            "density clustering."
        ) from exc

    model = HDBSCAN(min_cluster_size=min_cluster_size)
    labels = model.fit_predict(vectors.astype(np.float32, copy=False))

    return _to_numpy(labels), {
        "n_noise": int(np.sum(_to_numpy(labels) == -1)),
    }


def _run_cpu_hdbscan(
    vectors: np.ndarray,
    min_cluster_size: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """
    Run HDBSCAN with the CPU hdbscan package.
    """

    try:
        import hdbscan
    except ImportError as exc:
        raise ImportError(
            "CPU HDBSCAN is unavailable. Install 'hdbscan' or disable density "
            "clustering."
        ) from exc

    model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    labels = model.fit_predict(vectors)

    return labels.astype(np.int32, copy=False), {
        "n_noise": int(np.sum(labels == -1)),
    }


def _select_spaces(
    reductions: dict[str, ReductionResult],
    space_names: Iterable[str] | None,
) -> dict[str, ReductionResult]:
    """
    Select embedding spaces by name.
    """

    if space_names is None:
        return reductions

    selected: dict[str, ReductionResult] = {}

    for name in space_names:
        if name not in reductions:
            available = ", ".join(sorted(reductions))
            raise ValueError(
                f"Unknown reduction space: {name}. Available spaces: {available}"
            )

        selected[name] = reductions[name]

    return selected


def _to_numpy(value) -> np.ndarray:
    """
    Convert common GPU array objects to NumPy.
    """

    if hasattr(value, "get"):
        return value.get()

    if hasattr(value, "to_numpy"):
        return value.to_numpy()

    return np.asarray(value)


def _safe_float(value) -> float | None:
    """
    Convert a numeric value to float when available.
    """

    if value is None:
        return None

    try:
        return float(value)
    except TypeError:
        return float(_to_numpy(value))