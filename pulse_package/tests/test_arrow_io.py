from pathlib import Path

import numpy as np

from pulse.arrow_io import (
    load_clusters_arrow,
    load_embeddings_arrow,
    save_clusters_arrow,
    save_embeddings_arrow,
)
from pulse.contracts import ClusterResult, EmbeddingResult


def test_save_and_load_embeddings_arrow(tmp_path: Path) -> None:
    path = tmp_path / "embeddings.arrow"

    embeddings = EmbeddingResult(
        row_indices=np.array([0, 1, 2]),
        vectors=np.array(
            [
                [0.1, 0.2],
                [0.3, 0.4],
                [0.5, 0.6],
            ],
            dtype=np.float32,
        ),
        model_name="test-model",
        data_hash="test-hash",
    )

    save_embeddings_arrow(embeddings, path)

    loaded = load_embeddings_arrow(
        path=path,
        model_name="test-model",
        data_hash="test-hash",
    )

    assert path.exists()
    assert np.array_equal(loaded.row_indices, embeddings.row_indices)
    assert np.allclose(loaded.vectors, embeddings.vectors)
    assert loaded.model_name == "test-model"
    assert loaded.data_hash == "test-hash"


def test_save_and_load_clusters_arrow(tmp_path: Path) -> None:
    path = tmp_path / "clusters.arrow"

    clusters = ClusterResult(
        method="kmeans",
        space_name="pca50",
        row_indices=np.array([0, 1, 2]),
        labels=np.array([1, 1, 2]),
        backend="sklearn",
    )

    save_clusters_arrow(clusters, path)

    loaded = load_clusters_arrow(
        path=path,
        method="kmeans",
        space_name="pca50",
        backend="sklearn",
    )

    assert path.exists()
    assert np.array_equal(loaded.row_indices, clusters.row_indices)
    assert np.array_equal(loaded.labels, clusters.labels)
    assert loaded.method == "kmeans"
    assert loaded.space_name == "pca50"
    assert loaded.backend == "sklearn"