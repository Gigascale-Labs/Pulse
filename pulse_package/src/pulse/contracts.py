from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl


@dataclass
class PreparedPosts:
    """
    Prepared text records used as input for embedding.

    Attributes:
        posts: Polars DataFrame containing the prepared records.
        texts: Text values extracted from the configured text column.
        row_indices: Stable row positions for alignment with embeddings and outputs.
        text_column: Name of the text column in the posts DataFrame.
        metadata: Additional information about data loading and preprocessing.
    """

    posts: pl.DataFrame
    texts: list[str]
    row_indices: np.ndarray
    text_column: str = "text"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.text_column not in self.posts.columns:
            raise ValueError(f"Missing text column: {self.text_column}")

        if len(self.posts) != len(self.texts):
            raise ValueError(
                "posts and texts must have the same length. "
                f"Got {len(self.posts)} posts and {len(self.texts)} texts."
            )

        if len(self.row_indices) != len(self.posts):
            raise ValueError(
                "row_indices and posts must have the same length. "
                f"Got {len(self.row_indices)} row indices and {len(self.posts)} posts."
            )

        if self.row_indices.ndim != 1:
            raise ValueError(
                f"row_indices must be a 1D array. Got shape {self.row_indices.shape}."
            )


@dataclass
class EmbeddingResult:
    """
    Embedding vectors generated for prepared text records.

    Attributes:
        row_indices: Row positions aligned with the embedding matrix.
        vectors: Two-dimensional embedding matrix with shape
            ``(n_records, embedding_dim)``.
        model_name: Name of the embedding model used to generate the vectors.
        data_hash: Hash identifying the input/configuration used for this result.
        metadata: Additional embedding metadata such as batch size and vector dimension.
    """

    row_indices: np.ndarray
    vectors: np.ndarray
    model_name: str
    data_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.vectors.ndim != 2:
            raise ValueError(
                f"vectors must be a 2D array. Got shape {self.vectors.shape}."
            )

        if len(self.row_indices) != self.vectors.shape[0]:
            raise ValueError(
                "row_indices and vectors must have the same number of rows. "
                f"Got {len(self.row_indices)} row indices and "
                f"{self.vectors.shape[0]} vectors."
            )

        if self.row_indices.ndim != 1:
            raise ValueError(
                f"row_indices must be a 1D array. Got shape {self.row_indices.shape}."
            )

        if self.vectors.dtype != np.float32:
            self.vectors = self.vectors.astype(np.float32)


@dataclass
class SampledPosts:
    """
    Sampled records and embeddings used for downstream modeling.

    Attributes:
        posts: Sampled Polars DataFrame.
        texts: Text values for the sampled records.
        row_indices: Original row positions for the sampled records.
        vectors: Embedding vectors aligned with the sampled records.
        metadata: Additional information about the sampling process.
    """

    posts: pl.DataFrame
    texts: list[str]
    row_indices: np.ndarray
    vectors: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n_records = len(self.posts)

        if len(self.texts) != n_records:
            raise ValueError(
                "texts and posts must have the same length. "
                f"Got {len(self.texts)} texts and {n_records} posts."
            )

        if len(self.row_indices) != n_records:
            raise ValueError(
                "row_indices and posts must have the same length. "
                f"Got {len(self.row_indices)} row indices and {n_records} posts."
            )

        if self.row_indices.ndim != 1:
            raise ValueError(
                f"row_indices must be a 1D array. Got shape {self.row_indices.shape}."
            )

        if self.vectors.ndim != 2:
            raise ValueError(
                f"vectors must be a 2D array. Got shape {self.vectors.shape}."
            )

        if self.vectors.shape[0] != n_records:
            raise ValueError(
                "vectors and posts must have the same number of rows. "
                f"Got {self.vectors.shape[0]} vectors and {n_records} posts."
            )

        if self.vectors.dtype != np.float32:
            self.vectors = self.vectors.astype(np.float32)


@dataclass
class ReductionResult:
    """
    Embedding vectors after dimensionality reduction.

    Attributes:
        name: Name of the reduced embedding space.
        row_indices: Row positions aligned with the reduced vectors.
        vectors: Reduced embedding matrix with shape ``(n_records, n_dimensions)``.
        metadata: Additional information about the reduction method.
    """

    name: str
    row_indices: np.ndarray
    vectors: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.vectors.ndim != 2:
            raise ValueError(
                f"vectors must be a 2D array. Got shape {self.vectors.shape}."
            )

        if len(self.row_indices) != self.vectors.shape[0]:
            raise ValueError(
                "row_indices and vectors must have the same number of rows. "
                f"Got {len(self.row_indices)} row indices and "
                f"{self.vectors.shape[0]} vectors."
            )

        if self.row_indices.ndim != 1:
            raise ValueError(
                f"row_indices must be a 1D array. Got shape {self.row_indices.shape}."
            )

        if self.vectors.dtype != np.float32:
            self.vectors = self.vectors.astype(np.float32)


@dataclass
class ClusterResult:
    """
    Cluster labels produced for a specific embedding space.

    Attributes:
        method: Clustering method name, such as ``kmeans`` or ``dbscan``.
        space_name: Name of the embedding space used for clustering.
        row_indices: Row positions aligned with the cluster labels.
        labels: One-dimensional array of cluster labels.
        backend: Implementation backend, such as ``sklearn`` or ``cuml``.
        metadata: Additional information about clustering parameters and metrics.
    """

    method: str
    space_name: str
    row_indices: np.ndarray
    labels: np.ndarray
    backend: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.labels.ndim != 1:
            raise ValueError(
                f"labels must be a 1D array. Got shape {self.labels.shape}."
            )

        if len(self.row_indices) != len(self.labels):
            raise ValueError(
                "row_indices and labels must have the same length. "
                f"Got {len(self.row_indices)} row indices and "
                f"{len(self.labels)} labels."
            )

        if self.row_indices.ndim != 1:
            raise ValueError(
                f"row_indices must be a 1D array. Got shape {self.row_indices.shape}."
            )

        if self.labels.dtype.kind not in {"i", "u"}:
            self.labels = self.labels.astype(np.int32)


@dataclass
class PersonaDumpResult:
    """
    Files written for one exported persona clustering run.

    Attributes:
        run_dir: Directory containing all files for the run.
        metadata_path: Path to the run metadata file.
        clusters_arrow_path: Path to the Arrow file containing cluster assignments.
        post_files: Human-readable text files containing sampled posts per cluster.
        metadata: Additional export metadata.
    """

    run_dir: Path
    metadata_path: Path
    clusters_arrow_path: Path
    post_files: list[Path]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    """
    Full result returned by the pipeline.

    Attributes:
        prepared: Prepared posts used as pipeline input.
        embeddings: Embedding vectors generated or loaded by the pipeline.
        sampled: Optional sampled subset used for modeling.
        reductions: Reduced embedding spaces keyed by name.
        clusters: Cluster results produced by the pipeline.
        dumps: Exported persona dump results.
        metadata: Additional run-level metadata.
    """

    prepared: PreparedPosts
    embeddings: EmbeddingResult
    sampled: SampledPosts | None
    reductions: dict[str, ReductionResult]
    clusters: list[ClusterResult]
    dumps: list[PersonaDumpResult]
    metadata: dict[str, Any] = field(default_factory=dict)