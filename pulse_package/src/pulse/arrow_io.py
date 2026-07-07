from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc

from pulse.contracts import ClusterResult, EmbeddingResult


def save_embeddings_arrow(
    embeddings: EmbeddingResult,
    path: str | Path,
) -> Path:
    """
    Write embeddings to an Arrow IPC file.

    Args:
        embeddings: Embedding result to write.
        path: Output Arrow file path.

    Returns:
        Path to the written file.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    vectors = _ensure_float32_matrix(embeddings.vectors)
    row_indices = _ensure_int64_vector(embeddings.row_indices)

    table = pa.table(
        {
            "row_index": pa.array(row_indices, type=pa.int64()),
            "embedding": pa.FixedSizeListArray.from_arrays(
                pa.array(vectors.reshape(-1), type=pa.float32()),
                vectors.shape[1],
            ),
        }
    )

    _write_ipc_table(table, path)
    return path


def load_embeddings_arrow(
    path: str | Path,
    model_name: str,
    data_hash: str,
    metadata: dict[str, Any] | None = None,
) -> EmbeddingResult:
    """
    Load embeddings from an Arrow IPC file.

    Args:
        path: Input Arrow file path.
        model_name: Name of the embedding model associated with the file.
        data_hash: Hash identifying the data and embedding configuration.
        metadata: Optional metadata to attach to the loaded result.

    Returns:
        Loaded embedding result.
    """

    path = Path(path)
    table = _read_ipc_table(path)

    if "row_index" not in table.column_names:
        raise ValueError(f"Missing row_index column in embedding file: {path}")

    if "embedding" not in table.column_names:
        raise ValueError(f"Missing embedding column in embedding file: {path}")

    row_indices = table.column("row_index").to_numpy(zero_copy_only=False)
    vectors = _fixed_size_list_column_to_numpy(table.column("embedding"))

    return EmbeddingResult(
        row_indices=row_indices.astype(np.int64),
        vectors=vectors.astype(np.float32),
        model_name=model_name,
        data_hash=data_hash,
        metadata={
            **(metadata or {}),
            "path": str(path),
            "n_vectors": int(vectors.shape[0]),
            "vector_dim": int(vectors.shape[1]),
            "loaded_from_cache": True,
        },
    )


def save_clusters_arrow(
    clusters: ClusterResult,
    path: str | Path,
) -> Path:
    """
    Write cluster assignments to an Arrow IPC file.

    Args:
        clusters: Cluster result to write.
        path: Output Arrow file path.

    Returns:
        Path to the written file.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    table = pa.table(
        {
            "row_index": pa.array(
                _ensure_int64_vector(clusters.row_indices),
                type=pa.int64(),
            ),
            "cluster": pa.array(
                _ensure_int32_vector(clusters.labels),
                type=pa.int32(),
            ),
        }
    )

    _write_ipc_table(table, path)
    return path


def load_clusters_arrow(
    path: str | Path,
    method: str,
    space_name: str,
    backend: str,
    metadata: dict[str, Any] | None = None,
) -> ClusterResult:
    """
    Load cluster assignments from an Arrow IPC file.

    Args:
        path: Input Arrow file path.
        method: Clustering method name.
        space_name: Name of the embedding space used for clustering.
        backend: Clustering backend name.
        metadata: Optional metadata to attach to the loaded result.

    Returns:
        Loaded cluster result.
    """

    path = Path(path)
    table = _read_ipc_table(path)

    if "row_index" not in table.column_names:
        raise ValueError(f"Missing row_index column in cluster file: {path}")

    if "cluster" not in table.column_names:
        raise ValueError(f"Missing cluster column in cluster file: {path}")

    row_indices = table.column("row_index").to_numpy(zero_copy_only=False)
    labels = table.column("cluster").to_numpy(zero_copy_only=False)

    return ClusterResult(
        method=method,
        space_name=space_name,
        row_indices=row_indices.astype(np.int64),
        labels=labels.astype(np.int32),
        backend=backend,
        metadata={
            **(metadata or {}),
            "path": str(path),
            "loaded_from_cache": True,
        },
    )


def _write_ipc_table(table: pa.Table, path: Path) -> None:
    """
    Write a PyArrow table using Arrow IPC file format.
    """

    with pa.OSFile(str(path), "wb") as sink:
        with ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)


def _read_ipc_table(path: Path) -> pa.Table:
    """
    Read a PyArrow table from Arrow IPC file format.
    """

    if not path.exists():
        raise FileNotFoundError(f"Arrow file does not exist: {path}")

    with pa.memory_map(str(path), "r") as source:
        with ipc.open_file(source) as reader:
            return reader.read_all()


def _fixed_size_list_column_to_numpy(column: pa.ChunkedArray) -> np.ndarray:
    """
    Convert a fixed-size-list Arrow column to a two-dimensional NumPy array.
    """

    combined = column.combine_chunks()

    if not pa.types.is_fixed_size_list(combined.type):
        raise ValueError(
            "Expected a fixed-size-list Arrow column for embeddings. "
            f"Got {combined.type}."
        )

    list_size = combined.type.list_size
    values = combined.values.to_numpy(zero_copy_only=False)

    return values.reshape((len(combined), list_size))


def _ensure_float32_matrix(values: np.ndarray) -> np.ndarray:
    """
    Validate and convert an array to a two-dimensional float32 matrix.
    """

    array = np.asarray(values)

    if array.ndim != 2:
        raise ValueError(f"Expected a 2D matrix. Got shape {array.shape}.")

    return array.astype(np.float32, copy=False)


def _ensure_int64_vector(values: np.ndarray) -> np.ndarray:
    """
    Validate and convert an array to a one-dimensional int64 vector.
    """

    array = np.asarray(values)

    if array.ndim != 1:
        raise ValueError(f"Expected a 1D vector. Got shape {array.shape}.")

    return array.astype(np.int64, copy=False)


def _ensure_int32_vector(values: np.ndarray) -> np.ndarray:
    """
    Validate and convert an array to a one-dimensional int32 vector.
    """

    array = np.asarray(values)

    if array.ndim != 1:
        raise ValueError(f"Expected a 1D vector. Got shape {array.shape}.")

    return array.astype(np.int32, copy=False)