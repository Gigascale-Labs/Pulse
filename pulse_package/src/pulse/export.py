'''Write persona export folders containing metadata, cluster assignments, and readable post samples.'''

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from pulse.arrow_io import save_clusters_arrow
from pulse.config import OutputConfig
from pulse.contracts import ClusterResult, PersonaDumpResult, SampledPosts
from pulse.hashing import clustering_run_hash


def export_persona_dumps(
    sampled: SampledPosts,
    clusters: list[ClusterResult],
    config: OutputConfig,
    data_hash: str,
) -> list[PersonaDumpResult]:
    """
    Write persona exports for clustering results.

    Args:
        sampled: Sampled posts and texts used for clustering.
        clusters: Cluster results to export.
        config: Output settings.
        data_hash: Hash identifying the input data and embedding configuration.

    Returns:
        Export results for each clustering run.
    """

    if not clusters:
        return []

    config.persona_dump_dir.mkdir(parents=True, exist_ok=True)

    return [
        export_single_persona_dump(
            sampled=sampled,
            clusters=cluster_result,
            output_dir=config.persona_dump_dir,
            data_hash=data_hash,
        )
        for cluster_result in clusters
    ]


def export_single_persona_dump(
    sampled: SampledPosts,
    clusters: ClusterResult,
    output_dir: Path,
    data_hash: str,
) -> PersonaDumpResult:
    """
    Write one persona export for one clustering result.

    Args:
        sampled: Sampled posts and texts used for clustering.
        clusters: Cluster labels aligned with the sampled records.
        output_dir: Root output directory.
        data_hash: Hash identifying the input data and embedding configuration.

    Returns:
        Export result with paths to written files.
    """

    _validate_cluster_alignment(sampled, clusters)

    metadata = _build_export_metadata(
        sampled=sampled,
        clusters=clusters,
        data_hash=data_hash,
    )

    run_hash = clustering_run_hash(
        data_hash=data_hash,
        clustering_metadata={
            "method": clusters.method,
            "space_name": clusters.space_name,
            "backend": clusters.backend,
            "metadata": clusters.metadata,
        },
    )

    run_dir = output_dir / _build_run_name(clusters, run_hash)
    run_dir.mkdir(parents=True, exist_ok=True)

    clusters_arrow_path = save_clusters_arrow(
        clusters=clusters,
        path=run_dir / "clusters.arrow",
    )

    metadata_path = run_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(_json_safe(metadata), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    post_files = _write_cluster_post_files(
        sampled=sampled,
        clusters=clusters,
        run_dir=run_dir,
    )

    return PersonaDumpResult(
        run_dir=run_dir,
        metadata_path=metadata_path,
        clusters_arrow_path=clusters_arrow_path,
        post_files=post_files,
        metadata=metadata,
    )


def _write_cluster_post_files(
    sampled: SampledPosts,
    clusters: ClusterResult,
    run_dir: Path,
) -> list[Path]:
    """
    Write readable post samples for each cluster label.
    """

    labels = clusters.labels
    unique_labels = sorted(int(label) for label in np.unique(labels))
    n_posts_per_cluster = int(clusters.metadata.get("n_posts_per_cluster", 500))

    post_files: list[Path] = []

    for label in unique_labels:
        label_positions = np.flatnonzero(labels == label)

        if len(label_positions) == 0:
            continue

        selected_positions = label_positions[:n_posts_per_cluster]
        file_path = run_dir / _cluster_post_filename(label)

        file_path.write_text(
            _format_cluster_posts(
                label=label,
                positions=selected_positions,
                sampled=sampled,
            ),
            encoding="utf-8",
        )

        post_files.append(file_path)

    return post_files


def _format_cluster_posts(
    label: int,
    positions: np.ndarray,
    sampled: SampledPosts,
) -> str:
    """
    Format post samples for one cluster label.
    """

    lines: list[str] = [
        f"Cluster: {label}",
        f"Number of exported posts: {len(positions)}",
        "",
    ]

    for export_index, position in enumerate(positions, start=1):
        row_index = int(sampled.row_indices[position])
        text = sampled.texts[position]

        lines.extend(
            [
                "=" * 80,
                f"Post {export_index}",
                f"row_index: {row_index}",
                "",
                text.strip(),
                "",
            ]
        )

    return "\n".join(lines)


def _build_export_metadata(
    sampled: SampledPosts,
    clusters: ClusterResult,
    data_hash: str,
) -> dict[str, Any]:
    """
    Build metadata for one persona export.
    """

    label_summary = _cluster_label_summary(clusters.labels)

    return {
        "data_hash": data_hash,
        "method": clusters.method,
        "space_name": clusters.space_name,
        "backend": clusters.backend,
        "n_records": int(len(sampled.posts)),
        "n_clusters_including_noise": int(len(label_summary)),
        "n_noise": int(label_summary.get("-1", 0)),
        "cluster_label_summary": label_summary,
        "clustering": clusters.metadata,
        "sample": sampled.metadata,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }


def _cluster_label_summary(labels: np.ndarray) -> dict[str, int]:
    """
    Count records per cluster label.
    """

    unique_labels, counts = np.unique(labels, return_counts=True)

    return {
        str(int(label)): int(count)
        for label, count in zip(unique_labels, counts, strict=True)
    }


def _validate_cluster_alignment(
    sampled: SampledPosts,
    clusters: ClusterResult,
) -> None:
    """
    Validate that cluster labels align with sampled records.
    """

    if len(sampled.posts) != len(clusters.labels):
        raise ValueError(
            "Sampled posts and cluster labels must have the same length. "
            f"Got {len(sampled.posts)} posts and {len(clusters.labels)} labels."
        )

    if not np.array_equal(sampled.row_indices, clusters.row_indices):
        raise ValueError(
            "Sampled row indices do not match cluster row indices. "
            "Cluster labels must be aligned with the sampled records."
        )


def _build_run_name(
    clusters: ClusterResult,
    run_hash: str,
) -> str:
    """
    Build a readable directory name for one export run.
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    method = _safe_name(clusters.method)
    space = _safe_name(clusters.space_name)

    if clusters.method == "kmeans":
        cluster_part = f"{clusters.metadata.get('n_clusters', 'unknown')}clusters"
    elif clusters.method == "dbscan":
        cluster_part = f"eps{clusters.metadata.get('eps', 'unknown')}"
    elif clusters.method == "hdbscan":
        cluster_part = f"min{clusters.metadata.get('min_cluster_size', 'unknown')}"
    else:
        cluster_part = "clusters"

    return f"{timestamp}_{space}_{method}_{cluster_part}_{run_hash}"


def _cluster_post_filename(label: int) -> str:
    """
    Build a filename for cluster post samples.
    """

    if label == -1:
        return "cluster_noise_posts.txt"

    return f"cluster_{label}_posts.txt"


def _safe_name(value: Any) -> str:
    """
    Convert a value into a filesystem-safe name.
    """

    text = str(value)
    safe = text.replace("/", "_").replace(" ", "_")
    return "".join(char for char in safe if char.isalnum() or char in {"_", "-", "."})


def _json_safe(value: Any) -> Any:
    """
    Convert common Python, NumPy, Polars, and Path values to JSON-safe values.
    """

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, pl.DataFrame):
        return {
            "type": "polars.DataFrame",
            "shape": value.shape,
            "columns": value.columns,
        }

    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [_json_safe(item) for item in value]

    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]

    return value