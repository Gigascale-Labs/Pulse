from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def stable_hash(values: dict[str, Any], length: int = 16) -> str:
    """
    Create a stable hash for a dictionary of configuration values.

    The input dictionary is converted to canonical JSON before hashing, so the
    same values produce the same hash across runs.

    Args:
        values: Dictionary containing values that identify a pipeline artifact.
        length: Number of characters to return from the SHA-256 digest.

    Returns:
        Short hexadecimal hash string.
    """

    if length <= 0:
        raise ValueError("length must be greater than zero.")

    normalized = _normalize_for_json(values)
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def embedding_data_hash(values: dict[str, Any]) -> str:
    """
    Create the hash used to identify compatible embedding files.

    Args:
        values: Configuration fields that affect the selected text records or
            generated embeddings.

    Returns:
        Short hexadecimal hash string.
    """

    return stable_hash(values)


def clustering_run_hash(data_hash: str, clustering_metadata: dict[str, Any]) -> str:
    """
    Create the hash used to identify one clustering export run.

    Args:
        data_hash: Hash identifying the embedding input data.
        clustering_metadata: Parameters that identify the clustering run.

    Returns:
        Short hexadecimal hash string.
    """

    return stable_hash(
        {
            "data_hash": data_hash,
            "clustering": clustering_metadata,
        }
    )


def _normalize_for_json(value: Any) -> Any:
    """
    Convert common Python objects into JSON-compatible values.
    """

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {
            str(key): _normalize_for_json(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }

    if isinstance(value, (list, tuple)):
        return [_normalize_for_json(item) for item in value]

    if isinstance(value, set):
        return sorted(_normalize_for_json(item) for item in value)

    return value