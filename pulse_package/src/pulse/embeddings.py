from __future__ import annotations

from pathlib import Path

import numpy as np

from persona_pipeline.arrow_io import load_embeddings_arrow, save_embeddings_arrow
from persona_pipeline.config import EmbeddingConfig
from persona_pipeline.contracts import EmbeddingResult, PreparedPosts


def build_embeddings(
    prepared: PreparedPosts,
    config: EmbeddingConfig,
    data_hash: str,
    device: str | None = None,
) -> EmbeddingResult:
    """
    Generate embeddings or load them from an Arrow cache file.

    Args:
        prepared: Prepared text records to embed.
        config: Embedding generation and cache settings.
        data_hash: Hash identifying the data and embedding configuration.
        device: Optional compute device passed to SentenceTransformer.

    Returns:
        Embedding vectors aligned with prepared records.
    """

    if config.generate:
        embeddings = generate_embeddings(
            prepared=prepared,
            config=config,
            data_hash=data_hash,
            device=device,
        )

        cache_path = get_embedding_cache_path(config, data_hash)
        save_embeddings_arrow(embeddings, cache_path)

        embeddings.metadata.update(
            {
                "cache_path": str(cache_path),
                "saved_to_cache": True,
            }
        )

        return embeddings

    if config.load_path is None:
        raise ValueError(
            "embedding.load_path must be provided when embedding.generate is false."
        )

    return load_embeddings_arrow(
        path=config.load_path,
        model_name=config.model_name,
        data_hash=data_hash,
        metadata={
            "model_name": config.model_name,
            "expected_data_hash": data_hash,
        },
    )


def generate_embeddings(
    prepared: PreparedPosts,
    config: EmbeddingConfig,
    data_hash: str,
    device: str | None = None,
) -> EmbeddingResult:
    """
    Generate embeddings for prepared text records.

    Args:
        prepared: Prepared text records to embed.
        config: Embedding generation settings.
        data_hash: Hash identifying the data and embedding configuration.
        device: Optional compute device passed to SentenceTransformer.

    Returns:
        Generated embedding vectors.
    """

    if not prepared.texts:
        raise ValueError("Cannot generate embeddings for an empty text collection.")

    from sentence_transformers import SentenceTransformer

    model_kwargs = {}
    if device:
        model_kwargs["device"] = _normalize_sentence_transformer_device(device)

    model = SentenceTransformer(config.model_name, **model_kwargs)

    vectors = model.encode(
        prepared.texts,
        batch_size=config.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=config.normalize_embeddings,
    )

    vectors = np.asarray(vectors, dtype=np.float32)

    return EmbeddingResult(
        row_indices=prepared.row_indices.copy(),
        vectors=vectors,
        model_name=config.model_name,
        data_hash=data_hash,
        metadata={
            "model_name": config.model_name,
            "batch_size": config.batch_size,
            "normalize_embeddings": config.normalize_embeddings,
            "n_vectors": int(vectors.shape[0]),
            "vector_dim": int(vectors.shape[1]),
            "generated": True,
            "device": device,
        },
    )


def get_embedding_cache_path(
    config: EmbeddingConfig,
    data_hash: str,
) -> Path:
    """
    Return the default Arrow cache path for an embedding result.

    Args:
        config: Embedding cache settings.
        data_hash: Hash identifying the data and embedding configuration.

    Returns:
        Path for the embedding Arrow file.
    """

    model_name = _safe_name(config.model_name)
    filename = f"embeddings_{model_name}_{data_hash}.arrow"
    return config.cache_dir / filename


def validate_embedding_alignment(
    prepared: PreparedPosts,
    embeddings: EmbeddingResult,
) -> None:
    """
    Validate that embeddings are aligned with prepared records.

    Args:
        prepared: Prepared text records.
        embeddings: Embedding vectors to validate.

    Raises:
        ValueError: If row indices or row counts do not match.
    """

    if len(prepared.posts) != embeddings.vectors.shape[0]:
        raise ValueError(
            "Prepared posts and embeddings must have the same number of rows. "
            f"Got {len(prepared.posts)} posts and "
            f"{embeddings.vectors.shape[0]} embeddings."
        )

    if not np.array_equal(prepared.row_indices, embeddings.row_indices):
        raise ValueError(
            "Prepared row indices do not match embedding row indices. "
            "The embedding file may not belong to the current prepared dataset."
        )


def _safe_name(value: str) -> str:
    """
    Convert a model name into a filesystem-safe string.
    """

    safe = value.replace("/", "__").replace(" ", "_")
    return "".join(char for char in safe if char.isalnum() or char in {"_", "-", "."})


def _normalize_sentence_transformer_device(device: str) -> str:
    """
    Convert a configured device string into a SentenceTransformer device string.
    """

    if "," in device:
        return device.split(",", maxsplit=1)[0].strip()

    return device.strip()