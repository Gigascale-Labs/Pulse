"""Sample posts and embeddings together while preserving alignment."""

from __future__ import annotations

import numpy as np
import polars as pl

from pulse.config import SamplingConfig
from pulse.contracts import EmbeddingResult, PreparedPosts, SampledPosts
from pulse.embeddings import validate_embedding_alignment


_SAMPLE_POSITION_COLUMN = "_sample_position"


def sample_posts_and_embeddings(
    prepared: PreparedPosts,
    embeddings: EmbeddingResult,
    config: SamplingConfig,
) -> SampledPosts:
    """
    Sample prepared records and aligned embeddings.

    Args:
        prepared: Prepared text records.
        embeddings: Embedding vectors aligned with the prepared records.
        config: Sampling settings.

    Returns:
        Sampled posts and embedding vectors.
    """

    validate_embedding_alignment(prepared, embeddings)

    n_available = len(prepared.posts)
    n_sample = min(config.sample_size, n_available)

    if n_available == 0:
        raise ValueError("Cannot sample from an empty prepared dataset.")

    if n_sample == n_available:
        return SampledPosts(
            posts=prepared.posts,
            texts=prepared.texts,
            row_indices=prepared.row_indices.copy(),
            vectors=embeddings.vectors.copy(),
            metadata={
                "sample_size": n_sample,
                "n_available": n_available,
                "seed": config.seed,
                "sampled": False,
            },
        )

    posts_with_position = prepared.posts.with_columns(
        pl.Series(_SAMPLE_POSITION_COLUMN, np.arange(n_available, dtype=np.int64))
    )

    sampled_posts = posts_with_position.sample(
        n=n_sample,
        seed=config.seed,
        shuffle=True,
    )

    sample_positions = sampled_posts.get_column(_SAMPLE_POSITION_COLUMN).to_numpy()
    sampled_posts = sampled_posts.drop(_SAMPLE_POSITION_COLUMN)

    sample_positions = np.asarray(sample_positions, dtype=np.int64)

    sampled_texts = [
        prepared.texts[position]
        for position in sample_positions
    ]

    sampled_row_indices = prepared.row_indices[sample_positions]
    sampled_vectors = embeddings.vectors[sample_positions]

    return SampledPosts(
        posts=sampled_posts,
        texts=sampled_texts,
        row_indices=sampled_row_indices,
        vectors=sampled_vectors,
        metadata={
            "sample_size": int(n_sample),
            "n_available": int(n_available),
            "seed": config.seed,
            "sampled": True,
        },
    )