import numpy as np
import polars as pl

from pulse.config import SamplingConfig
from pulse.contracts import EmbeddingResult, PreparedPosts
from pulse.sampling import sample_posts_and_embeddings


def test_sampling_preserves_alignment() -> None:
    posts = pl.DataFrame(
        {
            "text": ["a", "b", "c", "d"],
            "value": [10, 20, 30, 40],
        }
    )

    prepared = PreparedPosts(
        posts=posts,
        texts=["a", "b", "c", "d"],
        row_indices=np.array([0, 1, 2, 3]),
    )

    vectors = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
        ],
        dtype=np.float32,
    )

    embeddings = EmbeddingResult(
        row_indices=np.array([0, 1, 2, 3]),
        vectors=vectors,
        model_name="test-model",
        data_hash="test-hash",
    )

    sampled = sample_posts_and_embeddings(
        prepared=prepared,
        embeddings=embeddings,
        config=SamplingConfig(sample_size=2, seed=42),
    )

    assert len(sampled.posts) == 2
    assert len(sampled.texts) == 2
    assert len(sampled.row_indices) == 2
    assert sampled.vectors.shape == (2, 2)

    for text, row_index, vector in zip(
        sampled.texts,
        sampled.row_indices,
        sampled.vectors,
        strict=True,
    ):
        expected_position = int(row_index)
        assert text == prepared.texts[expected_position]
        assert np.array_equal(vector, vectors[expected_position])


def test_sampling_returns_all_rows_when_sample_size_is_large() -> None:
    posts = pl.DataFrame({"text": ["a", "b"]})

    prepared = PreparedPosts(
        posts=posts,
        texts=["a", "b"],
        row_indices=np.array([0, 1]),
    )

    embeddings = EmbeddingResult(
        row_indices=np.array([0, 1]),
        vectors=np.array([[0.0], [1.0]], dtype=np.float32),
        model_name="test-model",
        data_hash="test-hash",
    )

    sampled = sample_posts_and_embeddings(
        prepared=prepared,
        embeddings=embeddings,
        config=SamplingConfig(sample_size=10, seed=42),
    )

    assert len(sampled.posts) == 2
    assert sampled.metadata["sampled"] is False