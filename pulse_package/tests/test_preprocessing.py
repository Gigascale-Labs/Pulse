import numpy as np
import polars as pl

from pulse.contracts import PreparedPosts
from pulse.preprocessing import (
    filter_by_min_words,
    remove_spam_posts,
    remove_stopwords,
)


def test_remove_spam_posts_filters_matching_text() -> None:
    prepared = PreparedPosts(
        posts=pl.DataFrame(
            {
                "text": [
                    "normal post",
                    '{"op":"mint"} spam post',
                ]
            }
        ),
        texts=[
            "normal post",
            '{"op":"mint"} spam post',
        ],
        row_indices=np.array([0, 1]),
    )

    result = remove_spam_posts(
        prepared,
        spam_patterns=[r'"op"\s*:\s*"mint"'],
    )

    assert len(result.posts) == 1
    assert result.texts == ["normal post"]
    assert result.row_indices.tolist() == [0]


def test_filter_by_min_words_removes_short_texts() -> None:
    prepared = PreparedPosts(
        posts=pl.DataFrame(
            {
                "text": [
                    "short",
                    "this text has enough words",
                ]
            }
        ),
        texts=[
            "short",
            "this text has enough words",
        ],
        row_indices=np.array([0, 1]),
    )

    result = filter_by_min_words(prepared, min_words=3)

    assert len(result.posts) == 1
    assert result.texts == ["this text has enough words"]
    assert result.row_indices.tolist() == [1]


def test_remove_stopwords_preserves_alignment() -> None:
    prepared = PreparedPosts(
        posts=pl.DataFrame({"text": ["the quick brown fox"]}),
        texts=["the quick brown fox"],
        row_indices=np.array([0]),
    )

    result = remove_stopwords(prepared)

    assert len(result.posts) == 1
    assert len(result.texts) == 1
    assert result.row_indices.tolist() == [0]
    assert "quick" in result.texts[0]