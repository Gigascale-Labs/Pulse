from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import polars as pl

from pulse.config import PreprocessingConfig
from pulse.contracts import PreparedPosts


_SOURCE_ROW_COLUMN = "_source_row_index"


def preprocess_posts(
    prepared: PreparedPosts,
    config: PreprocessingConfig,
    tokenizer_name: str | None = None,
) -> PreparedPosts:
    """
    Apply text preprocessing and return prepared records for embedding.

    Args:
        prepared: Loaded posts before preprocessing.
        config: Text preprocessing settings.
        tokenizer_name: Optional tokenizer name used for chunking.

    Returns:
        Prepared posts after filtering, cleaning, and chunking.
    """

    result = prepared

    result = remove_spam_posts(result, config.spam_patterns)

    if config.lang_filter:
        result = filter_language(result, config.lang_filter)

    if config.remove_stopwords:
        result = remove_stopwords(result)

    if config.min_words_after_stopwords > 0:
        result = filter_by_min_words(result, config.min_words_after_stopwords)

    result = chunk_posts(
        result,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        tokenizer_name=tokenizer_name,
    )

    return PreparedPosts(
        posts=result.posts,
        texts=result.texts,
        row_indices=result.row_indices,
        text_column=result.text_column,
        metadata={
            **prepared.metadata,
            "preprocessing": {
                "spam_patterns": config.spam_patterns,
                "lang_filter": config.lang_filter,
                "remove_stopwords": config.remove_stopwords,
                "min_words_after_stopwords": config.min_words_after_stopwords,
                "chunk_size": config.chunk_size,
                "chunk_overlap": config.chunk_overlap,
                "n_posts_after_preprocessing": len(result.posts),
            },
        },
    )


def remove_spam_posts(
    prepared: PreparedPosts,
    spam_patterns: list[str],
) -> PreparedPosts:
    """
    Remove records whose text matches any configured spam pattern.

    Args:
        prepared: Prepared posts before spam filtering.
        spam_patterns: Regular expressions used to identify records to remove.

    Returns:
        Prepared posts after spam filtering.
    """

    if not spam_patterns:
        return prepared

    combined_pattern = "|".join(f"(?:{pattern})" for pattern in spam_patterns)

    keep_mask = (
        ~pl.Series(prepared.texts)
        .str.contains(combined_pattern, literal=False)
        .fill_null(False)
    )

    return _filter_prepared(prepared, keep_mask.to_list())


def filter_language(
    prepared: PreparedPosts,
    language: str,
) -> PreparedPosts:
    """
    Keep records detected as the requested language.

    Args:
        prepared: Prepared posts before language filtering.
        language: Language code to keep, such as ``en``.

    Returns:
        Prepared posts after language filtering.
    """

    keep = [_detect_language(text) == language for text in prepared.texts]
    return _filter_prepared(prepared, keep)


def remove_stopwords(prepared: PreparedPosts) -> PreparedPosts:
    """
    Remove common English stopwords from the text column.

    Args:
        prepared: Prepared posts before stopword removal.

    Returns:
        Prepared posts with updated text values.
    """

    stopwords = _load_stopwords()
    cleaned_texts = [
        _remove_stopwords_from_text(text, stopwords)
        for text in prepared.texts
    ]

    posts = prepared.posts.with_columns(
        pl.Series(prepared.text_column, cleaned_texts)
    )

    return PreparedPosts(
        posts=posts,
        texts=cleaned_texts,
        row_indices=prepared.row_indices,
        text_column=prepared.text_column,
        metadata={
            **prepared.metadata,
            "stopwords_removed": True,
        },
    )


def filter_by_min_words(
    prepared: PreparedPosts,
    min_words: int,
) -> PreparedPosts:
    """
    Keep records with at least the configured number of words.

    Args:
        prepared: Prepared posts before word-count filtering.
        min_words: Minimum number of whitespace-separated words required.

    Returns:
        Prepared posts after word-count filtering.
    """

    keep = [len(text.split()) >= min_words for text in prepared.texts]
    return _filter_prepared(prepared, keep)


def chunk_posts(
    prepared: PreparedPosts,
    chunk_size: int,
    chunk_overlap: int,
    tokenizer_name: str | None = None,
) -> PreparedPosts:
    """
    Split long text records into tokenizer-based chunks.

    Args:
        prepared: Prepared posts before chunking.
        chunk_size: Maximum number of tokens per chunk.
        chunk_overlap: Number of overlapping tokens between adjacent chunks.
        tokenizer_name: Optional tokenizer name. If omitted, a default tokenizer
            is used.

    Returns:
        Prepared posts where each row corresponds to one embedding text.
    """

    if len(prepared.posts) == 0:
        return prepared

    tokenizer = _load_tokenizer(tokenizer_name)

    chunk_texts: list[str] = []
    source_positions: list[int] = []

    for position, text in enumerate(prepared.texts):
        chunks = _chunk_text(text, tokenizer, chunk_size, chunk_overlap)

        for chunk in chunks:
            chunk_texts.append(chunk)
            source_positions.append(position)

    if not chunk_texts:
        return PreparedPosts(
            posts=prepared.posts.head(0),
            texts=[],
            row_indices=np.array([], dtype=np.int64),
            text_column=prepared.text_column,
            metadata={
                **prepared.metadata,
                "chunking": {
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "n_chunks": 0,
                },
            },
        )

    posts = prepared.posts[source_positions].with_columns(
        [
            pl.Series(prepared.text_column, chunk_texts),
            pl.Series(_SOURCE_ROW_COLUMN, prepared.row_indices[source_positions]),
        ]
    )

    row_indices = np.arange(len(posts), dtype=np.int64)

    return PreparedPosts(
        posts=posts,
        texts=chunk_texts,
        row_indices=row_indices,
        text_column=prepared.text_column,
        metadata={
            **prepared.metadata,
            "chunking": {
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "n_chunks": len(chunk_texts),
            },
        },
    )


def _filter_prepared(
    prepared: PreparedPosts,
    keep: Iterable[bool],
) -> PreparedPosts:
    """
    Filter prepared posts while preserving alignment across fields.
    """

    keep_values = list(keep)

    if len(keep_values) != len(prepared.posts):
        raise ValueError(
            "Filter mask length must match the number of posts. "
            f"Got {len(keep_values)} mask values and {len(prepared.posts)} posts."
        )

    posts = prepared.posts.filter(pl.Series(keep_values))
    texts = [
        text
        for text, keep_value in zip(prepared.texts, keep_values, strict=True)
        if keep_value
    ]
    row_indices = prepared.row_indices[np.asarray(keep_values, dtype=bool)]

    return PreparedPosts(
        posts=posts,
        texts=texts,
        row_indices=row_indices,
        text_column=prepared.text_column,
        metadata=prepared.metadata,
    )


def _detect_language(text: str) -> str | None:
    """
    Detect the language of a text value.
    """

    if not text.strip():
        return None

    try:
        from fast_langdetect import detect

        result = detect(text)
    except Exception:
        return None

    if isinstance(result, dict):
        return result.get("lang")

    return None


def _load_stopwords() -> set[str]:
    """
    Load English stopwords with a small fallback set.
    """

    try:
        from nltk.corpus import stopwords

        return set(stopwords.words("english"))
    except Exception:
        return {
            "a",
            "an",
            "and",
            "are",
            "as",
            "at",
            "be",
            "by",
            "for",
            "from",
            "has",
            "he",
            "in",
            "is",
            "it",
            "its",
            "of",
            "on",
            "that",
            "the",
            "to",
            "was",
            "were",
            "will",
            "with",
        }


def _remove_stopwords_from_text(text: str, stopwords: set[str]) -> str:
    """
    Remove stopwords from one text value.
    """

    words = text.split()
    kept_words = [
        word
        for word in words
        if _normalize_word(word) not in stopwords
    ]
    return " ".join(kept_words)


def _normalize_word(word: str) -> str:
    """
    Normalize a word before stopword comparison.
    """

    return re.sub(r"[^a-zA-Z']", "", word).lower()


def _load_tokenizer(tokenizer_name: str | None):
    """
    Load a tokenizer for chunking.
    """

    from transformers import AutoTokenizer

    name = tokenizer_name or "sentence-transformers/all-MiniLM-L6-v2"
    return AutoTokenizer.from_pretrained(name)


def _chunk_text(
    text: str,
    tokenizer,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    """
    Split one text value into decoded token chunks.
    """

    token_ids = tokenizer.encode(text, add_special_tokens=False)

    if not token_ids:
        return []

    if len(token_ids) <= chunk_size:
        return [text]

    step = chunk_size - chunk_overlap
    chunks: list[str] = []

    for start in range(0, len(token_ids), step):
        end = start + chunk_size
        chunk_ids = token_ids[start:end]

        if not chunk_ids:
            continue

        chunk = tokenizer.decode(chunk_ids, skip_special_tokens=True).strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(token_ids):
            break

    return chunks