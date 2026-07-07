from __future__ import annotations

from datetime import date
from typing import Any

import duckdb
import numpy as np
import polars as pl
import pyarrow as pa

from pulse.config import DataConfig
from pulse.contracts import PreparedPosts


def load_posts(config: DataConfig) -> PreparedPosts:
    """
    Load posts from the configured dataset and return prepared text records.

    The returned records are filtered by date and include a combined text column
    created from the configured title and content columns.

    Args:
        config: Dataset loading and date filtering settings.

    Returns:
        Prepared posts with Polars records, text values, and row indices.
    """

    _validate_date(config.start_date, "start_date")
    _validate_date(config.end_date, "end_date")

    arrow_table = _load_arrow_table(config)
    _validate_required_columns(
        arrow_table,
        [
            config.date_column,
            config.title_column,
            config.content_column,
        ],
    )

    posts = _query_posts(arrow_table, config)
    texts = _extract_texts(posts, config.text_column)
    row_indices = np.arange(len(posts), dtype=np.int64)

    return PreparedPosts(
        posts=posts,
        texts=texts,
        row_indices=row_indices,
        text_column=config.text_column,
        metadata={
            "dataset_name": config.dataset_name,
            "dataset_config": config.dataset_config,
            "split": config.split,
            "cache_dir": str(config.cache_dir),
            "start_date": config.start_date,
            "end_date": config.end_date,
            "n_posts": len(posts),
        },
    )


def _load_arrow_table(config: DataConfig) -> pa.Table:
    """
    Load the configured Hugging Face dataset split as a PyArrow table.
    """

    from datasets import Dataset, DatasetDict, load_dataset

    dataset = load_dataset(
        path=config.dataset_name,
        name=config.dataset_config,
        cache_dir=str(config.cache_dir),
    )

    if isinstance(dataset, DatasetDict):
        if config.split not in dataset:
            available = ", ".join(dataset.keys())
            raise ValueError(
                f"Dataset split '{config.split}' was not found. "
                f"Available splits: {available}"
            )
        selected = dataset[config.split]
    elif isinstance(dataset, Dataset):
        selected = dataset
    else:
        raise TypeError(f"Unsupported dataset object: {type(dataset)!r}")

    return selected.data.table


def _query_posts(arrow_table: pa.Table, config: DataConfig) -> pl.DataFrame:
    """
    Query the Arrow table with DuckDB and return a Polars DataFrame.
    """

    date_column = _quote_identifier(config.date_column)
    title_column = _quote_identifier(config.title_column)
    content_column = _quote_identifier(config.content_column)
    text_column = _quote_identifier(config.text_column)

    query = f"""
        SELECT
            *,
            TRIM(
                COALESCE(CAST({title_column} AS VARCHAR), '') ||
                ' ' ||
                COALESCE(CAST({content_column} AS VARCHAR), '')
            ) AS {text_column}
        FROM arrow_table
        WHERE CAST({date_column} AS DATE)
            BETWEEN DATE '{config.start_date}' AND DATE '{config.end_date}'
    """

    return duckdb.sql(query).pl()


def _extract_texts(posts: pl.DataFrame, text_column: str) -> list[str]:
    """
    Extract text values from a Polars DataFrame.
    """

    if text_column not in posts.columns:
        raise ValueError(f"Missing text column after loading: {text_column}")

    return (
        posts.get_column(text_column)
        .fill_null("")
        .cast(pl.Utf8)
        .to_list()
    )


def _validate_required_columns(table: pa.Table, columns: list[str]) -> None:
    """
    Validate that all required columns exist in the Arrow table.
    """

    available = set(table.column_names)
    missing = [column for column in columns if column not in available]

    if missing:
        raise ValueError(
            "Dataset is missing required columns: "
            + ", ".join(sorted(missing))
        )


def _validate_date(value: str, field_name: str) -> None:
    """
    Validate an ISO date string.
    """

    try:
        date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(
            f"{field_name} must be an ISO date string such as '2026-01-28'."
        ) from exc


def _quote_identifier(identifier: str) -> str:
    """
    Quote a SQL identifier for use in DuckDB queries.
    """

    escaped = identifier.replace('"', '""')
    return f'"{escaped}"'