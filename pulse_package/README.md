# persona-pipeline

This package is a first-pass conversion of `data_load_personae.ipynb` into an installable Python project.

The notebook did everything at top level: imports, configuration, data loading, cleaning, embedding, PCA/UMAP, clustering, and writing text dumps. In a package, top-level execution is moved into functions and a command-line entry point.

## Layout

```text
persona_pipeline_package/
├── pyproject.toml
├── examples/config.json
├── src/persona_pipeline/
│   ├── config.py        # all tunable settings and stable hashes
│   ├── gpu.py           # CUDA_VISIBLE_DEVICES and torch device setup
│   ├── data.py          # load, clean, language filter, stopword filter, chunk text
│   ├── embeddings.py    # generate/load/dump embedding Arrow files
│   ├── reduction.py     # subsample, PCA, UMAP
│   ├── clustering.py    # KMeans, DBSCAN, HDBSCAN, quality scoring
│   ├── export.py        # metadata, clusters.arrow, cluster text dumps
│   ├── pipeline.py      # orchestrates the full run
│   └── cli.py           # `persona-pipeline` command
└── tests/
```

## Install

```bash
cd persona_pipeline_package
uv sync
```

RAPIDS/cuML requires a compatible Linux + NVIDIA CUDA environment. The included `pyproject.toml` uses CUDA 12 package names (`cuml-cu12`, `cupy-cuda12x`). Change these if your target system uses a different supported CUDA stack.

## Run

Generate embeddings and run the full pipeline:

```bash
uv run persona-pipeline --config examples/config.json
```

Load existing embeddings instead:

```bash
uv run persona-pipeline --config examples/config.json --no-embed --embeddings /path/to/embeddings.arrow
```

Override device or batch size:

```bash
uv run persona-pipeline --device cuda:0 --batch-size 3000
```

## Build a distributable package

```bash
uv build
```

This creates files under `dist/`, usually a wheel (`.whl`) and source distribution (`.tar.gz`).

## Main idea

A notebook is a good experiment log. A package is reusable software. The conversion is mostly changing this:

```python
posts = load_dataset(...)
posts = posts.filter(...)
embeddings = model.encode(...)
```

into this:

```python
posts, texts, chunk_counts = prepare_posts(config)
embeddings = get_embeddings(texts, config, gpu)
reduction = reduce_embeddings(posts, embeddings, config)
```

That makes each stage importable, testable, and runnable from scripts or the command line.


########## CONFIG README ##############
How users will change config
1. They install your package

Example:

pip install persona-pipeline

or with GPU support:

pip install "persona-pipeline[cuda]"
2. They create their own config file

You provide an example config in the package docs:

runtime:
  device: cpu

data:
  dataset_name: SimulaMet/moltbook-observatory-archive
  dataset_config: posts
  split: archive
  cache_dir: ./data/cache
  start_date: "2026-01-28"
  end_date: "2026-02-15"

embedding:
  model_name: all-MiniLM-L6-v2
  batch_size: 128
  generate: true
  cache_dir: ./outputs/embeddings

sampling:
  sample_size: 50000
  seed: 67

clustering:
  kmeans:
    enabled: true
    backend: sklearn
    cluster_sizes: [6, 13, 17]

output:
  persona_dump_dir: ./outputs/persona_dumps

Then they run:

persona-pipeline run --config my_config.yaml

That is the normal workflow.

3 supported ways to configure the package

You should support these three levels.

Level 1: YAML config file

This is the main public interface.

persona-pipeline run --config configs/cpu.yaml

This is best for:

normal users
team members
reproducible runs
shared experiments
production jobs

The config file can be committed to Git, reviewed, and reused.

Level 2: Python API

Advanced users can configure it in Python:

from pathlib import Path
from persona_pipeline.config import PipelineConfig, DataConfig, KMeansConfig
from persona_pipeline.pipeline import run_pipeline

config = PipelineConfig()
config.data.start_date = "2026-02-01"
config.data.end_date = "2026-02-10"
config.clustering.kmeans.cluster_sizes = [10, 20, 30]
config.output.persona_dump_dir = Path("./outputs")

result = run_pipeline(config)

This is useful for:

notebooks
experiments
research users
programmatic sweeps
Level 3: CLI overrides

Later, not necessarily MVP, support quick command-line overrides:

persona-pipeline run \
  --config configs/base.yaml \
  --device cpu \
  --sample-size 10000 \
  --output-dir ./outputs/test

This is useful for quick testing.

But do not start with dozens of CLI flags. For MVP, --config is enough.


given by chatgpt; need to refine for markdown;

# Pulse Persona Discovery Pipeline

Pulse packages the persona discovery notebook into a configurable Python pipeline.

The pipeline loads text posts, preprocesses them, generates embeddings, reduces embedding spaces, clusters records, and exports persona-style cluster outputs.

## Scope

This package is focused on persona discovery from text embeddings. It is not a general machine learning framework or a universal clustering library.

## Pipeline

```text
Hugging Face dataset / Arrow data
→ DuckDB filtering
→ Polars DataFrame
→ text preprocessing
→ SentenceTransformer embeddings
→ Arrow embedding cache
→ sampling
→ PCA / UMAP reduction
→ clustering
→ Arrow cluster assignments
→ persona text exports
Installation

For development:

pip install -e ".[dev]"

For CPU usage:

pip install -e ".[cpu]"

For CUDA/cuML usage:

pip install -e ".[cuda]"
Usage

Validate a config:

pulse-personas validate-config --config examples/config.yaml

Run the pipeline:

pulse-personas run --config examples/config.yaml
Configuration

Users control pipeline behavior with YAML config files. They should not edit package source code for normal runs.

Common fields to change:

runtime:
  device: cpu

data:
  start_date: "2026-01-28"
  end_date: "2026-02-15"

embedding:
  generate: true
  batch_size: 128

sampling:
  sample_size: 10000

clustering:
  kmeans:
    backend: sklearn
    cluster_sizes: [6, 13, 17]

output:
  persona_dump_dir: outputs/persona_dumps
Outputs

Each clustering run writes a folder containing:

metadata.json
clusters.arrow
cluster_0_posts.txt
cluster_1_posts.txt
...
Development

Run tests:

pytest

Format/lint:

ruff check src tests


---

# `tests/test_contracts.py`

```python id="ylkd7x"
import numpy as np
import polars as pl
import pytest

from pulse.contracts import (
    ClusterResult,
    EmbeddingResult,
    PreparedPosts,
    ReductionResult,
    SampledPosts,
)


def test_prepared_posts_validates_lengths() -> None:
    posts = pl.DataFrame({"text": ["a", "b"]})

    with pytest.raises(ValueError, match="posts and texts"):
        PreparedPosts(
            posts=posts,
            texts=["a"],
            row_indices=np.array([0, 1]),
        )


def test_prepared_posts_requires_text_column() -> None:
    posts = pl.DataFrame({"body": ["a"]})

    with pytest.raises(ValueError, match="Missing text column"):
        PreparedPosts(
            posts=posts,
            texts=["a"],
            row_indices=np.array([0]),
            text_column="text",
        )


def test_embedding_result_requires_2d_vectors() -> None:
    with pytest.raises(ValueError, match="vectors must be a 2D array"):
        EmbeddingResult(
            row_indices=np.array([0, 1]),
            vectors=np.array([1.0, 2.0]),
            model_name="test-model",
            data_hash="test-hash",
        )


def test_embedding_result_converts_vectors_to_float32() -> None:
    result = EmbeddingResult(
        row_indices=np.array([0, 1]),
        vectors=np.array([[1.0], [2.0]], dtype=np.float64),
        model_name="test-model",
        data_hash="test-hash",
    )

    assert result.vectors.dtype == np.float32


def test_sampled_posts_validates_alignment() -> None:
    posts = pl.DataFrame({"text": ["a", "b"]})

    with pytest.raises(ValueError, match="vectors and posts"):
        SampledPosts(
            posts=posts,
            texts=["a", "b"],
            row_indices=np.array([0, 1]),
            vectors=np.array([[1.0]], dtype=np.float32),
        )


def test_reduction_result_validates_alignment() -> None:
    with pytest.raises(ValueError, match="row_indices and vectors"):
        ReductionResult(
            name="pca50",
            row_indices=np.array([0, 1]),
            vectors=np.array([[1.0]], dtype=np.float32),
        )


def test_cluster_result_validates_alignment() -> None:
    with pytest.raises(ValueError, match="row_indices and labels"):
        ClusterResult(
            method="kmeans",
            space_name="pca50",
            row_indices=np.array([0, 1]),
            labels=np.array([0]),
            backend="sklearn",
        )


def test_cluster_result_converts_labels_to_int32() -> None:
    result = ClusterResult(
        method="kmeans",
        space_name="pca50",
        row_indices=np.array([0, 1]),
        labels=np.array([0.0, 1.0]),
        backend="sklearn",
    )

    assert result.labels.dtype == np.int32