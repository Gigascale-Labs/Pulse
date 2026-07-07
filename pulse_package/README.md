# Pulse Persona Discovery Pipeline

Pulse Persona Discovery Pipeline packages the original persona discovery notebook into a configurable, installable Python project.

The pipeline loads text posts, preprocesses them, generates embeddings, reduces embedding spaces, clusters records, and exports persona-style cluster outputs.

## Scope

This package is focused on persona discovery from text embeddings.

It is not intended to be:

* a general machine learning framework
* a universal clustering library
* a replacement for exploratory notebook analysis

The main goal is to make the notebook workflow reusable, testable, configurable, and runnable from the command line.

## Pipeline Overview

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
```

## Project Layout

```text
persona_pipeline_package/
├── pyproject.toml
├── examples/
│   └── config.yaml
├── src/
│   └── pulse/
│       ├── config.py        # configuration models and validation
│       ├── gpu.py           # device and CUDA setup
│       ├── data.py          # data loading, filtering, cleaning, and chunking
│       ├── embeddings.py    # embedding generation, loading, and caching
│       ├── reduction.py     # sampling, PCA, and UMAP
│       ├── clustering.py    # KMeans, DBSCAN, HDBSCAN, and scoring
│       ├── export.py        # metadata, cluster Arrow files, and text dumps
│       ├── pipeline.py      # full pipeline orchestration
│       ├── cli.py           # command-line interface
│       └── contracts.py     # shared result objects and validation contracts
└── tests/
    └── test_contracts.py
```

## Installation

For local development:

```bash
pip install -e ".[dev]"
```

For CPU usage:

```bash
pip install -e ".[cpu]"
```

For CUDA/cuML usage:

```bash
pip install -e ".[cuda]"
```

CUDA/cuML support requires a compatible Linux and NVIDIA CUDA environment. Check that the CUDA version in `pyproject.toml` matches the target system.

## Usage

Validate a configuration file:

```bash
pulse-personas validate-config --config examples/config.yaml
```

Run the pipeline:

```bash
pulse-personas run --config examples/config.yaml
```

Generate embeddings and run the full pipeline:

```bash
pulse-personas run --config examples/config.yaml
```

Load existing embeddings instead of generating new ones:

```bash
pulse-personas run \
  --config examples/config.yaml \
  --no-embed \
  --embeddings /path/to/embeddings.arrow
```

## Configuration

Users configure the pipeline with YAML files. Normal users should not edit package source code to run experiments.

Example configuration:

```yaml
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
  generate: true
  batch_size: 128
  cache_dir: ./outputs/embeddings

sampling:
  sample_size: 50000
  seed: 67

reduction:
  pca:
    enabled: true
    n_components: 50

  umap:
    enabled: true
    n_components: 2
    n_neighbors: 15
    min_dist: 0.1

clustering:
  kmeans:
    enabled: true
    backend: sklearn
    cluster_sizes: [6, 13, 17]

  dbscan:
    enabled: false

  hdbscan:
    enabled: false

output:
  persona_dump_dir: ./outputs/persona_dumps
```

Run with the configuration file:

```bash
pulse-personas run --config examples/config.yaml
```

## Supported Configuration Methods

The package supports three configuration levels.

### 1. YAML Config File

This is the main public interface.

```bash
pulse-personas run --config configs/cpu.yaml
```

This is best for:

* normal users
* team members
* reproducible runs
* shared experiments
* production jobs

Config files can be committed to Git, reviewed, reused, and compared across experiments.

### 2. Python API

Advanced users can configure and run the pipeline directly from Python.

```python
from pathlib import Path

from pulse.config import PipelineConfig
from pulse.pipeline import run_pipeline

config = PipelineConfig()

config.data.start_date = "2026-02-01"
config.data.end_date = "2026-02-10"
config.clustering.kmeans.cluster_sizes = [10, 20, 30]
config.output.persona_dump_dir = Path("./outputs")

result = run_pipeline(config)
```

This is useful for:

* notebooks
* experiments
* research workflows
* programmatic sweeps

### 3. CLI Overrides

CLI overrides are useful for quick local testing.

Example:

```bash
pulse-personas run \
  --config configs/base.yaml \
  --device cpu \
  --sample-size 10000 \
  --output-dir ./outputs/test
```

For the MVP, `--config` is the most important command-line option. Additional CLI flags can be added gradually.

## Outputs

Each clustering run writes an output folder containing files such as:

```text
metadata.json
clusters.arrow
cluster_0_posts.txt
cluster_1_posts.txt
cluster_2_posts.txt
...
```

### Output Descriptions

| File                     | Description                                                        |
| ------------------------ | ------------------------------------------------------------------ |
| `metadata.json`          | Run metadata, configuration details, hashes, and pipeline settings |
| `clusters.arrow`         | Cluster assignments and related record metadata                    |
| `cluster_<id>_posts.txt` | Text dump of posts assigned to a specific cluster                  |

## Development

Run tests:

```bash
pytest
```

Run linting:

```bash
ruff check src tests
```

Run formatting:

```bash
ruff format src tests
```

Build the package:

```bash
python -m build
```

or, when using `uv`:

```bash
uv build
```

The build command creates distributable files under `dist/`, usually including a wheel file and a source distribution.

## Testing

The test suite validates the core data contracts used between pipeline stages.

Current contract tests cover:

* prepared post length validation
* required text column validation
* embedding vector shape validation
* automatic embedding vector conversion to `float32`
* sampled post and vector alignment
* reduction result alignment
* cluster label alignment
* automatic cluster label conversion to `int32`

Example:

```bash
pytest tests/test_contracts.py
```

## Main Design Idea

The original notebook executed everything at the top level:

```python
posts = load_dataset(...)
posts = posts.filter(...)
embeddings = model.encode(...)
```

The package version moves this logic into importable, testable functions:

```python
posts, texts, chunk_counts = prepare_posts(config)
embeddings = get_embeddings(texts, config, gpu)
reduction = reduce_embeddings(posts, embeddings, config)
clusters = cluster_embeddings(reduction, config)
```

This makes each pipeline stage easier to test, reuse, debug, and run from either scripts or the command line.

## Recommended Workflow

Create or edit a YAML config file:

```bash
cp examples/config.yaml configs/my_run.yaml
```

Validate it:

```bash
pulse-personas validate-config --config configs/my_run.yaml
```

Run the pipeline:

```bash
pulse-personas run --config configs/my_run.yaml
```

Inspect the generated outputs:

```bash
ls outputs/persona_dumps
```
