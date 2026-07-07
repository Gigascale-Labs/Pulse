from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml


DEFAULT_SPAM_PATTERNS = [
    r'"op"\s*:\s*"mint"',
    r'"op"\s*:\s*"link"',
    r'"p"\s*:\s*"mbc-20"',
]


@dataclass
class RuntimeConfig:
    """
    Runtime settings for device selection.

    Attributes:
        device: Requested compute device. Examples are ``cpu``, ``cuda:0``,
            ``cuda:1``, or ``cuda:0,1``.
    """

    device: str = "cuda:0,1"


@dataclass
class DataConfig:
    """
    Dataset loading and date filtering settings.

    Attributes:
        dataset_name: Hugging Face dataset name.
        dataset_config: Hugging Face dataset configuration name.
        split: Dataset split to read.
        cache_dir: Local cache directory for the dataset.
        start_date: Inclusive start date used for filtering records.
        end_date: Inclusive end date used for filtering records.
        date_column: Column used for date filtering.
        title_column: Column containing post titles.
        content_column: Column containing post content.
        text_column: Name of the combined text column created by the pipeline.
    """

    dataset_name: str = "SimulaMet/moltbook-observatory-archive"
    dataset_config: str = "posts"
    split: str = "archive"
    cache_dir: Path = Path("/datapool/analysis_data/proj-sim/observatory_data")

    start_date: str = "2026-01-28"
    end_date: str = "2026-02-15"
    date_column: str = "created_at"

    title_column: str = "title"
    content_column: str = "content"
    text_column: str = "text"

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)


@dataclass
class PreprocessingConfig:
    """
    Text filtering and chunking settings.

    Attributes:
        spam_patterns: Regular expressions used to remove spam-like records.
        lang_filter: Language code to keep after language detection.
        remove_stopwords: Whether to remove stopwords before embedding.
        min_words_after_stopwords: Minimum word count required after stopword removal.
        chunk_size: Maximum tokenizer length for each text chunk.
        chunk_overlap: Token overlap between neighbouring chunks.
    """

    spam_patterns: list[str] = field(default_factory=lambda: DEFAULT_SPAM_PATTERNS.copy())
    lang_filter: str = "en"
    remove_stopwords: bool = True
    min_words_after_stopwords: int = 10
    chunk_size: int = 512
    chunk_overlap: int = 64

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero.")

        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative.")

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")

        if self.min_words_after_stopwords < 0:
            raise ValueError("min_words_after_stopwords cannot be negative.")


@dataclass
class EmbeddingConfig:
    """
    Embedding generation and cache settings.

    Attributes:
        model_name: SentenceTransformer model name.
        batch_size: Batch size passed to the embedding model.
        generate: Whether to generate embeddings instead of loading an existing file.
        cache_dir: Directory where embedding Arrow files are written or searched.
        load_path: Optional explicit path to an existing embedding Arrow file.
        normalize_embeddings: Whether to L2-normalize generated embeddings.
    """

    model_name: str = "all-MiniLM-L6-v2"
    batch_size: int = 4069
    generate: bool = True
    cache_dir: Path = Path("/datapool/analysis_data/proj-sim/embeddings")
    load_path: Path | None = None
    normalize_embeddings: bool = True

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)

        if self.load_path is not None:
            self.load_path = Path(self.load_path)

        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")


@dataclass
class SamplingConfig:
    """
    Sampling settings used before dimensionality reduction and clustering.

    Attributes:
        sample_size: Maximum number of prepared records to sample.
        seed: Random seed used for sampling.
    """

    sample_size: int = 100_000
    seed: int = 67

    def __post_init__(self) -> None:
        if self.sample_size <= 0:
            raise ValueError("sample_size must be greater than zero.")


@dataclass
class PCAConfig:
    """
    PCA dimensionality reduction settings.

    Attributes:
        mode: PCA selection mode. Use ``variance`` to retain a target explained
            variance, or ``components`` to use a fixed number of dimensions.
        explained_variance: Target explained variance when mode is ``variance``.
        n_components: Number of PCA dimensions when mode is ``components``.
        pca50_components: Fixed PCA space used by the notebook for comparison.
    """

    mode: Literal["variance", "components"] = "variance"
    explained_variance: float = 0.90
    n_components: int = 50
    pca50_components: int = 50

    def __post_init__(self) -> None:
        if self.mode not in {"variance", "components"}:
            raise ValueError("mode must be either 'variance' or 'components'.")

        if not 0 < self.explained_variance <= 1:
            raise ValueError("explained_variance must be in the range (0, 1].")

        if self.n_components <= 0:
            raise ValueError("n_components must be greater than zero.")

        if self.pca50_components <= 0:
            raise ValueError("pca50_components must be greater than zero.")


@dataclass
class UMAPConfig:
    """
    UMAP dimensionality reduction settings.

    Attributes:
        enabled: Whether to create UMAP embedding spaces.
        n_components: Number of UMAP dimensions.
        n_neighbors: Number of neighbours used by UMAP.
        min_dist: Minimum distance parameter used by UMAP.
        random_state: Random seed used by UMAP.
        metric: Distance metric used by UMAP.
    """

    enabled: bool = True
    n_components: int = 3
    n_neighbors: int = 30
    min_dist: float = 0.0
    random_state: int = 42
    metric: str = "cosine"

    def __post_init__(self) -> None:
        if self.n_components <= 0:
            raise ValueError("n_components must be greater than zero.")

        if self.n_neighbors <= 0:
            raise ValueError("n_neighbors must be greater than zero.")

        if self.min_dist < 0:
            raise ValueError("min_dist cannot be negative.")


@dataclass
class ReductionConfig:
    """
    Dimensionality reduction settings.

    Attributes:
        pca: PCA settings.
        umap: UMAP settings.
    """

    pca: PCAConfig = field(default_factory=PCAConfig)
    umap: UMAPConfig = field(default_factory=UMAPConfig)


@dataclass
class KMeansConfig:
    """
    KMeans clustering settings.

    Attributes:
        enabled: Whether to run KMeans clustering.
        backend: Implementation backend. The initial package should support
            ``cuml`` for notebook parity and ``sklearn`` for CPU fallback.
        cluster_sizes: K values used for final KMeans runs.
        sweep_min_k: Minimum K used during silhouette sweep.
        sweep_max_k: Maximum K used during silhouette sweep.
        random_state: Random seed passed to KMeans.
        n_init: Number of KMeans initializations.
        n_posts_per_cluster: Number of representative posts exported per cluster.
    """

    enabled: bool = True
    backend: Literal["cuml", "sklearn"] = "cuml"
    cluster_sizes: list[int] = field(default_factory=lambda: [6, 13, 17, 29])
    sweep_min_k: int = 3
    sweep_max_k: int = 29
    random_state: int = 42
    n_init: int = 10
    n_posts_per_cluster: int = 500

    def __post_init__(self) -> None:
        if self.backend not in {"cuml", "sklearn"}:
            raise ValueError("backend must be either 'cuml' or 'sklearn'.")

        if not self.cluster_sizes:
            raise ValueError("cluster_sizes cannot be empty.")

        if any(k <= 1 for k in self.cluster_sizes):
            raise ValueError("all cluster_sizes values must be greater than one.")

        if self.sweep_min_k <= 1:
            raise ValueError("sweep_min_k must be greater than one.")

        if self.sweep_max_k < self.sweep_min_k:
            raise ValueError("sweep_max_k must be greater than or equal to sweep_min_k.")

        if self.n_init <= 0:
            raise ValueError("n_init must be greater than zero.")

        if self.n_posts_per_cluster <= 0:
            raise ValueError("n_posts_per_cluster must be greater than zero.")


@dataclass
class DensityClusteringConfig:
    """
    Optional density-clustering settings.

    Attributes:
        enabled: Whether to run DBSCAN and HDBSCAN experiments.
        dbscan_eps_values: Epsilon values used for DBSCAN.
        dbscan_min_samples: Minimum samples used for DBSCAN.
        hdbscan_min_cluster_sizes: Minimum cluster sizes used for HDBSCAN.
        n_posts_per_cluster: Number of representative posts exported per cluster.
    """

    enabled: bool = False
    dbscan_eps_values: list[float] = field(
        default_factory=lambda: [0.2, 0.5, 1.0, 2.0, 3.0, 5.0]
    )
    dbscan_min_samples: int = 10
    hdbscan_min_cluster_sizes: list[int] = field(
        default_factory=lambda: [50, 100, 200, 500, 1000]
    )
    n_posts_per_cluster: int = 3

    def __post_init__(self) -> None:
        if any(eps <= 0 for eps in self.dbscan_eps_values):
            raise ValueError("all dbscan_eps_values must be greater than zero.")

        if self.dbscan_min_samples <= 0:
            raise ValueError("dbscan_min_samples must be greater than zero.")

        if any(size <= 0 for size in self.hdbscan_min_cluster_sizes):
            raise ValueError("all hdbscan_min_cluster_sizes must be greater than zero.")

        if self.n_posts_per_cluster <= 0:
            raise ValueError("n_posts_per_cluster must be greater than zero.")


@dataclass
class ClusteringConfig:
    """
    Clustering settings.

    Attributes:
        kmeans: KMeans clustering settings.
        density: Optional DBSCAN and HDBSCAN settings.
    """

    kmeans: KMeansConfig = field(default_factory=KMeansConfig)
    density: DensityClusteringConfig = field(default_factory=DensityClusteringConfig)


@dataclass
class OutputConfig:
    """
    Output directory settings.

    Attributes:
        persona_dump_dir: Root directory for persona export runs.
    """

    persona_dump_dir: Path = Path("/datapool/analysis_data/proj-sim/persona_dumps")

    def __post_init__(self) -> None:
        self.persona_dump_dir = Path(self.persona_dump_dir)


@dataclass
class PipelineConfig:
    """
    Full configuration for the persona pipeline.

    Attributes:
        runtime: Runtime and device settings.
        data: Dataset loading settings.
        preprocessing: Text preprocessing settings.
        embedding: Embedding generation and cache settings.
        sampling: Sampling settings.
        reduction: Dimensionality reduction settings.
        clustering: Clustering settings.
        output: Output settings.
    """

    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    data: DataConfig = field(default_factory=DataConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    reduction: ReductionConfig = field(default_factory=ReductionConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "PipelineConfig":
        """
        Build a PipelineConfig from a nested dictionary.
        """

        return cls(
            runtime=RuntimeConfig(**values.get("runtime", {})),
            data=DataConfig(**values.get("data", {})),
            preprocessing=PreprocessingConfig(**values.get("preprocessing", {})),
            embedding=EmbeddingConfig(**values.get("embedding", {})),
            sampling=SamplingConfig(**values.get("sampling", {})),
            reduction=ReductionConfig(
                pca=PCAConfig(**values.get("reduction", {}).get("pca", {})),
                umap=UMAPConfig(**values.get("reduction", {}).get("umap", {})),
            ),
            clustering=ClusteringConfig(
                kmeans=KMeansConfig(**values.get("clustering", {}).get("kmeans", {})),
                density=DensityClusteringConfig(
                    **values.get("clustering", {}).get("density", {})
                ),
            ),
            output=OutputConfig(**values.get("output", {})),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        """
        Load configuration from a YAML file.
        """

        path = Path(path)
        values = yaml.safe_load(path.read_text()) or {}
        if not isinstance(values, dict):
            raise ValueError(f"Configuration file must contain a mapping: {path}")
        return cls.from_dict(values)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the configuration to a plain dictionary.
        """

        return _paths_to_strings(asdict(self))

    def embedding_hash_fields(self) -> dict[str, Any]:
        """
        Return fields that determine whether embedding cache files are reusable.
        """

        return {
            "start_date": self.data.start_date,
            "end_date": self.data.end_date,
            "spam_patterns": self.preprocessing.spam_patterns,
            "lang_filter": self.preprocessing.lang_filter,
            "embedding_model": self.embedding.model_name,
            "chunk_size": self.preprocessing.chunk_size,
            "chunk_overlap": self.preprocessing.chunk_overlap,
        }


def _paths_to_strings(value: Any) -> Any:
    """
    Convert Path objects inside nested containers to strings.
    """

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {key: _paths_to_strings(item) for key, item in value.items()}

    if isinstance(value, list):
        return [_paths_to_strings(item) for item in value]

    return value