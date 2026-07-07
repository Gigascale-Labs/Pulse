import numpy as np
import polars as pl

from pulse.config import PipelineConfig
from pulse.contracts import (
    ClusterResult,
    EmbeddingResult,
    PersonaDumpResult,
    PreparedPosts,
    ReductionResult,
    SampledPosts,
)
from pulse.pipeline import run_pipeline


def test_pipeline_orchestration_with_fake_stages(monkeypatch, tmp_path) -> None:
    prepared = PreparedPosts(
        posts=pl.DataFrame({"text": ["alpha", "beta", "gamma"]}),
        texts=["alpha", "beta", "gamma"],
        row_indices=np.array([0, 1, 2]),
        metadata={"stage": "loaded"},
    )

    embeddings = EmbeddingResult(
        row_indices=np.array([0, 1, 2]),
        vectors=np.array(
            [
                [0.0, 0.0],
                [1.0, 1.0],
                [2.0, 2.0],
            ],
            dtype=np.float32,
        ),
        model_name="test-model",
        data_hash="test-hash",
    )

    sampled = SampledPosts(
        posts=prepared.posts,
        texts=prepared.texts,
        row_indices=prepared.row_indices,
        vectors=embeddings.vectors,
    )

    reduction = ReductionResult(
        name="pca50",
        row_indices=np.array([0, 1, 2]),
        vectors=np.array(
            [
                [0.0],
                [1.0],
                [2.0],
            ],
            dtype=np.float32,
        ),
    )

    cluster = ClusterResult(
        method="kmeans",
        space_name="pca50",
        row_indices=np.array([0, 1, 2]),
        labels=np.array([0, 1, 1]),
        backend="sklearn",
        metadata={"n_clusters": 2, "n_posts_per_cluster": 2},
    )

    dump = PersonaDumpResult(
        run_dir=tmp_path / "run",
        metadata_path=tmp_path / "run" / "metadata.json",
        clusters_arrow_path=tmp_path / "run" / "clusters.arrow",
        post_files=[],
    )

    monkeypatch.setattr(
        "pulse.pipeline.configure_runtime",
        lambda config: type(
            "RuntimeInfo",
            (),
            {
                "requested_device": "cpu",
                "device_type": "cpu",
                "embedding_device": "cpu",
                "cuda_visible_devices": None,
                "cuda_available": False,
            },
        )(),
    )

    monkeypatch.setattr(
        "pulse.pipeline.require_backend_available",
        lambda backend, device_type: None,
    )

    monkeypatch.setattr(
        "pulse.pipeline.load_posts",
        lambda config: prepared,
    )

    monkeypatch.setattr(
        "pulse.pipeline.preprocess_posts",
        lambda prepared, config, tokenizer_name=None: prepared,
    )

    monkeypatch.setattr(
        "pulse.pipeline.build_embeddings",
        lambda prepared, config, data_hash, device=None: embeddings,
    )

    monkeypatch.setattr(
        "pulse.pipeline.validate_embedding_alignment",
        lambda prepared, embeddings: None,
    )

    monkeypatch.setattr(
        "pulse.pipeline.sample_posts_and_embeddings",
        lambda prepared, embeddings, config: sampled,
    )

    monkeypatch.setattr(
        "pulse.pipeline.reduce_embeddings",
        lambda sampled, config, backend: {"pca50": reduction},
    )

    monkeypatch.setattr(
        "pulse.pipeline.cluster_reductions",
        lambda reductions, config: [cluster],
    )

    monkeypatch.setattr(
        "pulse.pipeline.export_persona_dumps",
        lambda sampled, clusters, config, data_hash: [dump],
    )

    config = PipelineConfig()
    config.runtime.device = "cpu"
    config.clustering.kmeans.backend = "sklearn"
    config.output.persona_dump_dir = tmp_path

    result = run_pipeline(config)

    assert result.prepared is prepared
    assert result.embeddings is embeddings
    assert result.sampled is sampled
    assert result.reductions == {"pca50": reduction}
    assert result.clusters == [cluster]
    assert result.dumps == [dump]
    assert "data_hash" in result.metadata