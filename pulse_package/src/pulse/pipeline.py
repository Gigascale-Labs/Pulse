'''Connect all pipeline stages in the correct order.'''

from __future__ import annotations

from typing import Any

from pulse.clustering import cluster_reductions
from pulse.config import PipelineConfig
from pulse.contracts import PipelineResult
from pulse.data import load_posts
from pulse.embeddings import build_embeddings, validate_embedding_alignment
from pulse.export import export_persona_dumps
from pulse.hashing import embedding_data_hash
from pulse.preprocessing import preprocess_posts
from pulse.reduction import reduce_embeddings
from pulse.runtime import configure_runtime, require_backend_available
from pulse.sampling import sample_posts_and_embeddings


def run_pipeline(config: PipelineConfig) -> PipelineResult:
    """
    Run the full persona discovery pipeline.

    Args:
        config: Pipeline configuration.

    Returns:
        Full pipeline result containing intermediate objects and export paths.
    """

    runtime = configure_runtime(config.runtime)

    require_backend_available(
        backend=config.clustering.kmeans.backend,
        device_type=runtime.device_type,
    )

    data_hash = embedding_data_hash(config.embedding_hash_fields())

    prepared = load_posts(config.data)

    prepared = preprocess_posts(
        prepared=prepared,
        config=config.preprocessing,
        tokenizer_name=config.embedding.model_name,
    )

    embeddings = build_embeddings(
        prepared=prepared,
        config=config.embedding,
        data_hash=data_hash,
        device=runtime.embedding_device,
    )

    validate_embedding_alignment(
        prepared=prepared,
        embeddings=embeddings,
    )

    sampled = sample_posts_and_embeddings(
        prepared=prepared,
        embeddings=embeddings,
        config=config.sampling,
    )

    reductions = reduce_embeddings(
        sampled=sampled,
        config=config.reduction,
        backend=config.clustering.kmeans.backend,
    )

    clusters = cluster_reductions(
        reductions=reductions,
        config=config.clustering,
    )

    dumps = export_persona_dumps(
        sampled=sampled,
        clusters=clusters,
        config=config.output,
        data_hash=data_hash,
    )

    return PipelineResult(
        prepared=prepared,
        embeddings=embeddings,
        sampled=sampled,
        reductions=reductions,
        clusters=clusters,
        dumps=dumps,
        metadata=_build_pipeline_metadata(
            config=config,
            data_hash=data_hash,
            runtime_metadata={
                "requested_device": runtime.requested_device,
                "device_type": runtime.device_type,
                "embedding_device": runtime.embedding_device,
                "cuda_visible_devices": runtime.cuda_visible_devices,
                "cuda_available": runtime.cuda_available,
            },
        ),
    )


def _build_pipeline_metadata(
    config: PipelineConfig,
    data_hash: str,
    runtime_metadata: dict[str, Any],
) -> dict[str, Any]:
    """
    Build metadata for the full pipeline run.
    """

    return {
        "data_hash": data_hash,
        "runtime": runtime_metadata,
        "config": config.to_dict(),
    }


