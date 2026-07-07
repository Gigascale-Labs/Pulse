from pathlib import Path

import pytest
import yaml

from pulse.config import PipelineConfig, PreprocessingConfig, SamplingConfig


def test_pipeline_config_loads_from_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"

    config_path.write_text(
        yaml.safe_dump(
            {
                "runtime": {
                    "device": "cpu",
                },
                "data": {
                    "start_date": "2026-01-28",
                    "end_date": "2026-02-15",
                    "cache_dir": str(tmp_path / "cache"),
                },
                "embedding": {
                    "batch_size": 32,
                    "cache_dir": str(tmp_path / "embeddings"),
                },
                "sampling": {
                    "sample_size": 100,
                },
                "clustering": {
                    "kmeans": {
                        "backend": "sklearn",
                        "cluster_sizes": [3, 5],
                    }
                },
                "output": {
                    "persona_dump_dir": str(tmp_path / "outputs"),
                },
            }
        )
    )

    config = PipelineConfig.from_yaml(config_path)

    assert config.runtime.device == "cpu"
    assert config.data.start_date == "2026-01-28"
    assert config.embedding.batch_size == 32
    assert config.sampling.sample_size == 100
    assert config.clustering.kmeans.backend == "sklearn"
    assert config.clustering.kmeans.cluster_sizes == [3, 5]
    assert config.output.persona_dump_dir == tmp_path / "outputs"


def test_pipeline_config_to_dict_converts_paths() -> None:
    config = PipelineConfig()
    values = config.to_dict()

    assert isinstance(values["data"]["cache_dir"], str)
    assert isinstance(values["embedding"]["cache_dir"], str)
    assert isinstance(values["output"]["persona_dump_dir"], str)


def test_embedding_hash_fields_contains_expected_values() -> None:
    config = PipelineConfig()
    fields = config.embedding_hash_fields()

    assert fields["start_date"] == config.data.start_date
    assert fields["end_date"] == config.data.end_date
    assert fields["embedding_model"] == config.embedding.model_name
    assert fields["chunk_size"] == config.preprocessing.chunk_size


def test_preprocessing_config_rejects_invalid_chunk_overlap() -> None:
    with pytest.raises(ValueError, match="chunk_overlap must be smaller"):
        PreprocessingConfig(chunk_size=128, chunk_overlap=128)


def test_sampling_config_rejects_non_positive_sample_size() -> None:
    with pytest.raises(ValueError, match="sample_size must be greater"):
        SamplingConfig(sample_size=0)