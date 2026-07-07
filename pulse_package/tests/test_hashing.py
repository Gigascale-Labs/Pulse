from pulse.hashing import clustering_run_hash, stable_hash


def test_stable_hash_is_independent_of_dict_order() -> None:
    first = stable_hash({"a": 1, "b": 2})
    second = stable_hash({"b": 2, "a": 1})

    assert first == second


def test_stable_hash_changes_when_values_change() -> None:
    first = stable_hash({"model": "all-MiniLM-L6-v2"})
    second = stable_hash({"model": "different-model"})

    assert first != second


def test_clustering_run_hash_uses_data_hash_and_clustering_metadata() -> None:
    first = clustering_run_hash(
        data_hash="abc123",
        clustering_metadata={"method": "kmeans", "k": 13},
    )
    second = clustering_run_hash(
        data_hash="abc123",
        clustering_metadata={"method": "kmeans", "k": 17},
    )

    assert first != second