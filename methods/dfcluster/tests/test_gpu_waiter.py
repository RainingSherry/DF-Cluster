from methods.dfcluster.gpu_waiter import _candidate


def test_candidate_requires_legal_empty_idle_gpu():
    rows = [
        {"index": 0, "memory_used_mib": 0, "utilization_percent": 0, "has_compute_process": False},
        {"index": 1, "memory_used_mib": 5000, "utilization_percent": 0, "has_compute_process": False},
        {"index": 2, "memory_used_mib": 0, "utilization_percent": 20, "has_compute_process": False},
        {"index": 3, "memory_used_mib": 0, "utilization_percent": 0, "has_compute_process": True},
        {"index": 4, "memory_used_mib": 100, "utilization_percent": 0, "has_compute_process": False},
        {"index": 7, "memory_used_mib": 0, "utilization_percent": 0, "has_compute_process": False},
    ]
    assert _candidate(rows, 4096, 10) == 4


def test_candidate_returns_none_when_legal_pool_is_busy():
    rows = [
        {"index": index, "memory_used_mib": 5000, "utilization_percent": 50, "has_compute_process": True}
        for index in range(8)
    ]
    assert _candidate(rows, 4096, 10) is None
