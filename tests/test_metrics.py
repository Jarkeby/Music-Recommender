import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from recsys.metrics import ndcg_at_k, recall_at_k  # noqa: E402


def test_perfect_ranking():
    assert recall_at_k([1, 2, 3], {1, 2, 3}, 3) == 1.0
    assert abs(ndcg_at_k([1, 2, 3], {1, 2, 3}, 3) - 1.0) < 1e-9


def test_no_hits():
    assert recall_at_k([4, 5, 6], {1, 2}, 3) == 0.0
    assert ndcg_at_k([4, 5, 6], {1, 2}, 3) == 0.0


def test_order_matters_for_ndcg():
    assert ndcg_at_k([1, 9, 9], {1}, 3) > ndcg_at_k([9, 9, 1], {1}, 3)
