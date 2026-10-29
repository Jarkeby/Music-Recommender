"""Ranking metrics. Write tests for these first -- bugs here silently ruin everything."""
import numpy as np
import pandas as pd


def recall_at_k(recommended, relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    hits = len(set(recommended[:k]) & relevant)
    return hits / min(len(relevant), k)  # capped so a perfect list can score 1.0


def ndcg_at_k(recommended, relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    dcg = sum(1.0 / np.log2(i + 2) for i, item in enumerate(recommended[:k]) if item in relevant)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg


def sample_users(test_items: dict, max_users: int | None = None, seed: int = 0) -> list:
    """The users to evaluate on. Fixed seed so every model is scored on the same users."""
    users = list(test_items)
    if max_users and len(users) > max_users:
        users = [int(u) for u in np.random.default_rng(seed).choice(users, max_users, replace=False)]
    return users


def per_user_scores(recommender, test_items: dict, k: int = 20, users=None, batch_size: int = 1000) -> pd.DataFrame:
    """recall@k and NDCG@k for each user (one row per user), for breakdowns by user group."""
    users = list(test_items) if users is None else list(users)
    if hasattr(recommender, "recommend_batch"):
        recs = []
        for i in range(0, len(users), batch_size):
            recs.extend(recommender.recommend_batch(users[i:i + batch_size], k))
    else:
        recs = [recommender.recommend(int(u), k) for u in users]
    return pd.DataFrame({
        "user_idx": users,
        "recall": [recall_at_k(r, test_items[u], k) for r, u in zip(recs, users)],
        "ndcg": [ndcg_at_k(r, test_items[u], k) for r, u in zip(recs, users)],
    })


def evaluate(recommender, test_items: dict, k: int = 20, max_users: int | None = None) -> dict:
    """Average recall@k and NDCG@k over users with test items."""
    scores = per_user_scores(recommender, test_items, k, sample_users(test_items, max_users))
    return {f"recall@{k}": float(scores.recall.mean()), f"ndcg@{k}": float(scores.ndcg.mean()),
            "n_users": len(scores)}
