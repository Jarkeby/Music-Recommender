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


def summary_row(scores: pd.DataFrame, k: int, **extra) -> dict:
    """One results-table row (mean recall/NDCG over users) from per_user_scores output."""
    return {**extra, f"recall@{k}": scores.recall.mean(), f"ndcg@{k}": scores.ndcg.mean(), "n_users": len(scores)}


def activity_breakdown(per_user: pd.DataFrame, train_matrix, k: int) -> pd.DataFrame:
    """Mean recall/NDCG per model within user-activity quartiles (distinct tracks in train).

    per_user: per_user_scores output for all models stacked, with a `model` column.
    """
    activity = np.diff(train_matrix.indptr)  # distinct tracks per user in train
    per_user = per_user.assign(activity=activity[per_user.user_idx.to_numpy()])
    users = per_user.drop_duplicates("user_idx")
    edges = np.unique(np.quantile(users.activity, [0, .25, .5, .75, 1]))
    # pd.cut bins are right-inclusive, so every bucket after the first starts at lo + 1
    labels = [f"{int(lo) + (i > 0)}-{int(hi)}" for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:]))]
    per_user["activity_bucket"] = pd.cut(per_user.activity, edges, labels=labels, include_lowest=True)
    return (per_user.groupby(["activity_bucket", "model"], observed=True)
            .agg(**{f"recall@{k}": ("recall", "mean"), f"ndcg@{k}": ("ndcg", "mean"), "n_users": ("user_idx", "size")})
            .reset_index())


def evaluate(recommender, test_items: dict, k: int = 20, max_users: int | None = None) -> dict:
    """Average recall@k and NDCG@k over users with test items."""
    scores = per_user_scores(recommender, test_items, k, sample_users(test_items, max_users))
    return summary_row(scores, k)
