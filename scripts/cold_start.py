"""Cold-start experiment: can tags recommend tracks that have no plays at all?

A random 10% of tracks are made "cold": every one of their training plays is removed, so the
collaborative models know nothing about them. Each user is then asked to rank the cold tracks,
and is scored against the cold tracks they actually played (for the first time) in the test period.
    python scripts/cold_start.py
Needs data/raw/track_tags.parquet (scripts/pull_tags.py). Output: results/cold_start.csv
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from recsys.content import ContentRecommender, TagToALSRecommender, build_tag_matrix  # noqa: E402
from recsys.data import build_dataset, filter_sparse, load_listens, time_split  # noqa: E402
from recsys.metrics import per_user_scores, sample_users  # noqa: E402
from recsys.models import ALSRecommender  # noqa: E402


class RandomRecommender:
    """Baseline: any k cold tracks. With no plays to go on, this is what's left without content."""

    name = "random"

    def __init__(self, candidates, seed: int = 0):
        self.candidates, self.rng = np.asarray(candidates), np.random.default_rng(seed)

    def fit(self, train_matrix):
        return self

    def recommend_batch(self, user_idxs, k: int):
        return [self.rng.choice(self.candidates, k, replace=False).tolist() for _ in user_idxs]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/raw/listens.parquet")
    p.add_argument("--tags", default="data/raw/track_tags.parquet")
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--min-user", type=int, default=20)
    p.add_argument("--min-track", type=int, default=20)
    p.add_argument("--max-eval-users", type=int, default=5000)
    p.add_argument("--cold-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--als-alpha", type=float, default=40.0)
    p.add_argument("--als-factors", type=int, default=1024)
    p.add_argument("--als-reg", type=float, default=10.0)
    p.add_argument("--ridge-reg", type=float, default=1.0, help="Regularization of the tags -> ALS map")
    args = p.parse_args()

    df = filter_sparse(load_listens(args.data), args.min_user, args.min_track)
    train, test = time_split(df, test_frac=0.2)
    ds = build_dataset(train, test, alpha=args.als_alpha)  # test_items already excludes tracks heard in train
    n_tracks = len(ds.track_ids)

    rng = np.random.default_rng(args.seed)
    cold = np.sort(rng.choice(n_tracks, int(n_tracks * args.cold_frac), replace=False))
    warm_mask = np.ones(n_tracks, dtype=np.float32)
    warm_mask[cold] = 0
    train_warm = (ds.train_matrix @ sp.diags(warm_mask)).tocsr()
    train_warm.eliminate_zeros()

    cold_set = set(cold.tolist())
    has_history = np.diff(train_warm.indptr) > 0
    test_cold = {u: items & cold_set for u, items in ds.test_items.items() if has_history[u] and items & cold_set}
    print(f"{len(cold):,} cold tracks ({args.cold_frac:.0%}); {len(test_cold):,} users played one in the test period")

    features, vocab = build_tag_matrix(pd.read_parquet(args.tags), ds.track_ids)
    tagged = np.diff(features.indptr) > 0
    print(f"Tag vocabulary {len(vocab):,}; {tagged.mean():.0%} of tracks and {tagged[cold].mean():.0%} "
          f"of cold tracks have at least one tag")

    als = ALSRecommender(factors=args.als_factors, regularization=args.als_reg).fit(train_warm)
    models = [
        RandomRecommender(cold, seed=args.seed),
        ContentRecommender(features, candidates=cold),
        TagToALSRecommender(als, features, warm=np.flatnonzero(warm_mask), candidates=cold, reg=args.ridge_reg),
    ]
    users = sample_users(test_cold, args.max_eval_users, seed=args.seed)
    results = []
    for m in models:
        scores = per_user_scores(m.fit(train_warm), test_cold, args.k, users)
        row = {"model": m.name, f"recall@{args.k}": scores.recall.mean(), f"ndcg@{args.k}": scores.ndcg.mean(),
               "n_users": len(scores)}
        results.append(row)
        print(f"{m.name:11s} recall@{args.k}={row[f'recall@{args.k}']:.4f}  ndcg@{args.k}={row[f'ndcg@{args.k}']:.4f}")

    out = pd.DataFrame(results)
    Path("results").mkdir(exist_ok=True)
    out.to_csv("results/cold_start.csv", index=False)
    print("\n", out.to_string(index=False))


if __name__ == "__main__":
    main()
