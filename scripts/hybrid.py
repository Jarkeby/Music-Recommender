"""Hybrid: blend ALS and tag-content scores, weight * z(ALS) + (1 - weight) * z(content).

The blend weight is chosen on the VALIDATION split (same scheme as tune.py), then ALS, content,
and the hybrid at that weight are each scored once on the test set.
    python scripts/hybrid.py
Needs data/raw/track_tags.parquet (scripts/pull_tags.py).
Outputs: results/hybrid_tuning.csv, results/hybrid.csv, results/hybrid_by_activity.csv
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from recsys.content import ContentRecommender, HybridRecommender, build_tag_matrix  # noqa: E402
from recsys.data import build_dataset, filter_sparse, load_listens, time_split  # noqa: E402
from recsys.metrics import per_user_scores, sample_users  # noqa: E402
from recsys.models import ALSRecommender  # noqa: E402
from run_experiment import activity_breakdown  # noqa: E402


def fit_models(train, test, tags, args):
    ds = build_dataset(train, test, alpha=args.als_alpha)
    als = ALSRecommender(factors=args.als_factors, regularization=args.als_reg).fit(ds.train_matrix)
    content = ContentRecommender(build_tag_matrix(tags, ds.track_ids)[0]).fit(ds.train_matrix)
    return ds, als, content


def score_row(model, ds, users, k, **extra):
    scores = per_user_scores(model, ds.test_items, k, users).assign(model=model.name)
    row = {"model": model.name, **extra, f"recall@{k}": scores.recall.mean(), f"ndcg@{k}": scores.ndcg.mean(),
           "n_users": len(scores)}
    return row, scores


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/raw/listens.parquet")
    p.add_argument("--tags", default="data/raw/track_tags.parquet")
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--min-user", type=int, default=20)
    p.add_argument("--min-track", type=int, default=20)
    p.add_argument("--max-eval-users", type=int, default=5000)
    p.add_argument("--weights", type=float, nargs="+", default=[0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 1.0])
    p.add_argument("--als-alpha", type=float, default=40.0)
    p.add_argument("--als-factors", type=int, default=1024)
    p.add_argument("--als-reg", type=float, default=10.0)
    args = p.parse_args()
    k = args.k

    df = filter_sparse(load_listens(args.data), args.min_user, args.min_track)
    tags = pd.read_parquet(args.tags)
    train_full, test = time_split(df, test_frac=0.2)
    train, val = time_split(train_full, test_frac=0.2)

    print("Choosing the blend weight on the validation split")
    ds, als, content = fit_models(train, val, tags, args)
    users = sample_users(ds.test_items, args.max_eval_users)
    tuning = []
    for w in args.weights:
        row, _ = score_row(HybridRecommender(als, content, w).fit(ds.train_matrix), ds, users, k, weight=w)
        tuning.append(row)
        print(f"  weight={w:<5} recall@{k}={row[f'recall@{k}']:.4f}  ndcg@{k}={row[f'ndcg@{k}']:.4f}", flush=True)
    tuning = pd.DataFrame(tuning)
    Path("results").mkdir(exist_ok=True)
    tuning.to_csv("results/hybrid_tuning.csv", index=False)
    best_w = float(tuning.sort_values(f"ndcg@{k}", ascending=False).weight.iloc[0])
    print(f"Best weight by validation ndcg@{k}: {best_w}")

    print("\nTest set")
    ds, als, content = fit_models(train_full, test, tags, args)
    users = sample_users(ds.test_items, args.max_eval_users)
    results, per_user = [], []
    for m in [als, content, HybridRecommender(als, content, best_w).fit(ds.train_matrix)]:
        row, scores = score_row(m, ds, users, k)
        results.append(row)
        per_user.append(scores)
        print(f"  {m.name:8s} recall@{k}={row[f'recall@{k}']:.4f}  ndcg@{k}={row[f'ndcg@{k}']:.4f}")
    out = pd.DataFrame(results).assign(weight=best_w)
    out.to_csv("results/hybrid.csv", index=False)

    breakdown = activity_breakdown(pd.concat(per_user, ignore_index=True), ds.train_matrix, k)
    breakdown.to_csv("results/hybrid_by_activity.csv", index=False)
    print("\nBy user activity:\n", breakdown.pivot(index="activity_bucket", columns="model",
                                                    values=f"recall@{k}").round(4).to_string())


if __name__ == "__main__":
    main()
