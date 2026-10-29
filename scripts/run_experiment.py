"""Run all models on the same split and print/save a comparison table.

    python scripts/run_experiment.py --data data/raw/listens.parquet --k 20 --save-artifacts
Hyperparameter defaults are the best settings found by scripts/tune.py (validation split).
Outputs: results/results.csv, results/by_activity.csv, results/figures/recall_by_activity.png
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # implicit's ALS is much faster without BLAS threading

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from recsys.data import build_dataset, filter_sparse, load_listens, time_split  # noqa: E402
from recsys.metrics import per_user_scores, sample_users  # noqa: E402
from recsys.models import (ALSRecommender, BPRRecommender, ItemKNNRecommender,  # noqa: E402
                           PopularityRecommender)

MODEL_COLORS = {"popularity": "#2a78d6", "item_knn": "#eb6834", "als": "#1baf7a", "bpr": "#eda100"}


def activity_breakdown(per_user: pd.DataFrame, train_matrix, k: int) -> pd.DataFrame:
    """Mean recall/NDCG per model within user-activity quartiles (distinct tracks in train)."""
    activity = np.diff(train_matrix.indptr)  # distinct tracks per user in train
    per_user = per_user.assign(activity=activity[per_user.user_idx.to_numpy()])
    users = per_user.drop_duplicates("user_idx")
    edges = np.unique(np.quantile(users.activity, [0, .25, .5, .75, 1]))
    # pd.cut bins are right-inclusive, so every bucket after the first starts at lo + 1
    labels = [f"{int(lo) + (i > 0)}-{int(hi)}" for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:]))]
    per_user["activity_bucket"] = pd.cut(per_user.activity, edges, labels=labels, include_lowest=True)
    out = (per_user.groupby(["activity_bucket", "model"], observed=True)
           .agg(**{f"recall@{k}": ("recall", "mean"), f"ndcg@{k}": ("ndcg", "mean"), "n_users": ("user_idx", "size")})
           .reset_index())
    return out


def plot_breakdown(table: pd.DataFrame, k: int, path: str):
    buckets = table.activity_bucket.cat.remove_unused_categories().cat.categories
    models = [m for m in MODEL_COLORS if m in set(table.model)]
    x = np.arange(len(buckets))
    width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, m in enumerate(models):
        vals = table[table.model == m].set_index("activity_bucket")[f"recall@{k}"].reindex(buckets)
        ax.bar(x + (i - (len(models) - 1) / 2) * width, vals, width * 0.92, label=m, color=MODEL_COLORS[m])
    ax.set_xticks(x, [f"{b}\ntracks" for b in buckets])
    ax.set(xlabel="User activity quartile (distinct tracks played in train)", ylabel=f"recall@{k}",
           title=f"Recall@{k} by user activity")
    ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, ncol=len(models), loc="lower left", bbox_to_anchor=(0, 1.0))
    ax.set_title(ax.get_title(), pad=28)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/raw/listens.parquet")
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--min-user", type=int, default=20)
    p.add_argument("--min-track", type=int, default=20)
    p.add_argument("--max-eval-users", type=int, default=5000)
    # alpha = confidence weight used to build each model's matrix (BPR ignores weights)
    p.add_argument("--alpha", type=float, default=10.0, help="Confidence weight for popularity/BPR")
    p.add_argument("--knn-alpha", type=float, default=0.0)
    p.add_argument("--knn-k", type=int, default=2000)
    p.add_argument("--als-alpha", type=float, default=40.0)
    p.add_argument("--als-factors", type=int, default=1024)
    p.add_argument("--als-reg", type=float, default=10.0)
    p.add_argument("--bpr-factors", type=int, default=512)
    p.add_argument("--bpr-lr", type=float, default=0.05)
    p.add_argument("--bpr-reg", type=float, default=0.01)
    p.add_argument("--bpr-iterations", type=int, default=600)
    p.add_argument("--save-artifacts", action="store_true", help="Save ALS item embeddings for the app")
    p.add_argument("--meta", default="data/raw/track_meta.parquet", help="Artist/title lookup for the app")
    args = p.parse_args()

    df = load_listens(args.data)
    print(f"Loaded {len(df):,} listens, {df.user_id.nunique():,} users, {df.track_id.nunique():,} tracks")
    df = filter_sparse(df, args.min_user, args.min_track)
    print(f"After filtering: {len(df):,} listens, {df.user_id.nunique():,} users, {df.track_id.nunique():,} tracks")

    train, test = time_split(df, test_frac=0.2)
    datasets = {a: build_dataset(train, test, alpha=a) for a in {args.alpha, args.knn_alpha, args.als_alpha}}
    ds = datasets[args.alpha]  # index maps and test items are identical across alphas
    print(f"Train matrix {ds.train_matrix.shape}, {len(ds.test_items):,} users with new test tracks")

    models = [
        (PopularityRecommender(), args.alpha),
        (ItemKNNRecommender(k_neighbors=args.knn_k), args.knn_alpha),
        (ALSRecommender(factors=args.als_factors, regularization=args.als_reg), args.als_alpha),
        (BPRRecommender(factors=args.bpr_factors, learning_rate=args.bpr_lr, regularization=args.bpr_reg,
                        iterations=args.bpr_iterations), args.alpha),
    ]
    users = sample_users(ds.test_items, args.max_eval_users)
    results, per_user = [], []
    for m, alpha in models:
        m.fit(datasets[alpha].train_matrix)
        scores = per_user_scores(m, ds.test_items, args.k, users).assign(model=m.name)
        per_user.append(scores)
        row = {"model": m.name, f"recall@{args.k}": scores.recall.mean(), f"ndcg@{args.k}": scores.ndcg.mean(),
               "n_users": len(scores)}
        results.append(row)
        print(f"{m.name:12s} recall@{args.k}={row[f'recall@{args.k}']:.4f}  ndcg@{args.k}={row[f'ndcg@{args.k}']:.4f}")

    out = pd.DataFrame(results)
    Path("results").mkdir(exist_ok=True)
    out.to_csv("results/results.csv", index=False)
    print("\n", out.to_string(index=False))

    breakdown = activity_breakdown(pd.concat(per_user, ignore_index=True), ds.train_matrix, args.k)
    breakdown.to_csv("results/by_activity.csv", index=False)
    plot_breakdown(breakdown, args.k, "results/figures/recall_by_activity.png")
    print("\nBy user activity:\n", breakdown.pivot(index="activity_bucket", columns="model",
                                                    values=f"recall@{args.k}").round(4).to_string())

    if args.save_artifacts:
        als = models[2][0]
        Path("data/processed").mkdir(parents=True, exist_ok=True)
        np.save("data/processed/item_factors.npy", als.item_factors)
        tracks = pd.DataFrame({"track_idx": range(len(ds.track_ids)), "track_id": ds.track_ids,
                               "n_listeners": ds.train_matrix.getnnz(axis=0)})
        if Path(args.meta).exists():
            tracks = tracks.merge(pd.read_parquet(args.meta), on="track_id", how="left")
        tracks.to_parquet("data/processed/tracks.parquet", index=False)
        print("Saved artifacts to data/processed/")


if __name__ == "__main__":
    main()
