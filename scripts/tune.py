"""Grid-search hyperparameters on a VALIDATION split, logging every run.

The test period (last 20% of the timeline) is never touched here. Instead the training
period is split again by time: its first 80% trains, its last 20% validates. Pick the best
settings from the log, then report them once on the test set with run_experiment.py.

    python scripts/tune.py                 # all models
    python scripts/tune.py --models als    # just ALS
Every run is appended to results/tuning_log.csv.
"""
import argparse
import itertools
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from recsys.data import build_dataset, filter_sparse, load_listens, time_split  # noqa: E402
from recsys.metrics import evaluate  # noqa: E402
from recsys.models import ALSRecommender, BPRRecommender, ItemKNNRecommender  # noqa: E402

# alpha is the confidence weight used to build the matrix (see build_dataset)
GRIDS = {
    "als": dict(factors=[64, 128, 256], regularization=[0.01, 0.1, 1.0], alpha=[1, 5, 10, 40]),
    "bpr": dict(factors=[64, 128, 256], learning_rate=[0.01, 0.05, 0.1], regularization=[0.001, 0.01],
                iterations=[100, 300]),
    "item_knn": dict(k_neighbors=[20, 50, 100, 200], alpha=[1, 10]),
}
MODELS = {"als": ALSRecommender, "bpr": BPRRecommender, "item_knn": ItemKNNRecommender}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/raw/listens.parquet")
    p.add_argument("--models", nargs="+", default=list(GRIDS), choices=list(GRIDS))
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--min-user", type=int, default=20)
    p.add_argument("--min-track", type=int, default=20)
    p.add_argument("--max-eval-users", type=int, default=5000)
    p.add_argument("--log", default="results/tuning_log.csv")
    p.add_argument("--grid", type=json.loads, help='Override the grid for a single --models entry, '
                   'e.g. --models als --grid "{\\"factors\\": [256, 512], \\"alpha\\": [40, 100]}"')
    args = p.parse_args()
    if args.grid:
        if len(args.models) != 1:
            p.error("--grid needs exactly one --models entry")
        GRIDS[args.models[0]] = args.grid

    df = filter_sparse(load_listens(args.data), args.min_user, args.min_track)
    train_full, _test = time_split(df, test_frac=0.2)   # test is set aside, never used here
    train, val = time_split(train_full, test_frac=0.2)
    print(f"Tuning on train {train.listened_at.min()} .. {train.listened_at.max()}, "
          f"validating on {val.listened_at.min()} .. {val.listened_at.max()}")

    datasets = {}  # alpha -> Dataset, built once per alpha
    run_id = time.strftime("%Y%m%d-%H%M%S")
    log_path = Path(args.log)
    log_path.parent.mkdir(exist_ok=True)

    for name in args.models:
        grid = GRIDS[name]
        for values in itertools.product(*grid.values()):
            params = dict(zip(grid, values))
            alpha = params.pop("alpha", 10.0)
            if alpha not in datasets:
                datasets[alpha] = build_dataset(train, val, alpha=alpha)
            ds = datasets[alpha]

            t0 = time.time()
            model = MODELS[name](**params).fit(ds.train_matrix)
            fit_s = time.time() - t0
            scores = evaluate(model, ds.test_items, k=args.k, max_users=args.max_eval_users)
            # fixed columns (params as JSON) so models with different hyperparameters share one CSV
            row = {"run_id": run_id, "model": name, "alpha": alpha, "params": json.dumps(params), **scores,
                   "fit_seconds": round(fit_s, 1)}
            print(f"{name:9s} alpha={alpha:<5} {params}  recall@{args.k}={scores[f'recall@{args.k}']:.4f}  "
                  f"ndcg@{args.k}={scores[f'ndcg@{args.k}']:.4f}  ({fit_s:.0f}s)", flush=True)
            pd.DataFrame([row]).to_csv(log_path, mode="a", header=not log_path.exists(), index=False)

    log = pd.read_csv(log_path)
    log = log[log.run_id.astype(str) == run_id]
    metric = f"ndcg@{args.k}"
    print(f"\nBest per model by validation {metric}:")
    for name, g in log.groupby("model"):
        best = g.sort_values(metric, ascending=False).iloc[0]
        print(f"  {name:9s} alpha={best.alpha} {best.params}  {metric}={best[metric]:.4f}  "
              f"recall@{args.k}={best[f'recall@{args.k}']:.4f}")


if __name__ == "__main__":
    main()
