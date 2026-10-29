"""Generate fake listening data so you can test the pipeline BEFORE getting real data.

Users belong to hidden taste clusters; each cluster prefers a subset of tracks, plus a
global popularity skew. A decent model should beat the popularity baseline on this.
    python scripts/make_synthetic_data.py
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--users", type=int, default=3000)
    p.add_argument("--tracks", type=int, default=2000)
    p.add_argument("--clusters", type=int, default=15)
    p.add_argument("--listens_per_user", type=int, default=60)
    p.add_argument("--out", default="data/raw/listens.parquet")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    track_pop = rng.lognormal(0, 1.0, args.tracks)  # long-tailed but not extreme
    track_pop /= track_pop.sum()
    cluster_of_track = rng.integers(0, args.clusters, args.tracks)

    rows = []
    start = pd.Timestamp("2024-01-01")
    for u in range(args.users):
        c = rng.integers(0, args.clusters)
        w = track_pop * np.where(cluster_of_track == c, 30.0, 1.0)
        w /= w.sum()
        n = max(20, int(rng.normal(args.listens_per_user, 20)))
        tracks = rng.choice(args.tracks, size=n, p=w)
        times = start + pd.to_timedelta(rng.uniform(0, 30 * 24 * 3600, n), unit="s")
        rows.append(pd.DataFrame({"user_id": f"user_{u}",
                                  "track_id": [f"track_{t}" for t in tracks],
                                  "listened_at": times}))
    df = pd.concat(rows, ignore_index=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"Saved {len(df):,} synthetic listens to {args.out}")


if __name__ == "__main__":
    main()
