"""Loading, filtering, splitting, and matrix building."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp


def load_listens(path: str) -> pd.DataFrame:
    """Expects columns: user_id, track_id, listened_at."""
    df = pd.read_parquet(path)
    df["listened_at"] = pd.to_datetime(df["listened_at"])
    return df[["user_id", "track_id", "listened_at"]].dropna()


def filter_sparse(df: pd.DataFrame, min_user: int = 20, min_track: int = 20, n_iter: int = 3) -> pd.DataFrame:
    """Drop users/tracks with too few listens. Repeats because filtering one affects the other."""
    for _ in range(n_iter):
        df = df[df.groupby("user_id")["track_id"].transform("size") >= min_user]
        df = df[df.groupby("track_id")["user_id"].transform("size") >= min_track]
    return df.reset_index(drop=True)


def time_split(df: pd.DataFrame, test_frac: float = 0.2):
    """Train on the earliest (1 - test_frac) of listens, test on the latest.

    Time-based splitting avoids leakage: the model never sees the future.
    """
    cutoff = df["listened_at"].quantile(1 - test_frac)
    return df[df["listened_at"] < cutoff].copy(), df[df["listened_at"] >= cutoff].copy()


@dataclass
class Dataset:
    train_matrix: sp.csr_matrix   # users x tracks, confidence-weighted play counts
    test_items: dict              # user_idx -> set(track_idx) of NEW tracks heard in test period
    user_ids: np.ndarray          # idx -> original user id
    track_ids: np.ndarray         # idx -> original track id


def build_dataset(train: pd.DataFrame, test: pd.DataFrame, alpha: float = 10.0) -> Dataset:
    """Build index maps from TRAIN only, then a sparse matrix and test ground truth.

    Confidence = 1 + alpha * log1p(play_count)  (a common implicit-feedback weighting).
    Test only counts users/tracks seen in training (warm-start evaluation), and only
    tracks the user had NOT already played in train (we recommend *new* music).
    """
    user_ids = train["user_id"].astype("category").cat.categories.to_numpy()
    track_ids = train["track_id"].astype("category").cat.categories.to_numpy()
    u_map = {u: i for i, u in enumerate(user_ids)}
    t_map = {t: i for i, t in enumerate(track_ids)}

    counts = train.groupby(["user_id", "track_id"]).size().reset_index(name="n")
    rows = counts["user_id"].map(u_map).to_numpy()
    cols = counts["track_id"].map(t_map).to_numpy()
    vals = 1.0 + alpha * np.log1p(counts["n"].to_numpy())
    mat = sp.csr_matrix((vals, (rows, cols)), shape=(len(user_ids), len(track_ids)), dtype=np.float32)

    t = test[test["user_id"].isin(u_map) & test["track_id"].isin(t_map)]
    t = pd.DataFrame({"u": t["user_id"].map(u_map).to_numpy(), "t": t["track_id"].map(t_map).to_numpy()})
    t = t.drop_duplicates()
    seen = pd.MultiIndex.from_arrays([rows, cols])
    t = t[~pd.MultiIndex.from_frame(t[["u", "t"]]).isin(seen)]  # only new tracks
    test_items = {int(u): set(g.tolist()) for u, g in t.groupby("u")["t"]}
    return Dataset(mat, test_items, user_ids, track_ids)
