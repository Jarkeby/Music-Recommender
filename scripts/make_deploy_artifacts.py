"""Shrink the app's artifacts so they fit in a GitHub repo (100 MB/file limit) for Streamlit Cloud.

The full ALS item factors are 90k x 1024 float32 (~370 MB). The app only needs cosine similarity, so the
unit-normalized vectors are quantized to int8 (scale 127; ~4x smaller, near-identical neighbours) and split
into shards. (PCA to fewer dimensions was tried first and wrecked the neighbours, so all 1024 dims are kept.)
    python scripts/make_deploy_artifacts.py
Reads data/processed/{item_factors.npy,tracks.parquet}; writes app/data/. Run after run_experiment.py --save-artifacts.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

N_SHARDS = 2


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    factors = np.load("data/processed/item_factors.npy")
    norm = factors / (np.linalg.norm(factors, axis=1, keepdims=True) + 1e-9)
    quant = np.round(norm * 127).astype(np.int8)

    out = Path("app/data")
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("item_factors*.npy"):
        old.unlink()
    for i, shard in enumerate(np.array_split(quant, N_SHARDS)):
        np.save(out / f"item_factors_{i}.npy", shard)
    tracks = pd.read_parquet("data/processed/tracks.parquet")
    keep = [c for c in ["track_idx", "track_id", "n_listeners", "artist_name", "recording_name"] if c in tracks]
    tracks[keep].to_parquet(out / "tracks.parquet", index=False)
    for f in sorted(out.iterdir()):
        print(f"{f}: {f.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
