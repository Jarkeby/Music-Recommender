
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
