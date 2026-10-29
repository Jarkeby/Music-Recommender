"""Download recent ListenBrainz daily incremental dumps (Parquet) -- no GCP account needed.

Each daily "spark" incremental dump is a ~175 MB tar
holding Parquet files of the listens submitted that day. Usage:
    python scripts/pull_from_dumps.py --days 14
Outputs:
    data/raw/listens.parquet     user_id | track_id | listened_at   (what the pipeline expects)
    data/raw/track_meta.parquet  track_id | artist_name | recording_name   (for the app)
Downloaded tars are cached in data/raw/dumps/ so re-runs don't re-download.

Note: dumps contain everything *submitted* that day, including users importing years of old
history (e.g. from Last.fm). We keep only listens whose timestamp falls inside the dump window,
so the time-based train/test split reflects real, current listening.
"""
import argparse
import re
import tarfile
import urllib.request
from pathlib import Path

import pandas as pd

BASE = "https://data.metabrainz.org/pub/musicbrainz/listenbrainz/incremental/"
COLS = ["listened_at", "user_id", "recording_msid", "recording_mbid", "artist_name", "recording_name"]


def list_dumps_by_date():
    """Return {date: [dump directory names]}. A date usually has one big dump plus a tiny
    catch-up dump (which one has the lower id varies), so we keep every dump for a date."""
    html = urllib.request.urlopen(BASE, timeout=60).read().decode()
    by_date = {}
    for n in sorted(set(re.findall(r'href="(listenbrainz-dump-\d+-\d{8}-\d{6}-incremental)/"', html))):
        by_date.setdefault(n.split("-")[3], []).append(n)
    return by_date


def download(name: str, cache: Path) -> Path:
    fname = name.replace("listenbrainz-dump-", "listenbrainz-spark-dump-") + ".tar"
    out = cache / fname
    if out.exists():
        print(f"  cached   {fname}")
        return out
    print(f"  fetching {fname} ...")
    tmp = out.with_suffix(".part")
    urllib.request.urlretrieve(BASE + name + "/" + fname, tmp)
    tmp.rename(out)
    return out


def read_tar(path: Path) -> pd.DataFrame:
    frames = []
    with tarfile.open(path) as tar:
        for m in tar.getmembers():
            if m.name.endswith(".parquet"):
                frames.append(pd.read_parquet(tar.extractfile(m), columns=COLS))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLS)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=14, help="How many recent dump dates to use (~175 MB each)")
    p.add_argument("--skip-latest", type=int, default=1, help="Skip the N newest days (may still be in progress)")
    p.add_argument("--out", default="data/raw/listens.parquet")
    p.add_argument("--meta-out", default="data/raw/track_meta.parquet")
    p.add_argument("--cache", default="data/raw/dumps")
    p.add_argument("--keep-backfill", action="store_true", help="Keep imported listens older than the window")
    args = p.parse_args()

    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    by_date = list_dumps_by_date()
    dates = sorted(by_date, reverse=True)[args.skip_latest:args.skip_latest + args.days]
    dumps = [n for d in sorted(dates) for n in by_date[d]]
    print(f"Using {len(dumps)} dumps across {len(dates)} dates: {dates[-1]} .. {dates[0]}")
    # dump dated D covers listens submitted on day D-1
    window_start = pd.Timestamp(dates[-1]) - pd.Timedelta(days=1)

    frames = []
    for name in dumps:
        df = read_tar(download(name, cache))
        # Prefer the MusicBrainz recording ID; fall back to ListenBrainz's own msid when unmatched.
        df["track_id"] = df["recording_mbid"].fillna(df["recording_msid"])
        n_raw = len(df)
        if not args.keep_backfill:
            df = df[df["listened_at"] >= window_start]
        frames.append(df[["user_id", "track_id", "listened_at", "artist_name", "recording_name"]])
        print(f"    {len(df):,} of {n_raw:,} listens kept")
    df = pd.concat(frames, ignore_index=True)
    df["user_id"] = df["user_id"].astype(str)
    df = df.dropna(subset=["track_id", "listened_at"]).drop_duplicates(["user_id", "track_id", "listened_at"])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df[["user_id", "track_id", "listened_at"]].to_parquet(args.out, index=False)
    meta = df.groupby("track_id", as_index=False)[["artist_name", "recording_name"]].first()
    meta.to_parquet(args.meta_out, index=False)
    print(f"Saved {len(df):,} listens ({df.user_id.nunique():,} users, {df.track_id.nunique():,} tracks) "
          f"spanning {df.listened_at.min()} .. {df.listened_at.max()} to {args.out}")
    print(f"Saved metadata for {len(meta):,} tracks to {args.meta_out}")


if __name__ == "__main__":
    main()
