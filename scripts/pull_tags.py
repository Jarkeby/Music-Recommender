"""Pull MusicBrainz tags for every track in the catalog via the ListenBrainz metadata API.

track_id is a MusicBrainz recording MBID, so one batched request returns up to ~100 recordings'
tags at three levels: the recording itself, its artist(s), and its release group (album).
Recording tags are sparse; artist tags cover most tracks, so we keep all three and let the
content model decide how to weight them. Usage:
    python scripts/pull_tags.py
Output:
    data/raw/track_tags.parquet  track_id | source (recording/artist/release_group) | tag | count
Raw responses are appended to data/raw/tags_cache.jsonl so an interrupted run resumes where it
stopped. ~900 requests for 90k tracks, roughly 30-40 minutes under the API rate limit.
"""
import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

API = "https://api.listenbrainz.org/1/metadata/recording/"
HEADERS = {"User-Agent": "music-recommender-student-project"}
SOURCES = ("recording", "artist", "release_group")


def fetch(mbids: list[str], retries: int = 5) -> dict:
    url = f"{API}?inc=tag&recording_mbids={','.join(mbids)}"
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=120) as r:
                data = json.load(r)
                if r.headers.get("X-RateLimit-Remaining") == "0":
                    time.sleep(int(r.headers.get("X-RateLimit-Reset-In", 10)) + 1)
                return data
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(int(e.headers.get("X-RateLimit-Reset-In", 10)) + 1)
            elif e.code >= 500:
                time.sleep(2 ** attempt)
            else:
                raise
        except (urllib.error.URLError, TimeoutError):
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Giving up on batch starting {mbids[0]}")


def to_rows(track_id: str, meta: dict) -> list[dict]:
    rows = []
    for source in SOURCES:
        for t in meta.get("tag", {}).get(source, []):
            rows.append({"track_id": track_id, "source": source, "tag": t["tag"].strip().lower(),
                         "count": t.get("count", 1)})
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tracks", default="data/processed/tracks.parquet")
    p.add_argument("--cache", default="data/raw/tags_cache.jsonl")
    p.add_argument("--out", default="data/raw/track_tags.parquet")
    p.add_argument("--batch-size", type=int, default=100)  # 200 gets 502s from the API
    args = p.parse_args()

    ids = pd.read_parquet(args.tracks)["track_id"].tolist()
    cache = Path(args.cache)
    done = set()
    if cache.exists():
        with cache.open() as f:
            for line in f:
                done.update(json.loads(line)["requested"])
    todo = [i for i in ids if i not in done]
    print(f"{len(ids):,} tracks, {len(done):,} already fetched, {len(todo):,} to go")

    start = time.time()
    with cache.open("a") as f:
        for b in range(0, len(todo), args.batch_size):
            batch = todo[b:b + args.batch_size]
            f.write(json.dumps({"requested": batch, "response": fetch(batch)}) + "\n")
            f.flush()
            n = b + len(batch)
            if (b // args.batch_size) % 20 == 0 or n == len(todo):
                eta = (time.time() - start) / n * (len(todo) - n)
                print(f"  {n:,}/{len(todo):,}  (eta {eta / 60:.0f} min)", flush=True)

    rows = []
    with cache.open() as f:
        for line in f:
            for track_id, meta in json.loads(line)["response"].items():
                rows.extend(to_rows(track_id, meta))
    # a track with several credited artists repeats tags; merge them
    tags = pd.DataFrame(rows).groupby(["track_id", "source", "tag"], as_index=False)["count"].sum()
    tags.to_parquet(args.out, index=False)
    covered = tags.groupby("source")["track_id"].nunique()
    print(f"Saved {len(tags):,} rows to {args.out}")
    print(f"Tracks with any tag: {tags.track_id.nunique():,}/{len(ids):,}")
    print("Tracks with tags, by source:\n" + covered.to_string())


if __name__ == "__main__":
    main()
