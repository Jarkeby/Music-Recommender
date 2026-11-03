
import argparse
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

LB_API = "https://api.listenbrainz.org/1/metadata/recording/?inc=artist&recording_mbids="
MB_API = "https://musicbrainz.org/ws/2/artist/{}?inc=url-rels&fmt=json"
HEADERS = {"User-Agent": "music-recommender-student-project (jacobhardwick4@gmail.com)"}
UUID = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
SPOTIFY_ARTIST = re.compile(r"^https://open\.spotify\.com/artist/\w+")


def get_json(url: str, retries: int = 5):
    # GET with backoff; returns None for a 404 (unknown id)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(int(e.headers.get("Retry-After", 2 ** attempt + 1)) if e.code in (429, 503) else 2 ** attempt)
        except (urllib.error.URLError, TimeoutError):
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Giving up on {url}")


def read_cache(path: Path) -> dict:
    out = {}
    if path.exists():
        with path.open(encoding="utf-8") as f:
            out = dict(json.loads(line) for line in f)
    return out


def top_artists(tracks_path: str, limit: int) -> pd.DataFrame:
    # Most-listened artists each with the MBID of their most-listened recording
    t = pd.read_parquet(tracks_path).dropna(subset=["artist_name"])
    listeners = t.groupby("artist_name")["n_listeners"].sum().nlargest(limit)
    best = (t[t.track_id.str.match(UUID) & t.artist_name.isin(listeners.index)]
            .sort_values("n_listeners", ascending=False).drop_duplicates("artist_name"))
    return best[["artist_name", "track_id"]].set_index("artist_name").loc[
        [a for a in listeners.index if a in set(best.artist_name)]].reset_index()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tracks", default="data/processed/tracks.parquet")
    p.add_argument("--limit", type=int, default=5000, help="How many top artists to look up")
    p.add_argument("--out", default="app/data/artist_links.parquet")
    p.add_argument("--cache-dir", default="data/raw")
    args = p.parse_args()

    artists = top_artists(args.tracks, args.limit)
    cache_dir = Path(args.cache_dir)
    mbid_cache_path, link_cache_path = cache_dir / "artist_mbids.jsonl", cache_dir / "artist_spotify.jsonl"
    mbids, links = read_cache(mbid_cache_path), read_cache(link_cache_path)  # artist_name -> mbid|None, mbid -> url|None

    # step 1: recording -> artist MBID (100 per request); single-artist credits only, matched by name
    todo = artists[~artists.artist_name.isin(mbids)]
    print(f"{len(artists):,} artists; step 1 (MBIDs): {len(todo):,} to look up")
    with mbid_cache_path.open("a", encoding="utf-8") as f:
        for b in range(0, len(todo), 100):
            batch = todo.iloc[b:b + 100]
            resp = get_json(LB_API + ",".join(batch.track_id)) or {}
            for name, rec in zip(batch.artist_name, batch.track_id):
                credit = (resp.get(rec) or {}).get("artist") or {}
                match = [a["artist_mbid"] for a in credit.get("artists", []) if a["name"].lower() == name.lower()]
                mbids[name] = match[0] if match else None
                f.write(json.dumps([name, mbids[name]]) + "\n")
            f.flush()
            time.sleep(0.5)

    # step 2: artist MBID -> Spotify URL (MusicBrainz allows 1 request/second)
    wanted = [m for m in dict.fromkeys(mbids[a] for a in artists.artist_name) if m]
    todo = [m for m in wanted if m not in links]
    print(f"step 2 (Spotify links): {len(todo):,} to look up, ~{len(todo) / 60:.0f} min")
    start = time.time()
    with link_cache_path.open("a", encoding="utf-8") as f:
        for i, m in enumerate(todo, 1):
            info = get_json(MB_API.format(m)) or {}
            urls = [r["url"]["resource"] for r in info.get("relations", []) if "url" in r]
            links[m] = next((u for u in urls if SPOTIFY_ARTIST.match(u)), None)
            f.write(json.dumps([m, links[m]]) + "\n")
            f.flush()
            if i % 100 == 0 or i == len(todo):
                print(f"  {i:,}/{len(todo):,}  (eta {(time.time() - start) / i * (len(todo) - i) / 60:.0f} min)", flush=True)
            time.sleep(1.05)

    out = pd.DataFrame({"artist_name": artists.artist_name,
                        "spotify_url": [links.get(mbids.get(a) or "") for a in artists.artist_name]}).dropna()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"Saved {len(out):,} Spotify links for {len(artists):,} artists ({len(out) / len(artists):.0%}) to {args.out}")


if __name__ == "__main__":
    main()
