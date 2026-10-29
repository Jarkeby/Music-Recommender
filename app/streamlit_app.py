"""Demo: pick tracks or artists you like, get recommendations from ALS item embeddings.

Run after `run_experiment.py --save-artifacts`:
    streamlit run app/streamlit_app.py

Artists have no embedding of their own, so each one is the popularity-weighted average of its
tracks' embeddings; similar artists are the nearest neighbours of that vector.

TODO (nice upgrades): use FAISS for big catalogs, add a cold-start path using tag features.
Deployed copy uses app/data (int8-quantized embeddings in shards); see scripts/make_deploy_artifacts.py.
"""
import re
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import scipy.sparse as sp
import streamlit as st
from sklearn.preprocessing import normalize
from streamlit_searchbox import st_searchbox

ROOT = Path(__file__).resolve().parents[1]
# app/data holds the slimmed-down copy that ships with the repo (scripts/make_deploy_artifacts.py);
# the full-size artifacts from run_experiment.py are used when present.
DATA_DIR = ROOT / "data/processed" if (ROOT / "data/processed/item_factors.npy").exists() else ROOT / "app/data"
MAX_SUGGESTIONS = 10
# separators between credited artists: "A feat. B", "A & B", "A, B", "A with B", ...
CREDIT_SPLIT = re.compile(r"\s+(?:feat\.?|ft\.?|featuring|with|vs\.?|x)\s+|\s*[&,/•;+]\s*|\s+and\s+")
MIN_ARTIST_TRACKS = 3  # artists with fewer tracks have too little data for a trustworthy vector
# A credit with several names ("A & B") is kept as its own artist, but with fewer than this many tracks
# it is probably a one-off collab, so its score is scaled by n_tracks / COLLAB_FULL_TRACKS.
COLLAB_FULL_TRACKS = 5


@st.cache_resource
def load():
    if (DATA_DIR / "item_factors.npy").exists():
        factors = np.load(DATA_DIR / "item_factors.npy")
    else:  # int8 shards from make_deploy_artifacts.py
        factors = np.concatenate([np.load(f) for f in sorted(DATA_DIR.glob("item_factors_*.npy"))]).astype(np.float32)
    tracks = pd.read_parquet(DATA_DIR / "tracks.parquet")
    norm = factors / (np.linalg.norm(factors, axis=1, keepdims=True) + 1e-9)
    has_meta = "artist_name" in tracks
    if has_meta:
        tracks["label"] = tracks["artist_name"].fillna("Unknown artist") + " – " + tracks["recording_name"].fillna(
            tracks["track_id"])
    else:  # artifacts saved before metadata was added
        tracks["label"] = tracks["track_id"]
    weight = tracks["n_listeners"].to_numpy(dtype=np.float32) if "n_listeners" in tracks else np.ones(len(tracks), np.float32)
    track_order = np.argsort(-weight)  # most-listened first, so search results come back in popularity order

    artists = artist_vectors = None
    if has_meta:
        known = tracks[tracks["artist_name"].notna()]
        codes, names = pd.factorize(known["artist_name"])
        by_artist = sp.csr_matrix((weight[known.index], (codes, known["track_idx"].to_numpy())),
                                  shape=(len(names), len(tracks)))
        artists = pd.DataFrame({"name": names, "listeners": np.asarray(by_artist.sum(axis=1)).ravel().astype(int),
                                "n_tracks": np.diff(by_artist.indptr)})
        artists["lowered"] = artists["name"].str.lower()
        by_popularity = np.argsort(-artists["listeners"].to_numpy())
        artists = artists.iloc[by_popularity].reset_index(drop=True)
        artist_vectors = normalize(by_artist[by_popularity] @ norm).astype(np.float32)
        artists["credits"] = [[c.strip() for c in CREDIT_SPLIT.split(n)] for n in artists["lowered"]]
        is_multi = np.array([len(c) > 1 for c in artists["credits"]])  # looks like a collab (or a duo/band with "&")
        artists["weight"] = np.where(is_multi, np.minimum(1.0, artists["n_tracks"] / COLLAB_FULL_TRACKS), 1.0)
    return norm, tracks, track_order, tracks["label"].str.lower(), artists, artist_vectors


def search_in(lowered: pd.Series, order, labels, text: str):
    """Autocomplete: entries whose text contains every typed word, in `order` (most-listened first)."""
    hit = np.ones(len(lowered), dtype=bool)
    for word in text.lower().split():
        hit &= lowered.str.contains(word, regex=False).to_numpy()
    return [(labels[i], int(i)) for i in order[hit[order]][:MAX_SUGGESTIONS]]


def picker(kind: str, label: str, placeholder: str, search_fn, names):
    """A search box that adds picks to a removable list in session state; returns the picked indices."""
    picks = st.session_state.setdefault(f"picked_{kind}", [])
    box = f"search_{kind}"

    def add(choice):  # submit_function: runs once per new selection, not on every rerun
        if choice not in picks:
            picks.append(choice)

    st_searchbox(search_fn, label=label, placeholder=placeholder, clear_on_submit=True, key=box,
                 submit_function=add)
    for i in list(picks):
        if st.button(f"✕  {names[i]}", key=f"remove_{kind}_{i}"):
            picks.remove(i)
            if box in st.session_state:  # forget the last selection so the same pick can be re-added
                st.session_state[box]["result"] = None
            st.rerun()
    return picks


st.title("Music recommender")
norm, tracks, track_order, track_lowered, artists, artist_vectors = load()
track_labels = tracks["label"].to_numpy()

tab_tracks, *tab_rest = st.tabs(["Tracks", "Artists"] if artists is not None else ["Tracks"])

with tab_tracks:
    picked = picker("track", "Search for a track you like (artist or title)", "e.g. radiohead creep",
                    lambda text: search_in(track_lowered, track_order, track_labels, text), track_labels)
    if picked:
        scores = norm @ norm[picked].mean(axis=0)
        scores[picked] = -np.inf
        top = np.argsort(-scores)[:10]
        st.subheader("Recommended tracks")
        rec = tracks.iloc[top]
        st.dataframe(
            pd.DataFrame({
                "Artist": rec.get("artist_name", rec["track_id"]),
                "Title": rec.get("recording_name", rec["track_id"]),
                "Score": scores[top].round(3),
            }),
            hide_index=True,
        )

if tab_rest:
    with tab_rest[0]:
        artist_names = artists["name"].to_numpy()
        picked_artists = picker("artist", "Search for an artist you like", "e.g. radiohead",
                                lambda text: search_in(artists["lowered"], np.arange(len(artists)), artist_names, text),
                                artist_names)
        if picked_artists:
            scores = artist_vectors @ artist_vectors[picked_artists].mean(axis=0)
            scores[artists["n_tracks"].to_numpy() < MIN_ARTIST_TRACKS] = -np.inf
            scores = np.where(scores > 0, scores * artists["weight"].to_numpy(), scores)  # soften thin collabs
            # hide the picked artists and every credit they appear in ("X feat. Y", "Y / X")
            picked_names = set(artists["lowered"].to_numpy()[picked_artists])
            scores[[bool(picked_names.intersection(c)) for c in artists["credits"]]] = -np.inf
            top = np.argsort(-scores)[:10]
            st.subheader("Similar artists")
            names = artist_names[top]
            st.dataframe(
                pd.DataFrame({
                    "Artist": names,
                    "Similarity": scores[top].round(3),
                    "Tracks in catalog": artists["n_tracks"].to_numpy()[top],
                    "Spotify": [f"https://open.spotify.com/search/{quote(n)}/artists" for n in names],
                }),
                column_config={"Spotify": st.column_config.LinkColumn("Spotify", display_text="Open")},
                hide_index=True,
            )
