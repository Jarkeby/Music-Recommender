import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from recsys.content import ContentRecommender, build_tag_matrix  # noqa: E402

TAGS = pd.DataFrame({
    "track_id": ["a", "b", "c", "a", "b", "c"],
    "source": ["artist", "artist", "artist", "recording", "recording", "recording"],
    "tag": ["rock", "rock", "jazz", "grunge", "grunge", "bebop"],
    "count": [3, 1, 2, 1, 1, 1],
})


def test_tag_matrix_rows_normalized_and_untagged_rows_empty():
    mat, vocab = build_tag_matrix(TAGS, np.array(["a", "b", "c", "d"]), min_tracks=1)
    assert mat.shape == (4, len(vocab))
    norms = np.sqrt(np.asarray(mat.multiply(mat).sum(axis=1)).ravel())
    np.testing.assert_allclose(norms[:3], 1.0, rtol=1e-5)
    assert norms[3] == 0  # "d" has no tags


def test_min_tracks_drops_rare_tags():
    _, vocab = build_tag_matrix(TAGS, np.array(["a", "b", "c"]), min_tracks=2)
    assert set(vocab) == {"rock", "grunge"}


def test_content_recommends_similar_unplayed_candidates():
    track_ids = np.array(["a", "b", "c"])
    mat, _ = build_tag_matrix(TAGS, track_ids, min_tracks=1)
    train = sp.csr_matrix(np.array([[1.0, 0, 0]], dtype=np.float32))  # one user who played "a"
    rec = ContentRecommender(mat).fit(train)
    assert rec.recommend(0, 1) == [1]  # "b" shares rock + grunge with "a"; "a" itself is excluded
    assert ContentRecommender(mat, candidates=[2]).fit(train).recommend(0, 1) == [2]
