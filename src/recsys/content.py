"""Content features from tags, and the recommenders that use them (cold start + hybrid).

Same interface as models.py: fit(train_matrix), then recommend_batch(user_idxs, k). Each also has
score(user_idxs) -> dense users x tracks matrix, so scores can be blended.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import normalize

# artist tags cover the most tracks but say the least about one song, so they count for less
SOURCE_WEIGHTS = {"recording": 1.0, "release_group": 0.7, "artist": 0.5}


def build_tag_matrix(tags: pd.DataFrame, track_ids, min_tracks: int = 5, max_tags: int = 5000,
                     source_weights: dict = SOURCE_WEIGHTS):
    """tracks x tags TF-IDF matrix (rows L2-normalized; untagged tracks are all-zero rows).

    tags: track_id | source | tag | count (from scripts/pull_tags.py). A tag's weight on a track is
    sum over sources of source_weight * log1p(count), then multiplied by the tag's IDF.
    """
    t_map = pd.Series(np.arange(len(track_ids)), index=track_ids)
    tags = tags[tags.track_id.isin(t_map.index)]
    tags = tags.assign(w=tags.source.map(source_weights) * np.log1p(tags["count"]))
    tf = tags.groupby(["track_id", "tag"], as_index=False)["w"].sum()

    df = tf.groupby("tag").size()
    vocab = df[df >= min_tracks].nlargest(max_tags).index
    tf = tf[tf.tag.isin(vocab)]
    tag_map = pd.Series(np.arange(len(vocab)), index=vocab)
    idf = np.log(len(track_ids) / df[vocab].to_numpy())

    mat = sp.csr_matrix((tf.w.to_numpy() * idf[tag_map[tf.tag].to_numpy()],
                         (t_map[tf.track_id].to_numpy(), tag_map[tf.tag].to_numpy())),
                        shape=(len(track_ids), len(vocab)), dtype=np.float32)
    return normalize(mat), np.asarray(vocab)


def _top_k(scores: np.ndarray, k: int, exclude: sp.csr_matrix | None = None, candidates=None):
    """Top-k column indices per row, skipping excluded (already played) and non-candidate items."""
    scores = scores.copy()
    if exclude is not None:
        r, c = exclude.nonzero()
        scores[r, c] = -np.inf
    if candidates is not None:
        mask = np.ones(scores.shape[1], dtype=bool)
        mask[candidates] = False
        scores[:, mask] = -np.inf
    top = np.argpartition(-scores, k, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(scores, top, axis=1), axis=1)
    return [[int(i) for i in row] for row in np.take_along_axis(top, order, axis=1)]


class _Scorer:
    """recommend_batch on top of score(); `candidates` restricts what can be recommended."""

    candidates = None

    def recommend_batch(self, user_idxs, k: int):
        user_idxs = np.asarray(user_idxs)
        return _top_k(self.score(user_idxs), k, self.train[user_idxs], self.candidates)

    def recommend(self, user_idx: int, k: int):
        return self.recommend_batch([user_idx], k)[0]


class ContentRecommender(_Scorer):
    """User profile = play-weighted average of the tag vectors of their tracks; score = cosine to it.

    Needs no plays of the candidate track, so it works for tracks the model has never seen.
    """

    name = "content"

    def __init__(self, item_features: sp.csr_matrix, candidates=None):
        self.features, self.candidates = item_features, candidates

    def fit(self, train_matrix):
        self.train = train_matrix.tocsr()
        return self

    def score(self, user_idxs):
        profiles = normalize(self.train[user_idxs] @ self.features).toarray()
        return np.asarray((self.features @ profiles.T).T)


class TagToALSRecommender(_Scorer):
    """Cold start through the ALS space: learn a linear map tags -> ALS item factors on warm tracks
    (ridge regression), predict factors for cold tracks, then score them with ALS user factors."""

    name = "tag_to_als"

    def __init__(self, als, item_features: sp.csr_matrix, warm, candidates=None, reg: float = 1.0):
        self.als, self.features, self.warm = als, item_features, np.asarray(warm)
        self.candidates, self.reg = candidates, reg

    def fit(self, train_matrix):
        self.train = train_matrix.tocsr()
        X = self.features[self.warm]
        Y = self.als.item_factors[self.warm]
        gram = (X.T @ X).toarray() + self.reg * np.eye(X.shape[1], dtype=np.float32)
        W = np.linalg.solve(gram, np.asarray(X.T @ Y))
        self.item_factors = np.asarray(self.features @ W, dtype=np.float32)
        return self

    def score(self, user_idxs):
        return self.als.user_factors[user_idxs] @ self.item_factors.T


def _standardize(s: np.ndarray) -> np.ndarray:
    """Per-user z-score, so ALS and content scores are on the same scale before blending."""
    return (s - s.mean(axis=1, keepdims=True)) / (s.std(axis=1, keepdims=True) + 1e-9)


class HybridRecommender(_Scorer):
    """weight * z(ALS score) + (1 - weight) * z(content score). Both models must already be fit."""

    name = "hybrid"

    def __init__(self, als, content: ContentRecommender, weight: float = 0.8):
        self.als, self.content, self.weight = als, content, weight

    def fit(self, train_matrix):
        self.train = train_matrix.tocsr()
        return self

    def score(self, user_idxs):
        als = self.als.user_factors[user_idxs] @ self.als.item_factors.T
        return self.weight * _standardize(als) + (1 - self.weight) * _standardize(self.content.score(user_idxs))
