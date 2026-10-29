"""Recommenders share one interface: fit(train_matrix) then recommend(user_idx, k) -> list of track idx.

Add your own (BPR, content-based, hybrid, ...) by following the same pattern.
"""
import numpy as np


class PopularityRecommender:
    """Recommend the most-played tracks the user hasn't heard. Hard to beat -- always include it."""

    name = "popularity"

    def fit(self, train_matrix):
        self.train = train_matrix.tocsr()
        self.order = np.argsort(-np.asarray(self.train.sum(axis=0)).ravel())
        return self

    def recommend(self, user_idx: int, k: int):
        seen = set(self.train[user_idx].indices)
        out = []
        for t in self.order:
            if t not in seen:
                out.append(int(t))
                if len(out) == k:
                    break
        return out


class _ImplicitWrapper:
    """Shared logic for models from the `implicit` library."""

    def _recommend(self, user_idx: int, k: int):
        ids, _ = self.model.recommend(user_idx, self.train[user_idx], N=k, filter_already_liked_items=True)
        return [int(i) for i in ids]

    def recommend(self, user_idx: int, k: int):
        return self._recommend(user_idx, k)

    def recommend_batch(self, user_idxs, k: int):
        """Recommend for many users in one call (much faster than looping for evaluation)."""
        user_idxs = np.asarray(user_idxs)
        ids, _ = self.model.recommend(user_idxs, self.train[user_idxs], N=k, filter_already_liked_items=True)
        return [[int(i) for i in row] for row in ids]


class ItemKNNRecommender(_ImplicitWrapper):
    """Item-item cosine similarity: 'people who played X also played Y'."""

    name = "item_knn"

    def __init__(self, k_neighbors: int = 50):
        self.k_neighbors = k_neighbors

    def fit(self, train_matrix):
        from implicit.nearest_neighbours import CosineRecommender
        self.train = train_matrix.tocsr()
        self.model = CosineRecommender(K=self.k_neighbors)
        self.model.fit(self.train, show_progress=False)
        return self


class ALSRecommender(_ImplicitWrapper):
    """Matrix factorization via alternating least squares (implicit feedback)."""

    name = "als"

    def __init__(self, factors: int = 64, regularization: float = 0.01, iterations: int = 15):
        self.factors, self.regularization, self.iterations = factors, regularization, iterations

    def fit(self, train_matrix):
        from implicit.als import AlternatingLeastSquares
        self.train = train_matrix.tocsr()
        self.model = AlternatingLeastSquares(
            factors=self.factors, regularization=self.regularization,
            iterations=self.iterations, random_state=0,
        )
        self.model.fit(self.train, show_progress=False)
        return self

    @property
    def item_factors(self):
        f = self.model.item_factors
        return f.to_numpy() if hasattr(f, "to_numpy") else np.asarray(f)

    @property
    def user_factors(self):
        f = self.model.user_factors
        return f.to_numpy() if hasattr(f, "to_numpy") else np.asarray(f)


class BPRRecommender(_ImplicitWrapper):
    """Bayesian Personalized Ranking: learns embeddings by ranking played tracks above unplayed ones.

    Unlike ALS it ignores play counts (any play = positive) and optimizes ranking directly.
    """

    name = "bpr"

    def __init__(self, factors: int = 64, learning_rate: float = 0.01, regularization: float = 0.01,
                 iterations: int = 100):
        self.factors, self.learning_rate = factors, learning_rate
        self.regularization, self.iterations = regularization, iterations

    def fit(self, train_matrix):
        from implicit.bpr import BayesianPersonalizedRanking
        self.train = train_matrix.tocsr()
        self.model = BayesianPersonalizedRanking(
            factors=self.factors, learning_rate=self.learning_rate, regularization=self.regularization,
            iterations=self.iterations, random_state=0,
        )
        self.model.fit(self.train, show_progress=False)
        return self
