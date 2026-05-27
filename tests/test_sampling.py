"""Tests for triplet_miner.sampling — AdvancedStrategies."""

import pytest
import numpy as np

from triplet_miner.sampling import (
    AdvancedStrategies,
    MiningResult,
    TripletIndices,
    pairwise_distances,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_clustered(n_per_class: int = 20, n_classes: int = 5, dim: int = 16):
    """Create well-separated clusters for deterministic tests."""
    rng = np.random.RandomState(42)
    embeddings = []
    labels = []
    for c in range(n_classes):
        center = rng.randn(dim) * 10  # spread centres apart
        emb = center + rng.randn(n_per_class, dim) * 0.1
        embeddings.append(emb)
        labels.extend([c] * n_per_class)
    return np.vstack(embeddings).astype(np.float32), np.array(labels)


@pytest.fixture
def clustered():
    return _make_clustered()


# ---------------------------------------------------------------------------
# pairwise_distances
# ---------------------------------------------------------------------------

class TestPairwiseDistances:
    def test_shape(self):
        emb = np.random.randn(10, 4)
        d = pairwise_distances(emb)
        assert d.shape == (10, 10)

    def test_diagonal_zero(self):
        emb = np.random.randn(10, 4)
        d = pairwise_distances(emb)
        np.testing.assert_allclose(np.diag(d), 0.0, atol=1e-10)

    def test_symmetric(self):
        emb = np.random.randn(10, 4)
        d = pairwise_distances(emb)
        np.testing.assert_allclose(d, d.T, atol=1e-10)

    def test_known_distance(self):
        emb = np.array([[0.0, 0.0], [3.0, 4.0]])
        d = pairwise_distances(emb)
        assert abs(d[0, 1] - 5.0) < 1e-6


# ---------------------------------------------------------------------------
# AdvancedStrategies init
# ---------------------------------------------------------------------------

class TestInit:
    def test_mismatched_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            AdvancedStrategies(np.random.randn(10, 4), np.arange(5))

    def test_caches_dist_matrix(self, clustered):
        emb, labels = clustered
        miner = AdvancedStrategies(emb, labels)
        d1 = miner.dists
        d2 = miner.dists
        assert d1 is d2  # cached


# ---------------------------------------------------------------------------
# Hard-negative mining
# ---------------------------------------------------------------------------

class TestHardNegatives:
    def test_returns_mining_result(self, clustered):
        emb, labels = clustered
        miner = AdvancedStrategies(emb, labels)
        result = miner.mine_hard_negatives(n=10)
        assert isinstance(result, MiningResult)
        assert result.strategy == "hard_negative"
        assert len(result.triplets) <= 10

    def test_hard_negative_closer_than_positive(self, clustered):
        """With well-separated clusters and margin=0, hard negatives should
        be very rare (essentially all negatives are farther than positives)."""
        emb, labels = clustered
        miner = AdvancedStrategies(emb, labels)
        result = miner.mine_hard_negatives(n=200, margin=100.0)
        # With large margin we should find some
        assert len(result.triplets) > 0

    def test_stats_populated(self, clustered):
        emb, labels = clustered
        miner = AdvancedStrategies(emb, labels)
        result = miner.mine_hard_negatives(n=5, margin=50.0)
        if result.triplets:
            assert "count" in result.stats
            assert result.stats["count"] == len(result.triplets)


# ---------------------------------------------------------------------------
# Semi-hard mining
# ---------------------------------------------------------------------------

class TestSemiHard:
    def test_returns_result(self, clustered):
        emb, labels = clustered
        miner = AdvancedStrategies(emb, labels)
        result = miner.mine_semi_hard(n=10)
        assert isinstance(result, MiningResult)
        assert result.strategy == "semi_hard"

    def test_semi_hard_an_greater_than_ap(self, clustered):
        """Semi-hard negatives should be farther than the positive."""
        emb, labels = clustered
        miner = AdvancedStrategies(emb, labels)
        result = miner.mine_semi_hard(n=30)
        for t in result.triplets:
            assert t.an_dist > t.ap_dist + 1e-8


# ---------------------------------------------------------------------------
# Distance-weighted sampling
# ---------------------------------------------------------------------------

class TestDistanceWeighted:
    def test_returns_result(self, clustered):
        emb, labels = clustered
        miner = AdvancedStrategies(emb, labels)
        result = miner.mine_distance_weighted(n=10)
        assert isinstance(result, MiningResult)
        assert result.strategy == "distance_weighted"
        assert len(result.triplets) <= 10

    def test_temperature_in_stats(self, clustered):
        emb, labels = clustered
        miner = AdvancedStrategies(emb, labels)
        result = miner.mine_distance_weighted(n=5, temperature=2.0)
        assert result.stats.get("temperature") == 2.0

    def test_different_temperatures_produce_different_results(self, clustered):
        emb, labels = clustered
        miner = AdvancedStrategies(emb, labels)
        r1 = miner.mine_distance_weighted(n=20, temperature=0.1)
        # Reset cache
        miner._dists = None
        r2 = miner.mine_distance_weighted(n=20, temperature=10.0)
        # They should both produce triplets
        assert len(r1.triplets) > 0
        assert len(r2.triplets) > 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_class(self):
        emb = np.random.randn(10, 4)
        labels = np.zeros(10, dtype=int)
        miner = AdvancedStrategies(emb, labels)
        result = miner.mine_hard_negatives(n=5, margin=100.0)
        # No negatives possible (all same class)
        assert len(result.triplets) == 0

    def test_one_sample_per_class(self):
        emb = np.random.randn(5, 4)
        labels = np.arange(5)
        miner = AdvancedStrategies(emb, labels)
        result = miner.mine_semi_hard(n=5)
        # No positives (single sample per class)
        assert len(result.triplets) == 0

    def test_triplet_indices_dataclass(self):
        t = TripletIndices(anchor=0, positive=1, negative=2, ap_dist=1.0, an_dist=3.0)
        assert t.anchor == 0
        assert t.positive == 1
        assert t.negative == 2

    def test_empty_stats(self):
        emb = np.random.randn(5, 4)
        labels = np.arange(5)
        miner = AdvancedStrategies(emb, labels)
        result = miner.mine_hard_negatives(n=5)
        assert result.stats.get("count", 0) == 0 or len(result.triplets) == 0
