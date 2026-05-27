"""Tests for triplet_miner.evaluation — TripletEvaluator."""

import pytest
import numpy as np

from triplet_miner.evaluation import (
    TripletEvaluator,
    TripletMetrics,
    ClusteringResult,
    EmbeddingReport,
    pairwise_distances,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_clustered(n_per_class: int = 15, n_classes: int = 4, dim: int = 8):
    rng = np.random.RandomState(123)
    embeddings, labels = [], []
    for c in range(n_classes):
        center = rng.randn(dim) * 10
        emb = center + rng.randn(n_per_class, dim) * 0.1
        embeddings.append(emb)
        labels.extend([c] * n_per_class)
    return np.vstack(embeddings).astype(np.float32), np.array(labels)


@pytest.fixture
def clustered():
    return _make_clustered()


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------

class TestInit:
    def test_mismatched_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            TripletEvaluator(np.random.randn(10, 4), np.arange(5))


# ---------------------------------------------------------------------------
# Silhouette score
# ---------------------------------------------------------------------------

class TestSilhouette:
    def test_well_separated_clusters(self, clustered):
        emb, labels = clustered
        ev = TripletEvaluator(emb, labels)
        sil = ev.silhouette_score()
        # Well-separated clusters should have high silhouette
        assert sil > 0.5

    def test_single_cluster(self):
        emb = np.random.randn(20, 4)
        labels = np.zeros(20, dtype=int)
        ev = TripletEvaluator(emb, labels)
        sil = ev.silhouette_score()
        assert sil == 0.0

    def test_single_point(self):
        emb = np.random.randn(1, 4)
        labels = np.array([0])
        ev = TripletEvaluator(emb, labels)
        sil = ev.silhouette_score()
        assert sil == 0.0

    def test_random_embeddings(self):
        rng = np.random.RandomState(0)
        emb = rng.randn(100, 8)
        labels = np.repeat(np.arange(5), 20)
        ev = TripletEvaluator(emb, labels)
        sil = ev.silhouette_score(sample_size=50)
        # Random embeddings → low/negative silhouette
        assert -1.0 <= sil <= 1.0


# ---------------------------------------------------------------------------
# Clustering purity
# ---------------------------------------------------------------------------

class TestClusteringPurity:
    def test_perfect_purity(self, clustered):
        emb, labels = clustered
        ev = TripletEvaluator(emb, labels)
        result = ev.clustering_purity()
        assert isinstance(result, ClusteringResult)
        # Well-separated clusters → perfect or near-perfect purity
        assert result.purity > 0.9

    def test_with_predicted_labels(self):
        labels = np.array([0, 0, 1, 1, 2, 2])
        pred = np.array([0, 0, 1, 1, 1, 1])
        emb = np.random.randn(6, 4)
        ev = TripletEvaluator(emb, labels)
        result = ev.clustering_purity(predicted_labels=pred)
        # Cluster 2 has labels [1,1] → 2 correct, cluster 1 has [2,2] → 2 correct,
        # cluster 0 has [0,0] → 2 correct = 4/6 purity... wait let me recount
        # pred=0: labels=[0,0] → max=2
        # pred=1: labels=[1,1,2,2] → max=2
        # total correct = 2+2 = 4, purity = 4/6
        assert abs(result.purity - 4.0 / 6.0) < 1e-6
        assert result.n_clusters == 2

    def test_cluster_sizes(self, clustered):
        emb, labels = clustered
        ev = TripletEvaluator(emb, labels)
        result = ev.clustering_purity()
        assert sum(result.cluster_sizes.values()) == len(labels)


# ---------------------------------------------------------------------------
# Triplet evaluation
# ---------------------------------------------------------------------------

class TestEvaluateTriplets:
    def test_basic_metrics(self, clustered):
        emb, labels = clustered
        ev = TripletEvaluator(emb, labels)

        # Simple triplets: anchor=0, positive=same class, negative=other class
        anchors = np.array([0])
        positives = np.array([1])  # same class (clustered, same class)
        negatives = np.array([15])  # different class

        metrics = ev.evaluate_triplets(anchors, positives, negatives)
        assert isinstance(metrics, TripletMetrics)
        assert metrics.n_triplets == 1
        assert metrics.avg_anchor_positive_dist > 0
        assert metrics.avg_anchor_negative_dist > 0
        assert metrics.avg_margin >= 0  # well separated
        assert metrics.frac_violations == 0.0

    def test_empty_triplets(self, clustered):
        emb, labels = clustered
        ev = TripletEvaluator(emb, labels)
        metrics = ev.evaluate_triplets(np.array([]), np.array([]), np.array([]))
        assert metrics.n_triplets == 0

    def test_mrr(self, clustered):
        emb, labels = clustered
        ev = TripletEvaluator(emb, labels)
        # When positive is the closest point to anchor, MRR should be high
        anchors = np.array([0])
        positives = np.array([1])
        negatives = np.array([15])
        metrics = ev.evaluate_triplets(anchors, positives, negatives)
        assert metrics.mean_reciprocal_rank > 0


# ---------------------------------------------------------------------------
# Full report
# ---------------------------------------------------------------------------

class TestFullReport:
    def test_without_triplets(self, clustered):
        emb, labels = clustered
        ev = TripletEvaluator(emb, labels)
        report = ev.full_report()
        assert isinstance(report, EmbeddingReport)
        assert report.n_embeddings == len(emb)
        assert report.n_labels == len(np.unique(labels))
        assert report.silhouette > 0.5
        assert report.clustering is not None
        assert report.clustering.purity > 0.9
        assert report.triplet_metrics is None

    def test_with_triplets(self, clustered):
        emb, labels = clustered
        ev = TripletEvaluator(emb, labels)
        report = ev.full_report(
            anchor_indices=np.array([0, 1]),
            positive_indices=np.array([1, 2]),
            negative_indices=np.array([15, 16]),
        )
        assert report.triplet_metrics is not None
        assert report.triplet_metrics.n_triplets == 2

    def test_summary_string(self, clustered):
        emb, labels = clustered
        ev = TripletEvaluator(emb, labels)
        report = ev.full_report(
            anchor_indices=np.array([0]),
            positive_indices=np.array([1]),
            negative_indices=np.array([15]),
        )
        s = report.summary()
        assert "Silhouette" in s
        assert "Purity" in s
        assert "Triplets" in s


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------

class TestDataclasses:
    def test_triplet_metrics_defaults(self):
        m = TripletMetrics()
        assert m.n_triplets == 0
        assert m.avg_margin == 0.0

    def test_clustering_result(self):
        c = ClusteringResult(purity=0.9, n_clusters=3, n_true_labels=4)
        assert c.purity == 0.9
        assert c.n_clusters == 3

    def test_embedding_report(self):
        r = EmbeddingReport(silhouette=0.5, n_embeddings=100, n_labels=5)
        assert r.silhouette == 0.5
