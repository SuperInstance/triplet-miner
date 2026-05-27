"""Evaluation metrics for triplet quality and embedding spaces.

Provides :class:`TripletEvaluator` which computes embedding quality metrics
(silhouette score, clustering purity) and per-triplet quality diagnostics.

Only requires numpy (optional dependency).  Pure-numpy approximations are
used so that scikit-learn is not required.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]


def _require_numpy() -> None:
    if np is None:
        raise ImportError(
            "numpy is required for evaluation metrics. "
            "Install with: pip install triplet-miner[numpy]"
        )


def pairwise_distances(embeddings: "np.ndarray") -> "np.ndarray":
    """Compute the full pairwise Euclidean distance matrix.

    Args:
        embeddings: (N, D) array of D-dimensional embeddings.

    Returns:
        (N, N) symmetric distance matrix with zeros on the diagonal.
    """
    dot = embeddings @ embeddings.T
    norms_sq = np.diag(dot).copy()
    dists_sq = norms_sq[:, None] + norms_sq[None, :] - 2.0 * dot
    np.maximum(dists_sq, 0.0, out=dists_sq)
    return np.sqrt(dists_sq)


# ---------------------------------------------------------------------------
# Metric dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TripletMetrics:
    """Metrics for a set of triplets.

    Attributes:
        n_triplets: Number of triplets evaluated.
        avg_anchor_positive_dist: Mean anchor-positive distance.
        avg_anchor_negative_dist: Mean anchor-negative distance.
        avg_margin: Mean margin (an_dist - ap_dist).
        frac_violations: Fraction of triplets violating margin (an_dist <= ap_dist).
        mean_reciprocal_rank: MRR of positives against negatives.
    """

    n_triplets: int = 0
    avg_anchor_positive_dist: float = 0.0
    avg_anchor_negative_dist: float = 0.0
    avg_margin: float = 0.0
    frac_violations: float = 0.0
    mean_reciprocal_rank: float = 0.0


@dataclass
class ClusteringResult:
    """Clustering purity metrics.

    Attributes:
        purity: Clustering purity (0–1).
        n_clusters: Number of unique predicted labels.
        n_true_labels: Number of unique true labels.
        cluster_sizes: Mapping from predicted cluster id to count.
    """

    purity: float = 0.0
    n_clusters: int = 0
    n_true_labels: int = 0
    cluster_sizes: Dict[int, int] = field(default_factory=dict)


@dataclass
class EmbeddingReport:
    """Full evaluation report for an embedding space.

    Attributes:
        silhouette: Approximate silhouette score (-1 to 1).
        clustering: Clustering purity result.
        triplet_metrics: Triplet-level quality metrics.
        n_embeddings: Total number of embeddings.
        n_labels: Number of unique labels.
    """

    silhouette: float = 0.0
    clustering: Optional[ClusteringResult] = None
    triplet_metrics: Optional[TripletMetrics] = None
    n_embeddings: int = 0
    n_labels: int = 0

    def summary(self) -> str:
        lines = [
            f"Embedding Report: {self.n_embeddings} embeddings, {self.n_labels} labels",
            f"  Silhouette:      {self.silhouette:.4f}",
        ]
        if self.clustering:
            c = self.clustering
            lines.append(f"  Purity:          {c.purity:.4f} ({c.n_clusters} clusters)")
        if self.triplet_metrics:
            t = self.triplet_metrics
            lines.append(f"  Triplets:        {t.n_triplets}")
            lines.append(f"  Avg AP dist:     {t.avg_anchor_positive_dist:.4f}")
            lines.append(f"  Avg AN dist:     {t.avg_anchor_negative_dist:.4f}")
            lines.append(f"  Avg Margin:      {t.avg_margin:.4f}")
            lines.append(f"  Violations:      {t.frac_violations:.2%}")
            lines.append(f"  MRR:             {t.mean_reciprocal_rank:.4f}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# TripletEvaluator
# ---------------------------------------------------------------------------

class TripletEvaluator:
    """Evaluate embedding quality using triplet metrics and clustering.

    Example::

        import numpy as np
        from triplet_miner.evaluation import TripletEvaluator

        emb = np.random.randn(200, 32).astype(np.float32)
        labels = np.repeat(np.arange(10), 20)

        ev = TripletEvaluator(emb, labels)
        print(ev.silhouette_score())
        print(ev.clustering_purity())
        report = ev.full_report()
        print(report.summary())
    """

    def __init__(
        self,
        embeddings: "np.ndarray",
        labels: "np.ndarray",
    ) -> None:
        _require_numpy()
        self.embeddings = np.asarray(embeddings, dtype=np.float64)
        self.labels = np.asarray(labels)
        if self.embeddings.shape[0] != self.labels.shape[0]:
            raise ValueError("embeddings and labels must have the same length")
        self._dists: Optional["np.ndarray"] = None

    @property
    def dists(self) -> "np.ndarray":
        if self._dists is None:
            self._dists = pairwise_distances(self.embeddings)
        return self._dists

    # -- Core metrics --------------------------------------------------------

    def silhouette_score(self, sample_size: int = 2000) -> float:
        """Compute an approximate silhouette score.

        For each sample point *i*, the silhouette is::

            s(i) = (b(i) - a(i)) / max(a(i), b(i))

        where *a(i)* is the mean intra-cluster distance and *b(i)* is the
        mean nearest-cluster distance.

        Uses random sub-sampling when N > *sample_size* to keep it fast.

        Args:
            sample_size: Maximum number of points to evaluate.

        Returns:
            Mean silhouette score (−1 to 1).
        """
        n = len(self.embeddings)
        if n <= 1:
            return 0.0

        unique_labels = np.unique(self.labels)
        if len(unique_labels) <= 1:
            return 0.0

        indices = np.arange(n)
        if n > sample_size:
            indices = np.random.choice(indices, sample_size, replace=False)

        silhouettes: List[float] = []
        dists = self.dists

        for i in indices:
            i = int(i)
            same = self.labels == self.labels[i]
            same[i] = False
            if not np.any(same):
                continue

            # a(i): mean distance to same-cluster points
            a = float(np.mean(dists[i, same]))

            # b(i): smallest mean distance to other clusters
            best_b = float("inf")
            for lbl in unique_labels:
                if lbl == self.labels[i]:
                    continue
                other = self.labels == lbl
                if not np.any(other):
                    continue
                b_candidate = float(np.mean(dists[i, other]))
                best_b = min(best_b, b_candidate)

            denom = max(a, best_b)
            silhouettes.append((best_b - a) / denom if denom > 0 else 0.0)

        return float(np.mean(silhouettes)) if silhouettes else 0.0

    def clustering_purity(self, predicted_labels: Optional["np.ndarray"] = None) -> ClusteringResult:
        """Compute clustering purity.

        If *predicted_labels* is ``None``, uses :func:`_nearest_centroid_labels`
        to derive cluster assignments from the embeddings.

        Purity is computed as::

            purity = (1/N) * sum_k max_j |cluster_k ∩ class_j|

        Args:
            predicted_labels: Optional cluster assignments.  Derived from
                nearest-centroid clustering if not provided.

        Returns:
            A :class:`ClusteringResult`.
        """
        if predicted_labels is None:
            predicted_labels = _nearest_centroid_labels(self.embeddings, self.labels)

        pred = np.asarray(predicted_labels)
        true = self.labels
        n = len(true)

        cluster_sizes: Dict[int, int] = {}
        correct = 0
        for k in np.unique(pred):
            mask = pred == k
            cluster_sizes[int(k)] = int(mask.sum())
            # Most common true label in this cluster
            counts: Dict[int, int] = {}
            for lbl in true[mask]:
                lbl = int(lbl)
                counts[lbl] = counts.get(lbl, 0) + 1
            correct += max(counts.values())

        purity = correct / n if n > 0 else 0.0
        return ClusteringResult(
            purity=purity,
            n_clusters=int(len(np.unique(pred))),
            n_true_labels=int(len(np.unique(true))),
            cluster_sizes=cluster_sizes,
        )

    def evaluate_triplets(
        self,
        anchor_indices: "np.ndarray",
        positive_indices: "np.ndarray",
        negative_indices: "np.ndarray",
    ) -> TripletMetrics:
        """Evaluate triplet quality from index arrays.

        Args:
            anchor_indices: (T,) array of anchor indices.
            positive_indices: (T,) array of positive indices.
            negative_indices: (T,) array of negative indices.

        Returns:
            A :class:`TripletMetrics`.
        """
        anchors = np.asarray(anchor_indices)
        positives = np.asarray(positive_indices)
        negatives = np.asarray(negative_indices)
        dists = self.dists

        n = len(anchors)
        if n == 0:
            return TripletMetrics()

        ap_dists = np.array([dists[int(anchors[i]), int(positives[i])] for i in range(n)])
        an_dists = np.array([dists[int(anchors[i]), int(negatives[i])] for i in range(n)])

        margins = an_dists - ap_dists
        violations = int(np.sum(an_dists <= ap_dists))

        # MRR: for each anchor, rank the positive among all candidates
        mrr_sum = 0.0
        for i in range(n):
            a_idx = int(anchors[i])
            p_idx = int(positives[i])
            pos_dist = dists[a_idx, p_idx]
            # Count how many negatives are closer
            all_dists = dists[a_idx]
            # Rank of positive (1-indexed)
            rank = int(np.sum(all_dists < pos_dist)) + 1
            mrr_sum += 1.0 / rank

        return TripletMetrics(
            n_triplets=n,
            avg_anchor_positive_dist=float(np.mean(ap_dists)),
            avg_anchor_negative_dist=float(np.mean(an_dists)),
            avg_margin=float(np.mean(margins)),
            frac_violations=violations / n,
            mean_reciprocal_rank=mrr_sum / n,
        )

    def full_report(
        self,
        anchor_indices: Optional["np.ndarray"] = None,
        positive_indices: Optional["np.ndarray"] = None,
        negative_indices: Optional["np.ndarray"] = None,
        silhouette_sample: int = 2000,
    ) -> EmbeddingReport:
        """Run all evaluations and return a comprehensive report.

        Args:
            anchor_indices: Optional anchor indices for triplet evaluation.
            positive_indices: Optional positive indices for triplet evaluation.
            negative_indices: Optional negative indices for triplet evaluation.
            silhouette_sample: Sample size for silhouette computation.

        Returns:
            An :class:`EmbeddingReport` with all metrics populated.
        """
        sil = self.silhouette_score(sample_size=silhouette_sample)
        clustering = self.clustering_purity()

        tm = None
        if anchor_indices is not None and positive_indices is not None and negative_indices is not None:
            tm = self.evaluate_triplets(anchor_indices, positive_indices, negative_indices)

        return EmbeddingReport(
            silhouette=sil,
            clustering=clustering,
            triplet_metrics=tm,
            n_embeddings=len(self.embeddings),
            n_labels=int(len(np.unique(self.labels))),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nearest_centroid_labels(
    embeddings: "np.ndarray", labels: "np.ndarray"
) -> "np.ndarray":
    """Assign each point to the nearest class centroid."""
    unique_labels = np.unique(labels)
    centroids = np.array([
        embeddings[labels == lbl].mean(axis=0) for lbl in unique_labels
    ])
    # Distance from each point to each centroid
    # (N, D) x (K, D)^T → (N, K) via broadcasting
    # dist^2 = sum(x^2) + sum(c^2) - 2*x.c
    x2 = np.sum(embeddings ** 2, axis=1, keepdims=True)  # (N, 1)
    c2 = np.sum(centroids ** 2, axis=1, keepdims=True).T  # (1, K)
    dists_sq = x2 + c2 - 2.0 * (embeddings @ centroids.T)
    nearest = np.argmin(dists_sq, axis=1)
    return unique_labels[nearest]
