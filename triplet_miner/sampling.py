"""Advanced sampling strategies for triplet mining.

Provides embedding-aware mining strategies including hard-negative mining,
semi-hard mining, and distance-weighted sampling. These operate on
pre-computed embeddings (numpy arrays) and are useful for training
contrastive / metric-learning models.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]


def _require_numpy() -> None:
    if np is None:
        raise ImportError(
            "numpy is required for advanced sampling. "
            "Install with: pip install triplet-miner[numpy]"
        )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def pairwise_distances(embeddings: "np.ndarray") -> "np.ndarray":
    """Compute the full pairwise Euclidean distance matrix.

    Args:
        embeddings: (N, D) array of D-dimensional embeddings.

    Returns:
        (N, N) symmetric distance matrix with zeros on the diagonal.
    """
    _require_numpy()
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a.b
    dot = embeddings @ embeddings.T
    norms_sq = np.diag(dot).copy()
    dists_sq = norms_sq[:, None] + norms_sq[None, :] - 2.0 * dot
    # Clamp small negatives from floating-point error
    np.maximum(dists_sq, 0.0, out=dists_sq)
    return np.sqrt(dists_sq)


def _positive_mask(labels: "np.ndarray") -> "np.ndarray":
    """Return (N, N) boolean mask where mask[i, j] is True if labels match."""
    return labels[:, None] == labels[None, :]


# ---------------------------------------------------------------------------
# Dataclasses for results
# ---------------------------------------------------------------------------

@dataclass
class TripletIndices:
    """Indices into an embedding matrix that form a triplet.

    Attributes:
        anchor: Row index of the anchor.
        positive: Row index of the positive.
        negative: Row index of the negative.
        ap_dist: Anchor-positive distance.
        an_dist: Anchor-negative distance.
    """

    anchor: int
    positive: int
    negative: int
    ap_dist: float = 0.0
    an_dist: float = 0.0


@dataclass
class MiningResult:
    """Result from an advanced mining run.

    Attributes:
        triplets: List of mined triplet index triples.
        strategy: Name of the strategy used.
        stats: Diagnostic statistics (counts, average distances, etc.).
    """

    triplets: List[TripletIndices] = field(default_factory=list)
    strategy: str = ""
    stats: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# AdvancedStrategies
# ---------------------------------------------------------------------------

class AdvancedStrategies:
    """Embedding-aware triplet mining strategies.

    All methods expect *embeddings* to be a (N, D) numpy array and *labels*
    to be a length-N array of integer class labels.  Positive pairs share
    the same label; negatives come from different labels.

    Example::

        import numpy as np
        from triplet_miner.sampling import AdvancedStrategies

        emb = np.random.randn(100, 64).astype(np.float32)
        labels = np.repeat(np.arange(10), 10)

        miner = AdvancedStrategies(emb, labels)
        result = miner.mine_hard_negatives(n=50)
        for t in result.triplets:
            print(t.anchor, t.positive, t.negative, t.an_dist)
    """

    def __init__(self, embeddings: "np.ndarray", labels: "np.ndarray") -> None:
        _require_numpy()
        self.embeddings = np.asarray(embeddings, dtype=np.float64)
        self.labels = np.asarray(labels)
        if self.embeddings.shape[0] != self.labels.shape[0]:
            raise ValueError("embeddings and labels must have the same length")
        self._dists: Optional["np.ndarray"] = None
        self._pos_mask: Optional["np.ndarray"] = None

    # -- cached properties ---------------------------------------------------

    @property
    def dists(self) -> "np.ndarray":
        if self._dists is None:
            self._dists = pairwise_distances(self.embeddings)
        return self._dists

    @property
    def pos_mask(self) -> "np.ndarray":
        if self._pos_mask is None:
            self._pos_mask = _positive_mask(self.labels)
            np.fill_diagonal(self._pos_mask, False)
        return self._pos_mask

    # -- public API ----------------------------------------------------------

    def mine_hard_negatives(
        self,
        n: int = 100,
        margin: float = 0.0,
    ) -> MiningResult:
        """Mine hard-negative triplets.

        A triplet (a, p, n) is *hard* when ``d(a, n) < d(a, p) + margin``,
        i.e. the negative is closer to the anchor than the positive.

        Args:
            n: Maximum number of triplets to return.
            margin: Extra margin applied to the hard-negative condition.

        Returns:
            A :class:`MiningResult` with up to *n* triplets.
        """
        triplets = self._collect_triplets(
            selector=self._hard_selector(margin=margin),
            max_count=n,
        )
        return MiningResult(
            triplets=triplets,
            strategy="hard_negative",
            stats=self._compute_stats(triplets),
        )

    def mine_semi_hard(
        self,
        n: int = 100,
    ) -> MiningResult:
        """Mine semi-hard triplets.

        Semi-hard negatives satisfy ``d(a, p) < d(a, n)`` — they are farther
        than the positive but still relatively close, making them informative
        without being overwhelming.

        Args:
            n: Maximum number of triplets to return.

        Returns:
            A :class:`MiningResult` with up to *n* triplets.
        """
        triplets = self._collect_triplets(
            selector=self._semi_hard_selector(),
            max_count=n,
        )
        return MiningResult(
            triplets=triplets,
            strategy="semi_hard",
            stats=self._compute_stats(triplets),
        )

    def mine_distance_weighted(
        self,
        n: int = 100,
        temperature: float = 1.0,
        clip: float = 1e-3,
    ) -> MiningResult:
        """Mine distance-weighted sampled triplets.

        Sampling probability is proportional to
        ``1 / max(d(a, n), clip)`` — negatives at moderate distances are
        preferred while very-close negatives are down-weighted, following
        the distance-weighted sampling idea from Wu et al. (2017).

        Args:
            n: Maximum number of triplets to return.
            temperature: Softmax temperature (higher → more uniform).
            clip: Minimum distance to avoid division by zero.

        Returns:
            A :class:`MiningResult` with up to *n* triplets.
        """
        triplets: List[TripletIndices] = []
        n_samples = 0
        total_an_dist = 0.0

        anchors = list(range(len(self.embeddings)))
        random.shuffle(anchors)

        for a in anchors:
            if len(triplets) >= n:
                break

            pos_indices = np.where(self.pos_mask[a])[0]
            if len(pos_indices) == 0:
                continue

            neg_indices = np.where(~self.pos_mask[a])[0]
            # Exclude self
            neg_indices = neg_indices[neg_indices != a]
            if len(neg_indices) == 0:
                continue

            p = int(random.choice(pos_indices))

            an_dists = self.dists[a, neg_indices].copy()

            # Compute sampling weights: 1/d, clipped and normalised
            weights = 1.0 / np.maximum(an_dists, clip)
            # Apply temperature to softmax
            weights = np.power(weights, 1.0 / max(temperature, 1e-9))
            weight_sum = weights.sum()
            if weight_sum <= 0:
                continue
            probs = weights / weight_sum

            neg_idx = int(np.random.choice(neg_indices, p=probs))
            an_dist = float(self.dists[a, neg_idx])
            ap_dist = float(self.dists[a, p])

            triplets.append(TripletIndices(
                anchor=a, positive=p, negative=neg_idx,
                ap_dist=ap_dist, an_dist=an_dist,
            ))
            n_samples += 1
            total_an_dist += an_dist

        stats = self._compute_stats(triplets)
        stats["temperature"] = temperature
        return MiningResult(
            triplets=triplets,
            strategy="distance_weighted",
            stats=stats,
        )

    # -- internal helpers ----------------------------------------------------

    def _collect_triplets(
        self,
        selector: "_SelectorFn",
        max_count: int,
    ) -> List[TripletIndices]:
        """Iterate over anchors and collect triplets via *selector*."""
        triplets: List[TripletIndices] = []
        anchors = list(range(len(self.embeddings)))
        random.shuffle(anchors)

        for a in anchors:
            if len(triplets) >= max_count:
                break
            pos_indices = np.where(self.pos_mask[a])[0]
            if len(pos_indices) == 0:
                continue
            neg_indices = np.where(~self.pos_mask[a])[0]
            neg_indices = neg_indices[neg_indices != a]
            if len(neg_indices) == 0:
                continue

            # Pick the closest positive
            pos_dists = self.dists[a, pos_indices]
            p = int(pos_indices[np.argmin(pos_dists)])
            ap_dist = float(self.dists[a, p])

            neg_idx = selector(a, p, ap_dist, neg_indices)
            if neg_idx is not None:
                an_dist = float(self.dists[a, neg_idx])
                triplets.append(TripletIndices(
                    anchor=a, positive=p, negative=neg_idx,
                    ap_dist=ap_dist, an_dist=an_dist,
                ))

        return triplets

    # -- selectors (return index or None) ------------------------------------

    def _hard_selector(self, margin: float = 0.0) -> "_SelectorFn":
        def _select(
            a: int, p: int, ap_dist: float, neg_indices: "np.ndarray"
        ) -> Optional[int]:
            an_dists = self.dists[a, neg_indices]
            # Hard: an_dist < ap_dist + margin
            hard_mask = an_dists < (ap_dist + margin)
            hard_negs = neg_indices[hard_mask]
            if len(hard_negs) == 0:
                return None
            # Pick the hardest (smallest distance to anchor)
            hard_dists = an_dists[hard_mask]
            return int(hard_negs[np.argmin(hard_dists)])
        return _select

    def _semi_hard_selector(self) -> "_SelectorFn":
        def _select(
            a: int, p: int, ap_dist: float, neg_indices: "np.ndarray"
        ) -> Optional[int]:
            an_dists = self.dists[a, neg_indices]
            # Semi-hard: ap_dist < an_dist (but not too far)
            semi_mask = an_dists > ap_dist
            semi_negs = neg_indices[semi_mask]
            if len(semi_negs) == 0:
                return None
            # Pick the closest semi-hard negative
            semi_dists = an_dists[semi_mask]
            return int(semi_negs[np.argmin(semi_dists)])
        return _select

    @staticmethod
    def _compute_stats(triplets: List[TripletIndices]) -> dict:
        if not triplets:
            return {"count": 0}
        ap_dists = [t.ap_dist for t in triplets]
        an_dists = [t.an_dist for t in triplets]
        return {
            "count": len(triplets),
            "avg_ap_dist": float(np.mean(ap_dists)),
            "avg_an_dist": float(np.mean(an_dists)),
            "min_an_dist": float(np.min(an_dists)),
            "max_an_dist": float(np.max(an_dists)),
        }


# Type alias for selector callbacks
_SelectorFn = "callable"
