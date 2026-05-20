"""Mining strategies for selecting negatives."""

from __future__ import annotations

import enum
import hashlib
import math
import random
from typing import List, Optional, Set, Tuple

from triplet_miner.triplet import Triplet


class MiningStrategy(enum.Enum):
    """Strategy for selecting negative examples."""

    RANDOM = "random"
    HARD_NEGATIVE = "hard_negative"
    SEMI_HARD = "semi_hard"
    DOMAIN_AWARE = "domain_aware"


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer."""
    return [w.lower() for w in text.split() if len(w) > 2]


def _token_set(text: str) -> Set[str]:
    return set(_tokenize(text))


def _jaccard(a: str, b: str) -> float:
    """Jaccard similarity between two texts based on word overlap."""
    sa, sb = _token_set(a), _token_set(b)
    if not sa and not sb:
        return 0.0
    union = sa | sb
    if not union:
        return 0.0
    return len(sa & sb) / len(union)


def _shingle_hash(text: str, k: int = 3) -> Set[int]:
    """Create k-shingle hashes for min-hash style similarity."""
    words = _tokenize(text)
    if len(words) < k:
        return set()
    return {
        int(hashlib.md5(" ".join(words[i : i + k]).encode()).hexdigest()[:8], 16)
        for i in range(len(words) - k + 1)
    }


def _shingle_similarity(a: str, b: str) -> float:
    """Shingle-based Jaccard similarity (more robust than word-level)."""
    sa, sb = _shingle_hash(a), _shingle_hash(b)
    if not sa and not sb:
        return 0.0
    union = sa | sb
    if not union:
        return 0.0
    return len(sa & sb) / len(union)


def compute_similarity(a: str, b: str) -> float:
    """Compute text similarity (shingle-based with word-Jaccard fallback)."""
    sim = _shingle_similarity(a, b)
    if sim > 0:
        return sim
    return _jaccard(a, b)


def select_negatives(
    anchor: str,
    positive: str,
    candidates: List[str],
    strategy: MiningStrategy,
    n: int = 1,
    anchor_source: str = "",
    candidate_sources: Optional[List[str]] = None,
) -> List[Tuple[str, float]]:
    """Select negative candidates according to the given strategy.

    Returns list of (negative_text, similarity_to_anchor).
    """

    if not candidates:
        return []

    if strategy == MiningStrategy.RANDOM:
        return _random_negatives(anchor, candidates, n)

    elif strategy == MiningStrategy.HARD_NEGATIVE:
        return _hard_negatives(anchor, candidates, n)

    elif strategy == MiningStrategy.SEMI_HARD:
        return _semi_hard_negatives(anchor, positive, candidates, n)

    elif strategy == MiningStrategy.DOMAIN_AWARE:
        return _domain_aware_negatives(
            anchor, candidates, n, anchor_source, candidate_sources
        )

    return _random_negatives(anchor, candidates, n)


def _random_negatives(
    anchor: str, candidates: List[str], n: int
) -> List[Tuple[str, float]]:
    """Randomly pick n candidates as negatives."""
    sample = random.sample(candidates, min(n, len(candidates)))
    return [(c, compute_similarity(anchor, c)) for c in sample]


def _hard_negatives(
    anchor: str, candidates: List[str], n: int
) -> List[Tuple[str, float]]:
    """Pick candidates most similar to the anchor (hardest negatives)."""
    scored = [(c, compute_similarity(anchor, c)) for c in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:n]


def _semi_hard_negatives(
    anchor: str, positive: str, candidates: List[str], n: int
) -> List[Tuple[str, float]]:
    """Pick negatives that are harder than random but not the hardest."""
    pos_sim = compute_similarity(anchor, positive)
    scored = [(c, compute_similarity(anchor, c)) for c in candidates]

    # Want negatives with similarity between 0 and pos_sim
    semi_hard = [
        (c, s) for c, s in scored if 0 < s < pos_sim
    ]

    if not semi_hard:
        # Fallback: pick from middle of scored list
        scored.sort(key=lambda x: x[1])
        mid = len(scored) // 2
        return scored[max(0, mid - n // 2) : mid + n // 2 + 1][:n]

    semi_hard.sort(key=lambda x: x[1], reverse=True)
    return semi_hard[:n]


def _domain_aware_negatives(
    anchor: str,
    candidates: List[str],
    n: int,
    anchor_source: str = "",
    candidate_sources: Optional[List[str]] = None,
) -> List[Tuple[str, float]]:
    """Pick negatives from different sources/repos than the anchor."""
    if candidate_sources is None:
        candidate_sources = [""] * len(candidates)

    # Filter to candidates from different sources
    cross_domain = [
        (c, s, src)
        for c, s, src in zip(
            candidates,
            [compute_similarity(anchor, c) for c in candidates],
            candidate_sources,
        )
        if src != anchor_source
    ]

    if not cross_domain:
        # Fallback to hard negatives if all same domain
        return _hard_negatives(anchor, candidates, n)

    cross_domain.sort(key=lambda x: x[1], reverse=True)
    return [(c, s) for c, s, _ in cross_domain[:n]]
