"""Tests for mining strategies."""

import pytest
from triplet_miner.strategies import (
    MiningStrategy,
    compute_similarity,
    select_negatives,
    _jaccard,
)


class TestSimilarity:
    def test_identical_strings(self):
        sim = compute_similarity("hello world foo bar", "hello world foo bar")
        assert sim > 0.9

    def test_disjoint_strings(self):
        sim = compute_similarity("alpha beta gamma", "delta epsilon zeta")
        assert sim < 0.3

    def test_partial_overlap(self):
        sim = compute_similarity("fix the login bug in auth", "fix the password reset flow")
        assert 0.0 < sim < 1.0

    def test_empty_strings(self):
        sim = compute_similarity("", "")
        assert sim == 0.0

    def test_jaccard_direct(self):
        # tokens shorter than 3 chars are filtered, so use longer words
        assert _jaccard("alpha beta gamma", "alpha beta gamma") == 1.0
        assert _jaccard("", "") == 0.0


class TestSelectNegatives:
    def test_random_returns_correct_count(self):
        candidates = [f"code block {i}" for i in range(20)]
        negs = select_negatives("query", "positive", candidates, MiningStrategy.RANDOM, n=3)
        assert len(negs) == 3

    def test_random_fewer_candidates(self):
        negs = select_negatives("q", "p", ["only one"], MiningStrategy.RANDOM, n=5)
        assert len(negs) == 1

    def test_hard_negative_picks_similar(self):
        anchor = "implement user authentication system"
        candidates = [
            "user auth login implementation",
            "database migration script",
            "color theme changes",
        ]
        negs = select_negatives(anchor, "positive", candidates, MiningStrategy.HARD_NEGATIVE, n=1)
        assert len(negs) == 1
        # "user auth login" should be most similar to anchor
        assert negs[0][0] == "user auth login implementation"

    def test_semi_hard(self):
        anchor = "fix the login authentication bug"
        positive = "fix the login authentication bug in auth.py"
        candidates = [
            "completely unrelated text about database schema",
            "fix the password reset authentication service",
            "implement dark mode toggle feature",
        ]
        negs = select_negatives(anchor, positive, candidates, MiningStrategy.SEMI_HARD, n=1)
        assert len(negs) >= 1

    def test_domain_aware(self):
        candidates = ["code from repo A", "code from repo B"]
        sources = ["repo-a", "repo-b"]
        negs = select_negatives(
            "query", "positive", candidates, MiningStrategy.DOMAIN_AWARE,
            n=1, anchor_source="repo-a", candidate_sources=sources,
        )
        assert len(negs) == 1
        # Should pick from repo-b (different source)
        assert negs[0][0] == "code from repo B"

    def test_domain_aware_fallback(self):
        candidates = ["code from same repo"]
        sources = ["repo-a"]
        negs = select_negatives(
            "query", "positive", candidates, MiningStrategy.DOMAIN_AWARE,
            n=1, anchor_source="repo-a", candidate_sources=sources,
        )
        # Falls back to hard negative
        assert len(negs) == 1

    def test_empty_candidates(self):
        negs = select_negatives("q", "p", [], MiningStrategy.RANDOM, n=1)
        assert negs == []

    def test_returns_similarity(self):
        negs = select_negatives("hello world", "p", ["hello foo"], MiningStrategy.RANDOM, n=1)
        assert len(negs) == 1
        assert isinstance(negs[0][1], float)
