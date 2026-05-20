"""Tests for quality filters."""

import pytest
from triplet_miner.filters import QualityFilter, detect_languages, hashlib_content
from triplet_miner.triplet import Triplet


def _make_triplet(anchor="fix the login bug in auth module",
                  positive="def login(user): return authenticate(user)",
                  negative="class Database: def connect(self): pass",
                  similarity=0.8, **meta):
    return Triplet(
        anchor=anchor, positive=positive, negative=negative,
        similarity=similarity, source="test", metadata=meta,
    )


class TestQualityFilter:
    def test_passes_good_triplet(self):
        qf = QualityFilter()
        t = _make_triplet()
        assert qf.score(t) > 0.0

    def test_rejects_short_anchor(self):
        qf = QualityFilter(min_length=10)
        t = _make_triplet(anchor="hi")
        assert qf.score(t) == 0.0

    def test_rejects_short_positive(self):
        qf = QualityFilter(min_length=10)
        t = _make_triplet(positive="x")
        assert qf.score(t) == 0.0

    def test_filter_removes_low_quality(self):
        qf = QualityFilter(min_quality=0.3)
        ts = [
            _make_triplet(),  # good
            _make_triplet(anchor="hi"),  # too short, score=0
        ]
        result = qf.filter(ts)
        assert len(result) == 1

    def test_deduplication(self):
        qf = QualityFilter(deduplicate=True)
        t1 = _make_triplet()
        t2 = _make_triplet()
        ts = [t1, t2]
        result = qf.filter(ts)
        assert len(result) == 1

    def test_no_dedup(self):
        qf = QualityFilter(deduplicate=False)
        t1 = _make_triplet()
        t2 = _make_triplet()
        ts = [t1, t2]
        result = qf.filter(ts)
        assert len(result) == 2

    def test_language_filter(self):
        qf = QualityFilter(languages={"python"})
        t = _make_triplet(files=["auth.py", "login.py"])
        result = qf.filter([t])
        assert len(result) == 1

    def test_language_filter_mismatch(self):
        qf = QualityFilter(languages={"rust"})
        t = _make_triplet(files=["auth.py"])
        result = qf.filter([t])
        assert len(result) == 0


class TestDetectLanguages:
    def test_python(self):
        assert "python" in detect_languages(["main.py", "utils.py"])

    def test_mixed(self):
        langs = detect_languages(["app.js", "server.ts", "style.css"])
        assert "javascript" in langs
        assert "typescript" in langs
        assert "css" in langs

    def test_unknown(self):
        assert detect_languages(["data.bin"]) == set()


class TestHashing:
    def test_stable(self):
        h1 = hashlib_content("hello world")
        h2 = hashlib_content("hello world")
        assert h1 == h2

    def test_different(self):
        h1 = hashlib_content("hello")
        h2 = hashlib_content("world")
        assert h1 != h2
