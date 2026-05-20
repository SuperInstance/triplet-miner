"""Tests for Triplet dataclass and dataset wrappers."""

import json
import pytest
from triplet_miner.triplet import Triplet, triplets_to_dicts, triplets_from_dicts


class TestTriplet:
    def test_construction(self):
        t = Triplet(
            anchor="fix login bug",
            positive="def login(user): ...",
            negative="class Database: ...",
            similarity=0.85,
            source="myrepo",
            metadata={"sha": "abc123"},
        )
        assert t.anchor == "fix login bug"
        assert t.positive == "def login(user): ..."
        assert t.negative == "class Database: ..."
        assert t.similarity == 0.85
        assert t.source == "myrepo"
        assert t.metadata["sha"] == "abc123"

    def test_defaults(self):
        t = Triplet(anchor="a", positive="p", negative="n")
        assert t.similarity == 0.0
        assert t.source == ""
        assert t.metadata == {}

    def test_to_dict(self):
        t = Triplet(anchor="a", positive="p", negative="n", metadata={"x": 1})
        d = t.to_dict()
        assert d["anchor"] == "a"
        assert d["metadata"]["x"] == 1

    def test_from_dict(self):
        d = {"anchor": "a", "positive": "p", "negative": "n", "similarity": 0.5, "source": "r", "metadata": {}}
        t = Triplet.from_dict(d)
        assert t.anchor == "a"
        assert t.similarity == 0.5

    def test_roundtrip_dict(self):
        t = Triplet(anchor="hello", positive="world", negative="foo", similarity=0.9, source="s", metadata={"key": "val"})
        assert Triplet.from_dict(t.to_dict()) == t

    def test_to_json(self):
        t = Triplet(anchor="a", positive="p", negative="n")
        j = t.to_json()
        data = json.loads(j)
        assert data["anchor"] == "a"

    def test_from_json(self):
        j = '{"anchor": "a", "positive": "p", "negative": "n", "similarity": 0.0, "source": "", "metadata": {}}'
        t = Triplet.from_json(j)
        assert t.anchor == "a"


class TestBatchConversion:
    def test_triplets_to_dicts(self):
        ts = [
            Triplet(anchor="a1", positive="p1", negative="n1"),
            Triplet(anchor="a2", positive="p2", negative="n2"),
        ]
        dicts = triplets_to_dicts(ts)
        assert len(dicts) == 2
        assert dicts[0]["anchor"] == "a1"

    def test_triplets_from_dicts(self):
        dicts = [
            {"anchor": "a1", "positive": "p1", "negative": "n1", "similarity": 0.0, "source": "", "metadata": {}},
        ]
        ts = triplets_from_dicts(dicts)
        assert len(ts) == 1
        assert ts[0].anchor == "a1"
