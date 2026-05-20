"""Tests for exporters."""

import json
import os
import csv
import tempfile
import pytest
from triplet_miner.triplet import Triplet
from triplet_miner.exporters import export_json, export_csv, export_triplets


def _sample_triplets():
    return [
        Triplet(
            anchor="fix login bug",
            positive="def login(user): return auth(user)",
            negative="class DB: pass",
            similarity=0.85,
            source="test-repo",
            metadata={"sha": "abc123"},
        ),
        Triplet(
            anchor="add dark mode",
            positive="body { background: #000; }",
            negative="print('hello')",
            similarity=0.7,
            source="test-repo",
            metadata={"sha": "def456"},
        ),
    ]


class TestExportJSON:
    def test_export(self, tmp_path):
        path = str(tmp_path / "triplets.json")
        ts = _sample_triplets()
        export_json(ts, path)
        with open(path) as f:
            data = json.load(f)
        assert len(data) == 2
        assert data[0]["anchor"] == "fix login bug"

    def test_roundtrip(self, tmp_path):
        path = str(tmp_path / "triplets.json")
        ts = _sample_triplets()
        export_json(ts, path)
        with open(path) as f:
            data = json.load(f)
        restored = [Triplet.from_dict(d) for d in data]
        assert len(restored) == len(ts)
        for r, t in zip(restored, ts):
            assert r.anchor == t.anchor
            assert r.positive == t.positive


class TestExportCSV:
    def test_export(self, tmp_path):
        path = str(tmp_path / "triplets.csv")
        ts = _sample_triplets()
        export_csv(ts, path)
        with open(path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 2
        assert rows[0]["anchor"] == "fix login bug"
        assert rows[0]["similarity"] == "0.85"


class TestAutoDetect:
    def test_json_extension(self, tmp_path):
        path = str(tmp_path / "out.json")
        result = export_triplets(_sample_triplets(), path)
        assert result == path
        assert os.path.exists(path)

    def test_csv_extension(self, tmp_path):
        path = str(tmp_path / "out.csv")
        result = export_triplets(_sample_triplets(), path)
        assert result == path
        assert os.path.exists(path)

    def test_unknown_extension(self, tmp_path):
        path = str(tmp_path / "out.xml")
        with pytest.raises(ValueError, match="Unsupported"):
            export_triplets(_sample_triplets(), path)

    def test_creates_parent_dirs(self, tmp_path):
        path = str(tmp_path / "sub" / "dir" / "out.json")
        export_triplets(_sample_triplets(), path)
        assert os.path.exists(path)
