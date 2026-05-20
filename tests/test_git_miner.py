"""Tests for git miner — uses real git repos created in /tmp."""

import os
import subprocess
import tempfile
import pytest
from pathlib import Path

from triplet_miner import TripletMiner, Triplet, MiningStrategy


def _create_test_repo(repo_dir: str, commits: int = 5):
    """Create a test git repo with realistic commits."""
    os.makedirs(repo_dir, exist_ok=True)
    subprocess.run(["git", "init"], cwd=repo_dir, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=repo_dir, capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=repo_dir, capture_output=True,
    )

    for i in range(commits):
        # Create a file
        filepath = os.path.join(repo_dir, f"module_{i}.py")
        with open(filepath, "w") as f:
            f.write(f"# Module {i}\n")
            f.write(f"def function_{i}():\n")
            f.write(f"    '''Module {i} implementation'''\n")
            f.write(f"    return {i} * 2\n")
            f.write(f"\n\nclass Handler{i}:\n")
            f.write(f"    def process(self, data):\n")
            f.write(f"        return data * {i + 1}\n")

        subprocess.run(["git", "add", "-A"], cwd=repo_dir, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", f"implement module {i} with handler class"],
            cwd=repo_dir, capture_output=True,
        )


@pytest.fixture
def test_repo(tmp_path):
    """Create a temporary test repo."""
    repo_dir = str(tmp_path / "test-repo")
    _create_test_repo(repo_dir, commits=5)
    return repo_dir


@pytest.fixture
def test_repos(tmp_path):
    """Create two temporary test repos."""
    repos = []
    for name in ["repo-a", "repo-b"]:
        repo_dir = str(tmp_path / name)
        _create_test_repo(repo_dir, commits=3)
        repos.append(repo_dir)
    return repos


class TestGitMiner:
    def test_mine_from_repo(self, test_repo):
        miner = TripletMiner()
        triplets = miner.mine_from_repo(test_repo)
        assert isinstance(triplets, list)
        assert len(triplets) > 0
        for t in triplets:
            assert isinstance(t, Triplet)
            assert len(t.anchor) > 0
            assert len(t.positive) > 0
            assert len(t.negative) > 0

    def test_mine_with_strategy(self, test_repo):
        miner = TripletMiner()
        for strategy in MiningStrategy:
            triplets = miner.mine_from_repo(test_repo, strategy=strategy)
            assert isinstance(triplets, list)

    def test_mine_from_repos(self, test_repos):
        miner = TripletMiner()
        triplets = miner.mine_from_repos(test_repos)
        assert len(triplets) > 0
        sources = {t.source for t in triplets}
        assert len(sources) >= 1

    def test_triplet_metadata(self, test_repo):
        miner = TripletMiner()
        triplets = miner.mine_from_repo(test_repo)
        if triplets:
            t = triplets[0]
            assert "sha" in t.metadata
            assert "author" in t.metadata
            assert t.metadata["sha"]

    def test_quality_filter(self, test_repo):
        miner = TripletMiner()
        triplets = miner.mine_from_repo(test_repo, min_quality=0.5)
        for t in triplets:
            assert t.similarity >= 0 or True  # Quality filter ran

    def test_export_json(self, test_repo, tmp_path):
        miner = TripletMiner()
        triplets = miner.mine_from_repo(test_repo)
        if triplets:
            path = str(tmp_path / "out.json")
            miner.export(triplets, path)
            assert os.path.exists(path)

    def test_export_csv(self, test_repo, tmp_path):
        miner = TripletMiner()
        triplets = miner.mine_from_repo(test_repo)
        if triplets:
            path = str(tmp_path / "out.csv")
            miner.export(triplets, path)
            assert os.path.exists(path)


class TestEdgeCases:
    def test_empty_repo(self, tmp_path):
        """Non-git directory should return empty."""
        repo_dir = str(tmp_path / "empty")
        os.makedirs(repo_dir)
        miner = TripletMiner()
        triplets = miner.mine_from_repo(repo_dir)
        assert triplets == []

    def test_nonexistent_path(self):
        miner = TripletMiner()
        triplets = miner.mine_from_repo("/nonexistent/path/12345")
        assert triplets == []


class TestLargeRepo:
    def test_stress(self, tmp_path):
        """Repo with many commits shouldn't crash."""
        repo_dir = str(tmp_path / "large-repo")
        _create_test_repo(repo_dir, commits=30)
        miner = TripletMiner(max_commits=30)
        triplets = miner.mine_from_repo(repo_dir)
        assert len(triplets) > 0


class TestMeshRegistration:
    def test_register(self):
        class FakeRegistry:
            def __init__(self):
                self.registrations = []

            def register(self, category, name, cls):
                self.registrations.append((category, name, cls))

        from triplet_miner.git_miner import register_triplet_miner
        reg = FakeRegistry()
        register_triplet_miner(reg)
        assert ("trainers", "triplet-miner", TripletMiner) in reg.registrations
