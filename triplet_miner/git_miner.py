"""Git-powered triplet miner — the main entry point."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

from triplet_miner.triplet import Triplet, _make_pytorch_dataset, _make_hf_dataset
from triplet_miner.strategies import (
    MiningStrategy,
    compute_similarity,
    select_negatives,
)
from triplet_miner.filters import QualityFilter
from triplet_miner.exporters import export_json, export_csv, export_triplets


# ─── Strategy resolution ───────────────────────────────────────────

def _resolve_strategy(strategy) -> MiningStrategy:
    """Accept MiningStrategy enum or string name."""
    if isinstance(strategy, MiningStrategy):
        return strategy
    if isinstance(strategy, str):
        try:
            return MiningStrategy(strategy.lower())
        except ValueError:
            raise ValueError(
                f"Unknown strategy {strategy!r}. "
                f"Choose from: {', '.join(s.value for s in MiningStrategy)}"
            )
    raise TypeError(f"strategy must be MiningStrategy or str, got {type(strategy).__name__}")


# ─── Git commit data ───────────────────────────────────────────────

def _parse_git_log(repo_path: str, max_commits: int = 500) -> List[Dict[str, Any]]:
    """Parse git log into structured commits."""
    result = subprocess.run(
        [
            "git", "log",
            f"-{max_commits}",
            "--format=%H|%an|%at|%s",
            "--numstat",
            "--no-merges",
        ],
        capture_output=True,
        text=True,
        cwd=str(repo_path),
        timeout=60,
    )

    if result.returncode != 0 or not result.stdout.strip():
        return []

    commits = []
    lines = result.stdout.strip().split("\n")
    i = 0

    while i < len(lines):
        line = lines[i]
        if not line or "|" not in line:
            i += 1
            continue

        parts = line.split("|", 3)
        if len(parts) < 4:
            i += 1
            continue

        sha, author, ts_str, message = parts

        insertions = 0
        deletions = 0
        files_changed = 0
        files = []

        i += 1
        while i < len(lines) and lines[i] and "\t" in lines[i]:
            stat = lines[i].split("\t")
            if len(stat) >= 3:
                try:
                    ins = int(stat[0]) if stat[0] != "-" else 0
                    dels = int(stat[1]) if stat[1] != "-" else 0
                    insertions += ins
                    deletions += dels
                    files_changed += 1
                    files.append(stat[2])
                except ValueError:
                    pass
            i += 1

        # Get the actual diff content for this commit (positive text)
        diff_text = _get_diff_text(repo_path, sha)

        commits.append({
            "sha": sha[:12],
            "author": author,
            "timestamp": float(ts_str),
            "message": message[:500],
            "insertions": insertions,
            "deletions": deletions,
            "files_changed": files_changed,
            "files": files,
            "diff": diff_text,
        })

    return commits


def _get_diff_text(repo_path: str, sha: str, max_bytes: int = 10000) -> str:
    """Get a truncated diff for a commit."""
    try:
        result = subprocess.run(
            ["git", "diff-tree", "--no-commit-id", "-p", sha],
            capture_output=True,
            text=True,
            cwd=str(repo_path),
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout[:max_bytes]
    except (subprocess.TimeoutExpired, Exception):
        pass
    return ""


def _get_file_contents(repo_path: str, max_files: int = 50) -> List[str]:
    """Get current file contents as potential negatives."""
    repo = Path(repo_path)
    contents = []
    count = 0

    for f in sorted(repo.rglob("*")):
        if count >= max_files:
            break
        if f.is_file() and not f.name.startswith("."):
            # Skip binary-ish files
            if f.suffix in {".py", ".js", ".ts", ".rs", ".go", ".java", ".md",
                           ".txt", ".yaml", ".yml", ".toml", ".json", ".sh"}:
                try:
                    text = f.read_text(errors="ignore")[:2000]
                    if len(text) > 20:
                        contents.append(text)
                        count += 1
                except Exception:
                    pass

    return contents


class TripletMiner:
    """Mine (anchor, positive, negative) triplets from git repositories.

    Usage:
        miner = TripletMiner()
        triplets = miner.mine_from_repo("/path/to/repo")
        miner.export(triplets, "output.json")
    """

    def __init__(
        self,
        default_strategy: MiningStrategy = MiningStrategy.HARD_NEGATIVE,
        max_commits: int = 500,
        negatives_per_anchor: int = 1,
    ):
        self.default_strategy = _resolve_strategy(default_strategy)
        self.max_commits = max_commits
        self.negatives_per_anchor = negatives_per_anchor

    def __repr__(self) -> str:
        return (
            f"TripletMiner(strategy={self.default_strategy.value}, "
            f"max_commits={self.max_commits}, "
            f"negatives_per_anchor={self.negatives_per_anchor})"
        )

    def mine_from_repo(
        self,
        repo_path: str,
        strategy: Optional[MiningStrategy] = None,
        min_quality: float = 0.0,
    ) -> List[Triplet]:
        """Mine triplets from a single git repository.

        Args:
            repo_path: Path to the git repository.
            strategy: Mining strategy (defaults to self.default_strategy).
            min_quality: Minimum quality score (0.0–1.0).

        Returns:
            List of Triplet objects.
        """
        strategy = _resolve_strategy(strategy or self.default_strategy)
        repo_path = str(repo_path)
        repo_name = Path(repo_path).resolve().name

        # Parse commits
        try:
            commits = _parse_git_log(repo_path, self.max_commits)
        except (FileNotFoundError, OSError):
            return []
        if not commits:
            return []

        # Collect commit diffs keyed by sha for fast exclusion
        commit_diffs: Dict[str, str] = {}
        for c in commits:
            if c["diff"]:
                commit_diffs[c["sha"]] = c["diff"]

        # Build candidate pool: (text, sha_or_none, source)
        all_candidates: List[str] = []
        candidate_shas: List[Optional[str]] = []  # sha for commit diffs, None for files
        candidate_sources: List[str] = []

        for c in commits:
            if c["diff"]:
                all_candidates.append(c["diff"])
                candidate_shas.append(c["sha"])
                candidate_sources.append(repo_name)

        # Also add current file contents as candidates
        file_contents = _get_file_contents(repo_path)
        for fc in file_contents:
            all_candidates.append(fc)
            candidate_shas.append(None)
            candidate_sources.append(repo_name)

        if not all_candidates:
            return []

        # Build sha→index set for O(1) exclusion lookups
        _candidate_sha_set = set(candidate_shas)

        # Build triplets
        triplets = []
        for c in commits:
            anchor = c["message"]
            positive = c["diff"]
            sha = c["sha"]

            if not anchor or not positive:
                continue
            if len(anchor) < 10 or len(positive) < 10:
                continue

            # Compute anchor-positive similarity
            similarity = compute_similarity(anchor, positive)

            # Select negatives (exclude this commit's own diff)
            exclude = {sha}
            neg_candidates = [
                text
                for text, cand_sha in zip(all_candidates, candidate_shas)
                if cand_sha not in exclude
            ]
            neg_sources = [
                src
                for src, cand_sha in zip(candidate_sources, candidate_shas)
                if cand_sha not in exclude
            ]

            negs = select_negatives(
                anchor=anchor,
                positive=positive,
                candidates=neg_candidates[:200],  # cap for perf
                strategy=strategy,
                n=self.negatives_per_anchor,
                anchor_source=repo_name,
                candidate_sources=neg_sources[:200],
            )

            for neg_text, neg_sim in negs:
                t = Triplet(
                    anchor=anchor,
                    positive=positive[:5000],  # cap length
                    negative=neg_text[:5000],
                    similarity=similarity,
                    source=repo_name,
                    metadata={
                        "sha": sha,
                        "author": c["author"],
                        "timestamp": c["timestamp"],
                        "files_changed": c["files_changed"],
                        "files": c["files"][:10],
                        "negative_similarity": neg_sim,
                    },
                )
                triplets.append(t)

        # Apply quality filter
        if min_quality > 0:
            qf = QualityFilter(min_quality=min_quality)
            triplets = qf.filter(triplets)

        return triplets

    def mine_from_repos(
        self,
        repos: List[str],
        strategy: Optional[MiningStrategy] = None,
        min_quality: float = 0.0,
    ) -> List[Triplet]:
        """Mine triplets from multiple repositories.

        Args:
            repos: List of paths to git repositories.
            strategy: Mining strategy.
            min_quality: Minimum quality score.

        Returns:
            Combined list of triplets from all repos.
        """
        strategy = _resolve_strategy(strategy or self.default_strategy)

        all_triplets = []

        for repo_path in repos:
            try:
                triplets = self.mine_from_repo(repo_path, strategy, min_quality=0.0)
                all_triplets.extend(triplets)
            except Exception as e:
                print(f"Warning: failed to mine {repo_path}: {e}")

        # Cross-repo negatives with DOMAIN_AWARE strategy
        if strategy == MiningStrategy.DOMAIN_AWARE and len(repos) > 1:
            all_triplets = self._add_cross_repo_negatives(all_triplets)

        # Quality filter
        if min_quality > 0:
            qf = QualityFilter(min_quality=min_quality)
            all_triplets = qf.filter(all_triplets)

        return all_triplets

    def _add_cross_repo_negatives(self, triplets: List[Triplet]) -> List[Triplet]:
        """Replace negatives with cross-repo ones for DOMAIN_AWARE strategy."""
        # Group by source
        by_source: Dict[str, List[Triplet]] = {}
        for t in triplets:
            by_source.setdefault(t.source, []).append(t)

        sources = list(by_source.keys())
        if len(sources) < 2:
            return triplets

        enhanced = []
        for t in triplets:
            # Find a negative from a different source
            other_sources = [s for s in sources if s != t.source]
            if other_sources:
                other_source = other_sources[hash(t.anchor) % len(other_sources)]
                other_triplets = by_source[other_source]
                if other_triplets:
                    neg_t = other_triplets[hash(t.anchor) % len(other_triplets)]
                    enhanced.append(Triplet(
                        anchor=t.anchor,
                        positive=t.positive,
                        negative=neg_t.positive,
                        similarity=t.similarity,
                        source=t.source,
                        metadata={**t.metadata, "negative_source": neg_t.source},
                    ))
                    continue
            enhanced.append(t)

        return enhanced

    def export(self, triplets: List[Triplet], path: str) -> str:
        """Export triplets to file (auto-detect format from extension)."""
        return export_triplets(triplets, path)

    def to_dataset(self, triplets: List[Triplet]):
        """Convert to PyTorch Dataset."""
        return _make_pytorch_dataset(triplets)

    def to_hf_dataset(self, triplets: List[Triplet]):
        """Convert to HuggingFace Dataset."""
        return _make_hf_dataset(triplets)


def register_triplet_miner(registry):
    """Register TripletMiner with a Mesh-style registry."""
    registry.register("trainers", "triplet-miner", TripletMiner)
