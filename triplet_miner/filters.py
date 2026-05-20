"""Quality filters for mined triplets."""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Set

from triplet_miner.triplet import Triplet


# Common programming language file extensions
LANG_EXTENSIONS: Dict[str, Set[str]] = {
    "python": {".py"},
    "javascript": {".js", ".jsx", ".mjs", ".cjs"},
    "typescript": {".ts", ".tsx"},
    "rust": {".rs"},
    "go": {".go"},
    "java": {".java"},
    "c": {".c", ".h"},
    "cpp": {".cpp", ".cc", ".cxx", ".hpp", ".hxx"},
    "ruby": {".rb"},
    "swift": {".swift"},
    "kotlin": {".kt", ".kts"},
    "scala": {".scala"},
    "r": {".r", ".R"},
    "shell": {".sh", ".bash"},
    "sql": {".sql"},
    "html": {".html", ".htm"},
    "css": {".css", ".scss", ".sass", ".less"},
    "markdown": {".md", ".rst"},
    "yaml": {".yml", ".yaml"},
    "json": {".json"},
    "toml": {".toml"},
}

# Extension → language reverse map
_EXT_TO_LANG: Dict[str, str] = {}
for lang, exts in LANG_EXTENSIONS.items():
    for ext in exts:
        _EXT_TO_LANG[ext] = lang


def detect_languages(files: List[str]) -> Set[str]:
    """Detect programming languages from a list of file paths."""
    langs = set()
    for f in files:
        ext = ""
        if "." in f:
            ext = "." + f.rsplit(".", 1)[-1]
        if ext in _EXT_TO_LANG:
            langs.add(_EXT_TO_LANG[ext])
    return langs


class QualityFilter:
    """Filter and score triplets by quality criteria.

    Args:
        min_length: Minimum character length for anchor and positive.
        max_length: Maximum character length (skip bloated entries).
        deduplicate: Remove near-duplicate triplets.
        languages: If set, only keep triplets with these languages in metadata.
        min_quality: Minimum overall quality score (0.0 – 1.0).
    """

    def __init__(
        self,
        min_length: int = 10,
        max_length: int = 50000,
        deduplicate: bool = True,
        languages: Optional[Set[str]] = None,
        min_quality: float = 0.0,
    ):
        self.min_length = min_length
        self.max_length = max_length
        self.deduplicate = deduplicate
        self.languages = languages
        self.min_quality = min_quality

    def score(self, triplet: Triplet) -> float:
        """Compute a quality score (0.0 – 1.0) for a triplet.

        Factors:
        - Anchor/positive length (too short or too long is penalized)
        - Similarity score
        - Metadata richness
        """
        score = 0.5  # base

        # Length quality — prefer 50–5000 chars, penalize extremes
        for text in (triplet.anchor, triplet.positive):
            length = len(text)
            if length < self.min_length:
                return 0.0  # fails hard filter
            if length > self.max_length:
                return 0.0
            # Sweet spot: 50–2000 chars
            if 50 <= length <= 2000:
                score += 0.1
            elif 20 <= length <= 5000:
                score += 0.05

        # Negative should be non-trivial
        if len(triplet.negative) < self.min_length:
            return 0.0

        # Similarity factor — higher is better for quality
        score += min(triplet.similarity * 0.2, 0.2)

        # Metadata richness
        if triplet.metadata:
            score += 0.05
            if "sha" in triplet.metadata:
                score += 0.05

        return min(score, 1.0)

    def filter(self, triplets: List[Triplet]) -> List[Triplet]:
        """Apply quality filters to a list of triplets."""
        result = []
        seen_hashes: Set[str] = set()

        for t in triplets:
            # Hard length filter
            if len(t.anchor) < self.min_length or len(t.positive) < self.min_length:
                continue
            if len(t.anchor) > self.max_length or len(t.positive) > self.max_length:
                continue
            if len(t.negative) < self.min_length:
                continue

            # Language filter
            if self.languages:
                t_langs = set()
                meta = t.metadata or {}
                if "languages" in meta:
                    t_langs = set(meta["languages"])
                if "files" in meta:
                    t_langs |= detect_languages(meta["files"])
                if not t_langs or not (t_langs & self.languages):
                    continue

            # Quality score
            q = self.score(t)
            if q < self.min_quality:
                continue

            # Deduplication — hash based on anchor + positive
            if self.deduplicate:
                h = hashlib_content(t.anchor + t.positive)
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)

            result.append(t)

        return result


def hashlib_content(text: str) -> str:
    """Stable hash for deduplication."""
    import hashlib

    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]
