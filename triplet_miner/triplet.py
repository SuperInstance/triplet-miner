"""Triplet dataclass and dataset wrappers."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any


@dataclass
class Triplet:
    """A (anchor, positive, negative) triplet for contrastive learning.

    Attributes:
        anchor: The query/example (e.g., commit message or function name).
        positive: Relevant content matching the anchor.
        negative: Unrelated content that should NOT match the anchor.
        similarity: Anchor-positive similarity score (0.0 – 1.0).
        source: Origin (e.g., repo name or path).
        metadata: Extra information (sha, author, files, etc.).
    """

    anchor: str
    positive: str
    negative: str
    similarity: float = 0.0
    source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Triplet":
        return cls(**d)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str) -> "Triplet":
        return cls.from_dict(json.loads(s))


def triplets_to_dicts(triplets: List[Triplet]) -> List[Dict[str, Any]]:
    """Convert a list of triplets to dicts."""
    return [t.to_dict() for t in triplets]


def triplets_from_dicts(dicts: List[Dict[str, Any]]) -> List[Triplet]:
    """Convert a list of dicts to triplets."""
    return [Triplet.from_dict(d) for d in dicts]


# ─── PyTorch Dataset (optional) ────────────────────────────────────

def _make_pytorch_dataset(triplets: List[Triplet]):
    """Create a PyTorch Dataset from triplets. Requires torch."""
    try:
        import torch
        from torch.utils.data import Dataset
    except ImportError:
        raise ImportError(
            "PyTorch is required for Dataset conversion. "
            "Install with: pip install triplet-miner[torch]"
        )

    class TripletDataset(Dataset):
        def __init__(self, data: List[Triplet]):
            self.data = data

        def __len__(self) -> int:
            return len(self.data)

        def __getitem__(self, idx: int):
            t = self.data[idx]
            return {
                "anchor": t.anchor,
                "positive": t.positive,
                "negative": t.negative,
                "similarity": torch.tensor(t.similarity, dtype=torch.float32),
                "source": t.source,
                "metadata": json.dumps(t.metadata),
            }

    return TripletDataset(triplets)


def _make_hf_dataset(triplets: List[Triplet]):
    """Create a HuggingFace Dataset from triplets. Requires datasets."""
    try:
        from datasets import Dataset as HFDataset
    except ImportError:
        raise ImportError(
            "HuggingFace datasets is required. "
            "Install with: pip install datasets"
        )

    data = {
        "anchor": [t.anchor for t in triplets],
        "positive": [t.positive for t in triplets],
        "negative": [t.negative for t in triplets],
        "similarity": [t.similarity for t in triplets],
        "source": [t.source for t in triplets],
        "metadata": [json.dumps(t.metadata) for t in triplets],
    }
    return HFDataset.from_dict(data)
