"""Export triplets to various formats."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List, Optional

from triplet_miner.triplet import Triplet, _make_pytorch_dataset, _make_hf_dataset


def export_json(triplets: List[Triplet], path: str) -> str:
    """Export triplets to JSON file."""
    data = [t.to_dict() for t in triplets]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path


def export_csv(triplets: List[Triplet], path: str) -> str:
    """Export triplets to CSV file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "anchor", "positive", "negative", "similarity", "source", "metadata",
        ])
        for t in triplets:
            writer.writerow([
                t.anchor,
                t.positive,
                t.negative,
                t.similarity,
                t.source,
                json.dumps(t.metadata, ensure_ascii=False),
            ])
    return path


def to_pytorch_dataset(triplets: List[Triplet]):
    """Convert triplets to a PyTorch Dataset. Requires torch."""
    return _make_pytorch_dataset(triplets)


def to_hf_dataset(triplets: List[Triplet]):
    """Convert triplets to a HuggingFace Dataset. Requires datasets."""
    return _make_hf_dataset(triplets)


def export_triplets(triplets: List[Triplet], path: str) -> str:
    """Auto-detect format from file extension and export."""
    p = Path(path)
    ext = p.suffix.lower()

    if ext == ".json":
        return export_json(triplets, path)
    elif ext == ".csv":
        return export_csv(triplets, path)
    else:
        raise ValueError(f"Unsupported export format: {ext}. Use .json or .csv")
