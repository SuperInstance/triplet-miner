# triplet-miner

**Mine (anchor, positive, negative) triplets from git history for contrastive learning.**

Git commits are natural triplet sources:
- **Anchor** → commit message (the intent)
- **Positive** → the code that was actually changed (relevant to the intent)
- **Negative** → unrelated code from other commits, files, or repos

This is useful for training embedding models, retrieval systems, and any contrastive learning pipeline.

## Install

```bash
pip install triplet-miner
```

With optional dependencies:
```bash
pip install triplet-miner[all]  # torch + numpy
```

## Quick Start

```python
from triplet_miner import TripletMiner, MiningStrategy

miner = TripletMiner()

# Mine from local git repos
triplets = miner.mine_from_repos(
    repos=["/path/to/repo1", "/path/to/repo2"],
    strategy=MiningStrategy.HARD_NEGATIVE,
    min_quality=0.5,
)

# Or from a single repo
triplets = miner.mine_from_repo("/path/to/repo")

# Export
miner.export(triplets, "triplets.json")
miner.export(triplets, "triplets.csv")

# Load for training
dataset = miner.to_dataset(triplets)  # PyTorch Dataset
```

## Triplet Structure

```python
@dataclass
class Triplet:
    anchor: str      # e.g., commit message or function name
    positive: str    # e.g., relevant code/doc
    negative: str    # e.g., unrelated code/doc
    similarity: float  # anchor-positive similarity score
    source: str      # repo name
    metadata: dict   # extra info
```

## Mining Strategies

| Strategy | Description | When to Use |
|----------|-------------|-------------|
| `RANDOM` | Random negative selection | Fast baseline, initial experiments |
| `HARD_NEGATIVE` | Negatives similar to the anchor but wrong | Best for training robust embeddings |
| `SEMI_HARD` | Negatives between random and hard | Balanced difficulty for stable training |
| `DOMAIN_AWARE` | Negatives from different repos/domains | Cross-domain generalization |

## Quality Filters

- **Minimum length**: Skip trivially short anchors/positives
- **Deduplication**: Remove duplicate or near-duplicate triplets
- **Language detection**: Filter by programming language
- **Quality scoring**: Automated quality assessment

## Export Formats

- **JSON**: `miner.export(triplets, "out.json")`
- **CSV**: `miner.export(triplets, "out.csv")`
- **PyTorch Dataset**: `miner.to_dataset(triplets)`
- **HuggingFace Dataset**: `miner.to_hf_dataset(triplets)`

## Works with Any Git Repo

Not just SuperInstance — any git repository works. Point it at your codebase and get training data.

## License

MIT
