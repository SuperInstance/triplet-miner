# triplet-miner

Mine (anchor, positive, negative) triplets from git history for contrastive learning.

Given a git repo, it extracts commit messages as anchors, diffs as positives, and selects negatives from other commits/file contents using configurable strategies.

## Quick Start

```python
from triplet_miner import TripletMiner, MiningStrategy

miner = TripletMiner(
    default_strategy=MiningStrategy.HARD_NEGATIVE,
    max_commits=500,
    negatives_per_anchor=1,
)

triplets = miner.mine_from_repo("/path/to/repo")
print(f"Mined {len(triplets)} triplets")

# Export
miner.export(triplets, "output.json")
```

## Command Line

```bash
# Install
pip install -e .

# Run from Python
python -c "
from triplet_miner import TripletMiner
miner = TripletMiner()
triplets = miner.mine_from_repo('.')
for t in triplets[:3]:
    print(f'anchor: {t.anchor[:80]}...')
    print(f'positive length: {len(t.positive)} chars')
    print(f'negative length: {len(t.negative)} chars')
    print(f'similarity: {t.similarity:.3f}')
    print()
"
```

## What Gets Mined

For each non-merge commit in the repo:

| Field | Source |
|-------|--------|
| **anchor** | Commit message |
| **positive** | Full diff of the commit (capped at 5000 chars) |
| **negative** | Selected from other commits' diffs or current file contents |

Commits with messages <10 chars or empty diffs are skipped.

## Mining Strategies

```python
from triplet_miner import MiningStrategy
```

| Strategy | How it picks negatives |
|----------|----------------------|
| `RANDOM` | Random selection from candidate pool |
| `HARD_NEGATIVE` | Picks candidates most similar to anchor (hardest to distinguish) |
| `SEMI_HARD` | Picks candidates with similarity between 0 and anchor-positive similarity |
| `DOMAIN_AWARE` | Picks negatives from different repos (cross-repo mining only) |

Similarity is computed using 3-shingle Jaccard (min-hash style) with word-level Jaccard fallback.

### Strategy Examples

```python
# Hard negatives: best for training discriminative models
miner = TripletMiner(strategy=MiningStrategy.HARD_NEGATIVE)

# Semi-hard: balanced between easy and hard
miner = TripletMiner(strategy=MiningStrategy.SEMI_HARD)

# Multi-repo with cross-domain negatives
triplets = miner.mine_from_repos(
    ["/path/to/repo-a", "/path/to/repo-b", "/path/to/repo-c"],
    strategy=MiningStrategy.DOMAIN_AWARE,
)
```

With `DOMAIN_AWARE`, negatives come from different repos than the anchor, and a deterministic hash ensures consistency.

## Output Format

### JSON

```json
[
  {
    "anchor": "fix: handle empty diff in triplet mining",
    "positive": "diff --git a/triplet_miner/git_miner.py ...\n-index abc1234..def5678 100644\n...",
    "negative": "diff --git a/README.md ...\n Completely unrelated change...",
    "similarity": 0.23,
    "source": "triplet-miner",
    "metadata": {
      "sha": "abc1234def567",
      "author": "developer",
      "timestamp": 1700000000.0,
      "files_changed": 3,
      "files": ["triplet_miner/git_miner.py", "tests/test_git_miner.py"],
      "negative_similarity": 0.15
    }
  }
]
```

### CSV

```
anchor,positive,negative,similarity,source,metadata
"fix: handle...","diff --git...","diff --git...",0.23,"triplet-miner","{...}"
```

## Quality Filtering

```python
from triplet_miner import QualityFilter, Triplet

qf = QualityFilter(
    min_length=10,        # minimum chars for anchor/positive/negative
    max_length=50000,     # maximum chars
    deduplicate=True,     # remove near-duplicate triplets
    languages={"python", "rust"},  # only keep these languages
    min_quality=0.3,      # minimum quality score
)

filtered = qf.filter(triplets)
```

Quality score factors (0.0–1.0):
- Base: 0.5
- Length sweet spot (50–2000 chars): +0.1 per field
- Moderate length (20–5000): +0.05 per field
- Anchor-positive similarity: +0 to +0.2
- Metadata present: +0.05
- SHA in metadata: +0.05

Language detection uses file extensions mapped to 20+ languages.

## Multi-Repo Mining

```python
from triplet_miner import TripletMiner, MiningStrategy, QualityFilter

miner = TripletMiner(
    default_strategy=MiningStrategy.DOMAIN_AWARE,
    max_commits=200,
    negatives_per_anchor=2,
)

triplets = miner.mine_from_repos(
    ["/repos/plato-core", "/repos/fleet-router", "/repos/constraint-substrate"],
    min_quality=0.3,
)

miner.export(triplets, "training-data.json")
```

`mine_from_repos` mines each repo independently, then applies quality filtering. With `DOMAIN_AWARE`, cross-repo negatives replace same-repo ones.

## PyTorch / HuggingFace Integration

```python
from triplet_miner import TripletMiner

miner = TripletMiner()
triplets = miner.mine_from_repo(".")

# PyTorch Dataset (requires pip install triplet-miner[torch])
dataset = miner.to_dataset(triplets)
# TripletDataset(len=150)

# HuggingFace Dataset (requires pip install datasets)
hf_dataset = miner.to_hf_dataset(triplets)
```

## Install

```bash
pip install -e .

# With PyTorch support
pip install -e ".[torch]"

# With HuggingFace
pip install -e ".[torch]" datasets
```

## TypeScript / npm

An npm package is available at `npm/`:

```bash
cd npm
npm install
npm run build
```

## Tests

```bash
pytest tests/
```

## License

MIT
