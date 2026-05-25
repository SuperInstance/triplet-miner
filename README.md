# triplet-miner

[![PyPI version](https://img.shields.io/pypi/v/triplet-miner.svg)](https://pypi.org/project/triplet-miner/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)

**Mine (anchor, positive, negative) triplets from git history for contrastive learning.**

Git commits are natural triplet sources:
- **Anchor** → commit message (the intent)
- **Positive** → the code that was actually changed (relevant to the intent)
- **Negative** → unrelated code from other commits, files, or repos

This is useful for training embedding models, retrieval systems, and any contrastive learning pipeline. Works with **any git repository** — not just SuperInstance.

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
    metadata: dict   # extra info (sha, author, files, etc.)
```

Each triplet includes rich metadata from the git history:

```json
{
  "anchor": "Fix off-by-one in lattice snap algorithm",
  "positive": "diff --git a/src/lattice.rs ...",
  "negative": "diff --git a/src/cli/theme.rs ...",
  "similarity": 0.23,
  "source": "constraint-theory-core",
  "metadata": {
    "sha": "a1b2c3d4e5f6",
    "author": "developer",
    "timestamp": 1716000000.0,
    "files_changed": 2,
    "files": ["src/lattice.rs", "tests/test_lattice.rs"],
    "negative_similarity": 0.05
  }
}
```

## How It Works

### Git Mining Pipeline

```
Git Repository
    ↓
Parse git log (--numstat, --format) → structured commits
    ↓
Get diff per commit (git diff-tree -p) → positive text
    ↓
Collect file contents → candidate negatives pool
    ↓
For each commit:
  anchor = commit message
  positive = commit diff
  negatives = select from pool (strategy-dependent)
    ↓
Quality filter (length, dedup, language, quality score)
    ↓
Output: List[Triplet]
```

### Mining Strategies

| Strategy | Description | Negative Selection | When to Use |
|----------|-------------|-------------------|-------------|
| `RANDOM` | Random negative selection | Random candidates from pool | Fast baseline, initial experiments |
| `HARD_NEGATIVE` | Negatives most similar to anchor | Top-K by shingle similarity to anchor | Best for training robust embeddings |
| `SEMI_HARD` | Negatives between random and hard | Similarity between 0 and anchor-positive sim | Balanced difficulty for stable training |
| `DOMAIN_AWARE` | Negatives from different repos | Cross-repo candidates, ranked by similarity | Cross-domain generalization |

### Similarity Computation

Text similarity uses a two-stage approach:

1. **Shingle similarity** (primary) — 3-word shingle hashes with Jaccard overlap, more robust than word-level comparison
2. **Word Jaccard** (fallback) — Token overlap when shingles are too sparse

### Quality Filters

| Filter | Default | Description |
|--------|---------|-------------|
| **Minimum length** | 10 chars | Skip trivially short anchors/positives/negatives |
| **Maximum length** | 50,000 chars | Skip bloated entries |
| **Deduplication** | enabled | Remove duplicate/near-duplicate triplets (SHA-256 of anchor+positive) |
| **Language detection** | any | Filter by programming language (20+ languages) |
| **Quality scoring** | 0.0 | Automated quality assessment (0.0–1.0) |

Quality scoring factors:
- **Length sweet spot** — 50–2000 chars gets highest score; <20 or >5000 penalized
- **Similarity** — Higher anchor-positive similarity → higher quality
- **Metadata richness** — Presence of SHA, files, and author info boosts score

```python
from triplet_miner import QualityFilter

qf = QualityFilter(
    min_length=20,
    deduplicate=True,
    languages={"rust", "python"},
    min_quality=0.5,
)
filtered = qf.filter(triplets)
```

### Language Detection

Detects 20+ programming languages from file extensions:

Python, JavaScript, TypeScript, Rust, Go, Java, C, C++, Ruby, Swift, Kotlin, Scala, R, Shell, SQL, HTML, CSS, Markdown, YAML, JSON, TOML.

```python
from triplet_miner.filters import detect_languages

langs = detect_languages(["src/main.rs", "lib/core.py", "README.md"])
# → {"rust", "python", "markdown"}
```

## Export Formats

| Format | Method | Description |
|--------|--------|-------------|
| **JSON** | `export(triplets, "out.json")` | Array of triplet objects |
| **CSV** | `export(triplets, "out.csv")` | Flattened columns |
| **PyTorch Dataset** | `to_dataset(triplets)` | `TripletDataset` with tensor similarity |
| **HuggingFace Dataset** | `to_hf_dataset(triplets)` | `datasets.Dataset` for training pipelines |

### PyTorch Dataset

```python
from triplet_miner import TripletMiner, TripletDataset

miner = TripletMiner()
triplets = miner.mine_from_repo("/path/to/repo")

dataset = miner.to_dataset(triplets)
# Each item: {"anchor": str, "positive": str, "negative": str,
#             "similarity": tensor, "source": str, "metadata": str}
```

### HuggingFace Dataset

```python
hf_dataset = miner.to_hf_dataset(triplets)
# Columns: anchor, positive, negative, similarity, source, metadata
```

## Multi-Repo Mining

When mining from multiple repos with `DOMAIN_AWARE` strategy, cross-repo negatives are automatically added:

```python
miner = TripletMiner(strategy=MiningStrategy.DOMAIN_AWARE)

triplets = miner.mine_from_repos([
    "/repos/constraint-theory-core",
    "/repos/flux-verify-api",
    "/repos/plato-core",
])

# Negatives come from DIFFERENT repos than the anchor,
# maximizing domain diversity for contrastive learning.
```

For other strategies, results are concatenated with optional quality filtering applied at the end.

## API Reference

### TripletMiner

```python
TripletMiner(
    default_strategy=MiningStrategy.HARD_NEGATIVE,
    max_commits=500,            # max commits per repo
    negatives_per_anchor=1,     # negatives per anchor
)

miner.mine_from_repo(repo_path, strategy=None, min_quality=0.0) → List[Triplet]
miner.mine_from_repos(repos, strategy=None, min_quality=0.0) → List[Triplet]
miner.export(triplets, path)     # auto-detects format from extension
miner.to_dataset(triplets)       # → TripletDataset (requires torch)
miner.to_hf_dataset(triplets)    # → HuggingFace Dataset (requires datasets)
```

### QualityFilter

```python
QualityFilter(
    min_length=10,
    max_length=50000,
    deduplicate=True,
    languages=None,       # set of language names, e.g. {"rust", "python"}
    min_quality=0.0,      # minimum quality score (0.0–1.0)
)

qf.score(triplet) → float       # quality score 0.0–1.0
qf.filter(triplets) → list      # apply all filters
```

### MiningStrategy

```python
MiningStrategy.RANDOM          # random negatives
MiningStrategy.HARD_NEGATIVE   # most similar to anchor
MiningStrategy.SEMI_HARD       # between random and hard
MiningStrategy.DOMAIN_AWARE    # cross-repo negatives
```

### Utility Functions

```python
from triplet_miner.strategies import compute_similarity

sim = compute_similarity("fix the parser", "refactor parser module")
# → float (0.0–1.0, shingle-based with Jaccard fallback)
```

## Works with Any Git Repo

Not just SuperInstance — any git repository works. Point it at your codebase and get training data for:

- **Code embeddings** — train models to understand code similarity
- **Retrieval-augmented generation** — improve code search with better embeddings
- **Commit understanding** — models that can match intents to code changes
- **Code review automation** — identify related changes across a codebase

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## License

MIT

## Ecosystem

Part of the [SuperInstance](https://github.com/SuperInstance) ecosystem:

| Package | Description |
|---------|-------------|
| [plato-core](https://github.com/SuperInstance/plato-core) | Base types + mesh registry |
| [tensor-spline](https://github.com/SuperInstance/tensor-spline) | SplineLinear neural compression |
| [eisenstein-embed](https://github.com/SuperInstance/eisenstein-embed) | 5-layer matching cascade |
| [plato-training](https://github.com/SuperInstance/plato-training) | Training monolith |
| [device-router](https://github.com/SuperInstance/device-router) | Heterogeneous compute routing |
| [triplet-miner](https://github.com/SuperInstance/triplet-miner) | Git-powered contrastive data |
| [micro-onnx](https://github.com/SuperInstance/micro-onnx) | ONNX export + benchmark |
| [quality-gate-stream](https://github.com/SuperInstance/quality-gate-stream) | Quality scoring pipeline |
