# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-05-20

### Added
- Initial release: mine (anchor, positive, negative) triplets from git history.
- `TripletMiner` class with single-repo and multi-repo mining.
- Four mining strategies: random, hard_negative, semi_hard, domain_aware.
- `QualityFilter` for length, deduplication, language, and quality scoring.
- Export to JSON and CSV.
- Optional PyTorch `TripletDataset` and HuggingFace Dataset conversion.
- Mesh-style registry hook (`register_triplet_miner`).

### Changed
- **Breaking:** `TripletDataset` promoted to a module-level class (was inner class in factory function).
  This enables `isinstance()` checks and subclassing.
- Removed unused `gitpython` dependency — the package uses `subprocess.run` throughout.
- Negative candidate filtering now uses set-based SHA exclusion instead of fragile index arithmetic.
- `TripletMiner.__init__` and strategy-accepting methods now accept string strategy names
  (e.g. `"random"`) in addition to `MiningStrategy` enum values.

### Added (polish)
- `__repr__` on `TripletMiner` and `TripletDataset`.
- `py.typed` marker for PEP 561 compliance.
- Explicit `__all__` exports including `TripletDataset`.
