"""triplet-miner: Mine (anchor, positive, negative) triplets from git history."""

from triplet_miner.triplet import Triplet, TripletDataset
from triplet_miner.strategies import MiningStrategy
from triplet_miner.git_miner import TripletMiner
from triplet_miner.filters import QualityFilter
from triplet_miner.exporters import export_triplets
from triplet_miner.sampling import AdvancedStrategies, MiningResult, TripletIndices
from triplet_miner.evaluation import TripletEvaluator, TripletMetrics, ClusteringResult, EmbeddingReport

__version__ = "0.1.0"
__all__ = [
    "Triplet",
    "TripletDataset",
    "TripletMiner",
    "MiningStrategy",
    "QualityFilter",
    "export_triplets",
    "AdvancedStrategies",
    "MiningResult",
    "TripletIndices",
    "TripletEvaluator",
    "TripletMetrics",
    "ClusteringResult",
    "EmbeddingReport",
]
