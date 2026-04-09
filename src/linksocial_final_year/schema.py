from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


PLATFORMS = ("google_plus", "instagram", "twitter")


@dataclass(frozen=True)
class ProfileRecord:
    profile_id: str
    identity_id: str
    dataset_partition: str
    platform: str
    username: str
    full_name: str
    bio: str
    external_urls: tuple[str, ...]
    raw_bigrams: tuple[str, ...]
    source_path: str

    def as_json(self) -> dict[str, object]:
        return {
            "profile_id": self.profile_id,
            "identity_id": self.identity_id,
            "dataset_partition": self.dataset_partition,
            "platform": self.platform,
            "username": self.username,
            "full_name": self.full_name,
            "bio": self.bio,
            "external_urls": list(self.external_urls),
            "raw_bigrams": list(self.raw_bigrams),
            "source_path": self.source_path,
        }

    @classmethod
    def from_json(cls, payload: dict[str, object]) -> "ProfileRecord":
        return cls(
            profile_id=str(payload["profile_id"]),
            identity_id=str(payload["identity_id"]),
            dataset_partition=str(payload["dataset_partition"]),
            platform=str(payload["platform"]),
            username=str(payload.get("username", "")),
            full_name=str(payload.get("full_name", "")),
            bio=str(payload.get("bio", "")),
            external_urls=tuple(str(x) for x in payload.get("external_urls", [])),
            raw_bigrams=tuple(str(x) for x in payload.get("raw_bigrams", [])),
            source_path=str(payload.get("source_path", "")),
        )


@dataclass(frozen=True)
class ExperimentConfig:
    raw_dir: Path = Path("data/raw/Dataset-LinkSocial")
    processed_path: Path = Path("data/processed/profiles.jsonl")
    results_dir: Path = Path("results/latest")
    semantic_cache_dir: Path = Path("data/processed/semantic_cache")
    demo_models_dir: Path = Path("data/processed/demo_models")
    train_fraction: float = 0.6
    seed: int = 42
    cluster_ratio: float = 0.02
    max_candidates: int = 150
    min_candidates: int = 30
    random_negatives_per_positive: int = 1
    hard_negatives_per_positive: int = 2
    semantic_model_name: str | None = "BAAI/bge-small-en-v1.5"
    semantic_batch_size: int = 128

    def candidate_pool_size(self, target_size: int) -> int:
        scaled = max(self.min_candidates, int(target_size * self.cluster_ratio))
        return min(self.max_candidates, max(1, scaled, 1))


@dataclass
class PairTask:
    source_platform: str
    target_platform: str
    train_groups: list[str]
    test_groups: list[str]
    train_group_profiles: dict[str, dict[str, ProfileRecord]]
    test_group_profiles: dict[str, dict[str, ProfileRecord]]


@dataclass
class ModelArtifacts:
    feature_names: list[str]
    metrics: dict[str, float] = field(default_factory=dict)
