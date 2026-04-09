from __future__ import annotations

import json
from pathlib import Path

from .schema import PairTask, ProfileRecord


PLATFORM_MAP = {
    "googlePlus": "google_plus",
    "instagram": "instagram",
    "twitter": "twitter",
}


def _iter_group_directories(raw_dir: Path) -> list[tuple[str, str, Path]]:
    groups: list[tuple[str, str, Path]] = []
    single_dir = raw_dir / "1.profile.data"
    if single_dir.exists():
        for group_dir in sorted(single_dir.iterdir()):
            if group_dir.is_dir():
                groups.append((f"1.profile.data::{group_dir.name}", "single", group_dir))

    pair_dir = raw_dir / "2.profile.data"
    if pair_dir.exists():
        for pair_group in sorted(pair_dir.iterdir()):
            if not pair_group.is_dir():
                continue
            for group_dir in sorted(pair_group.iterdir()):
                if group_dir.is_dir():
                    groups.append((f"2.profile.data/{pair_group.name}::{group_dir.name}", pair_group.name, group_dir))

    triple_dir = raw_dir / "3.profile.data"
    if triple_dir.exists():
        for group_dir in sorted(triple_dir.iterdir()):
            if group_dir.is_dir():
                groups.append((f"3.profile.data::{group_dir.name}", "triple", group_dir))
    return groups


def load_raw_profiles(raw_dir: Path) -> list[ProfileRecord]:
    profiles: list[ProfileRecord] = []
    for identity_id, partition, group_dir in _iter_group_directories(raw_dir):
        for json_path in sorted(group_dir.iterdir()):
            if json_path.suffix != ".json" or json_path.name in {"filename.json", "score_file.json"}:
                continue
            platform_token = json_path.name.split("-")[0]
            platform = PLATFORM_MAP.get(platform_token)
            if not platform:
                continue
            payload = json.loads(json_path.read_text())
            external_urls = payload.get("externalUrl", [])
            if isinstance(external_urls, str):
                external_urls = [external_urls]
            if external_urls is None:
                external_urls = []
            record = ProfileRecord(
                profile_id=f"{identity_id}::{platform}::{json_path.stem}",
                identity_id=identity_id,
                dataset_partition=partition,
                platform=platform,
                username=str(payload.get("userName", "") or ""),
                full_name=str(payload.get("fullName", "") or ""),
                bio=str(payload.get("bio", "") or ""),
                external_urls=tuple(str(item) for item in external_urls),
                raw_bigrams=tuple(str(item) for item in payload.get("bigrams", []) or []),
                source_path=str(json_path),
            )
            profiles.append(record)
    return profiles


def write_profiles_jsonl(profiles: list[ProfileRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for profile in profiles:
            handle.write(json.dumps(profile.as_json(), ensure_ascii=True) + "\n")


def read_profiles_jsonl(input_path: Path) -> list[ProfileRecord]:
    profiles: list[ProfileRecord] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            profiles.append(ProfileRecord.from_json(json.loads(line)))
    return profiles


def group_profiles_by_identity(profiles: list[ProfileRecord]) -> dict[str, dict[str, ProfileRecord]]:
    grouped: dict[str, dict[str, ProfileRecord]] = {}
    for profile in profiles:
        grouped.setdefault(profile.identity_id, {})[profile.platform] = profile
    return grouped


def build_pair_task(
    profiles: list[ProfileRecord],
    source_platform: str,
    target_platform: str,
    train_fraction: float,
    seed: int,
) -> PairTask:
    import random

    grouped = group_profiles_by_identity(profiles)
    eligible = [
        identity_id
        for identity_id, platform_map in grouped.items()
        if source_platform in platform_map and target_platform in platform_map
    ]
    rng = random.Random(seed)
    eligible.sort()
    rng.shuffle(eligible)
    cutoff = max(1, int(len(eligible) * train_fraction))
    train_groups = sorted(eligible[:cutoff])
    test_groups = sorted(eligible[cutoff:])
    if not test_groups:
        test_groups = train_groups[-max(1, len(train_groups) // 3) :]
        train_groups = train_groups[: -len(test_groups)] or train_groups

    return PairTask(
        source_platform=source_platform,
        target_platform=target_platform,
        train_groups=train_groups,
        test_groups=test_groups,
        train_group_profiles={group_id: grouped[group_id] for group_id in train_groups},
        test_group_profiles={group_id: grouped[group_id] for group_id in test_groups},
    )
