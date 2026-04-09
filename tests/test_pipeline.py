from __future__ import annotations

import json
from pathlib import Path

from linksocial_final_year.data import build_pair_task, load_raw_profiles, write_profiles_jsonl, read_profiles_jsonl
from linksocial_final_year.evaluation import run_pair_experiment
from linksocial_final_year.features import FeatureStore
from linksocial_final_year.schema import ExperimentConfig


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _build_fixture_dataset(base: Path) -> Path:
    raw_dir = base / "raw"

    triple = raw_dir / "3.profile.data"
    entries = {
        "alicegroup": {
            "googlePlus- alice.json": {"userName": "alice", "fullName": "Alice Johnson", "bio": "data scientist", "externalUrl": ["alice.dev"]},
            "instagram-alice.json": {"userName": "alicepics", "fullName": "Alice Johnson", "bio": "data scientist photos", "externalUrl": ["alice.dev"]},
            "twitter-alice.json": {"userName": "@alice", "fullName": "Alice Johnson", "bio": "data science and ml", "externalUrl": "alice.dev"},
        },
        "bobgroup": {
            "googlePlus- bob.json": {"userName": "bobby", "fullName": "Bob Smith", "bio": "coffee and code"},
            "instagram-bob.json": {"userName": "bobbytravels", "fullName": "Bob Smith", "bio": "coffee travel"},
            "twitter-bob.json": {"userName": "@bobby", "fullName": "Bob Smith", "bio": "coffee code"},
        },
        "carolgroup": {
            "googlePlus- carol.json": {"userName": "carolk", "fullName": "Carol King", "bio": "design systems"},
            "instagram-carol.json": {"userName": "carol.design", "fullName": "Carol King", "bio": "design studio"},
            "twitter-carol.json": {"userName": "@carolking", "fullName": "Carol King", "bio": "systems design"},
        },
        "davegroup": {
            "googlePlus- dave.json": {"userName": "davey", "fullName": "Dave Lee", "bio": "backend engineer"},
            "instagram-dave.json": {"userName": "dave.codes", "fullName": "Dave Lee", "bio": "engineering life"},
            "twitter-dave.json": {"userName": "@davelee", "fullName": "Dave Lee", "bio": "backend and infra"},
        },
    }
    for group_name, files in entries.items():
        for file_name, payload in files.items():
            _write_json(triple / group_name / file_name, payload)

    single = raw_dir / "1.profile.data" / "loner"
    _write_json(single / "twitter-loner.json", {"userName": "@loner", "fullName": "Solo Person", "bio": "just one profile"})

    pair_dir = raw_dir / "2.profile.data" / "Google_Insta" / "evegroup"
    _write_json(pair_dir / "googlePlus- eve.json", {"userName": "eve", "fullName": "Eve Adams", "bio": "security researcher"})
    _write_json(pair_dir / "instagram-eve.json", {"userName": "evegrams", "fullName": "Eve Adams", "bio": "security and privacy"})
    _write_json(pair_dir / "filename.json", {"ignore": True})
    _write_json(pair_dir / "score_file.json", {"ignore": True})
    return raw_dir


def test_load_raw_profiles_parses_dataset_structure(tmp_path: Path) -> None:
    raw_dir = _build_fixture_dataset(tmp_path)
    profiles = load_raw_profiles(raw_dir)
    assert len(profiles) == 15
    assert {profile.platform for profile in profiles} == {"google_plus", "instagram", "twitter"}


def test_prepare_and_reload_round_trip(tmp_path: Path) -> None:
    raw_dir = _build_fixture_dataset(tmp_path)
    profiles = load_raw_profiles(raw_dir)
    output_path = tmp_path / "processed" / "profiles.jsonl"
    write_profiles_jsonl(profiles, output_path)
    loaded = read_profiles_jsonl(output_path)
    assert len(loaded) == len(profiles)
    assert loaded[0].identity_id == profiles[0].identity_id


def test_pair_experiment_runs_end_to_end_on_fixture(tmp_path: Path) -> None:
    raw_dir = _build_fixture_dataset(tmp_path)
    profiles = load_raw_profiles(raw_dir)
    store = FeatureStore(profiles)
    config = ExperimentConfig(raw_dir=raw_dir, processed_path=tmp_path / "processed.jsonl", results_dir=tmp_path / "results", min_candidates=2, max_candidates=3, cluster_ratio=1.0)
    result, _, task = run_pair_experiment(profiles, store, config, "google_plus", "instagram")
    assert result.train_groups >= 1
    assert result.test_groups >= 1
    assert "linksocial_sgd_accuracy" in result.metrics
    assert task.source_platform == "google_plus"
