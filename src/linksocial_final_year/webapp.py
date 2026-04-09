from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .candidates import CandidateIndex
from .data import group_profiles_by_identity, read_profiles_jsonl
from .evaluation import model_names_for_store, score_candidate_set, train_pair_models
from .features import FeatureStore
from .schema import ExperimentConfig, ProfileRecord


class DemoService:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.profiles = read_profiles_jsonl(config.processed_path)
        self.profile_by_id = {profile.profile_id: profile for profile in self.profiles}
        self.grouped = group_profiles_by_identity(self.profiles)
        self.store = FeatureStore(
            self.profiles,
            semantic_model_name=config.semantic_model_name,
            semantic_cache_dir=config.semantic_cache_dir,
            semantic_batch_size=config.semantic_batch_size,
        )
        self.metrics = self._load_metrics()
        self.platform_profiles = {
            platform: [profile for profile in self.profiles if profile.platform == platform]
            for platform in ("google_plus", "instagram", "twitter")
        }
        self.indexes = {
            platform: CandidateIndex(targets, self.store)
            for platform, targets in self.platform_profiles.items()
        }
        self.pair_models = self._load_or_train_models()
        reference_models = next(iter(self.pair_models.values()))
        self.model_names = model_names_for_store(reference_models, self.store)

    def _load_metrics(self) -> dict[str, Any]:
        metrics_path = self.config.results_dir / "metrics.json"
        if not metrics_path.exists():
            return {}
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    def _models_cache_path(self) -> Path:
        return self.config.demo_models_dir / "pair_models.joblib"

    def _load_or_train_models(self) -> dict[tuple[str, str], dict[str, Any]]:
        cache_path = self._models_cache_path()
        if cache_path.exists():
            return joblib.load(cache_path)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        pair_specs = [
            ("google_plus", "instagram"),
            ("google_plus", "twitter"),
            ("instagram", "twitter"),
        ]
        pair_models: dict[tuple[str, str], dict[str, Any]] = {}
        for source_platform, target_platform in pair_specs:
            models, _ = train_pair_models(self.profiles, self.store, self.config, source_platform, target_platform)
            pair_models[tuple(sorted((source_platform, target_platform)))] = models
        joblib.dump(pair_models, cache_path)
        return pair_models

    def _serialize_profile(self, profile: ProfileRecord) -> dict[str, Any]:
        return {
            "profile_id": profile.profile_id,
            "identity_id": profile.identity_id,
            "platform": profile.platform,
            "username": profile.username,
            "full_name": profile.full_name,
            "bio": profile.bio,
            "external_urls": list(profile.external_urls),
            "source_path": profile.source_path,
            "platform_count": len(self.grouped.get(profile.identity_id, {})),
        }

    def overview(self) -> dict[str, Any]:
        dataset_counts = {
            platform: len(profiles)
            for platform, profiles in self.platform_profiles.items()
        }
        identity_counts = {
            "single_platform_groups": sum(1 for profiles in self.grouped.values() if len(profiles) == 1),
            "pair_platform_groups": sum(1 for profiles in self.grouped.values() if len(profiles) == 2),
            "triple_platform_groups": sum(1 for profiles in self.grouped.values() if len(profiles) == 3),
        }
        return {
            "dataset_counts": dataset_counts,
            "identity_counts": identity_counts,
            "models": self.model_names,
            "semantic_device": self.store.semantic_device,
            "metrics": self.metrics,
        }

    def search_profiles(self, platform: str | None, query: str | None, limit: int) -> list[dict[str, Any]]:
        candidates = self.profiles if platform is None else self.platform_profiles.get(platform, [])
        query_norm = (query or "").strip().lower()
        results: list[ProfileRecord] = []
        for profile in candidates:
            if query_norm:
                haystack = " ".join([profile.username, profile.full_name, profile.bio, profile.identity_id]).lower()
                if query_norm not in haystack:
                    continue
            results.append(profile)
            if len(results) >= limit:
                break
        return [self._serialize_profile(profile) for profile in results]

    def profile_detail(self, profile_id: str) -> dict[str, Any]:
        profile = self.profile_by_id.get(profile_id)
        if not profile:
            raise KeyError(profile_id)
        linked_profiles = [
            self._serialize_profile(other)
            for other in self.grouped.get(profile.identity_id, {}).values()
            if other.profile_id != profile.profile_id
        ]
        return {
            "profile": self._serialize_profile(profile),
            "linked_profiles": linked_profiles,
        }

    def link_profile(self, profile_id: str, target_platform: str, model: str, top_k: int) -> dict[str, Any]:
        source = self.profile_by_id.get(profile_id)
        if not source:
            raise KeyError(profile_id)
        if source.platform == target_platform:
            raise ValueError("Source and target platforms must differ")
        if target_platform not in self.indexes:
            raise ValueError(target_platform)

        pair_key = tuple(sorted((source.platform, target_platform)))
        models = self.pair_models.get(pair_key)
        if models is None:
            raise ValueError(f"No trained model for pair {pair_key}")

        candidate_pool = max(top_k * 4, self.config.min_candidates)
        candidates = self.indexes[target_platform].top_candidates(source, candidate_pool)
        model_names, score_map, feature_map = score_candidate_set(source, candidates, self.store, models)
        if model not in model_names:
            raise ValueError(f"Unknown model: {model}")

        sorted_candidates = sorted(
            candidates,
            key=lambda candidate: score_map[candidate.profile_id][model],
            reverse=True,
        )[:top_k]

        leader_by_model = {
            name: max(candidates, key=lambda candidate: score_map[candidate.profile_id][name]).profile_id
            for name in model_names
        }

        nodes = [
            {
                "id": source.profile_id,
                "label": source.username or source.full_name or source.identity_id,
                "platform": source.platform,
                "kind": "source",
                "score": 1.0,
            }
        ]
        edges = []
        serialized_candidates = []
        for candidate in sorted_candidates:
            candidate_scores = score_map[candidate.profile_id]
            nodes.append(
                {
                    "id": candidate.profile_id,
                    "label": candidate.username or candidate.full_name or candidate.identity_id,
                    "platform": candidate.platform,
                    "kind": "candidate",
                    "score": candidate_scores[model],
                }
            )
            edges.append(
                {
                    "source": source.profile_id,
                    "target": candidate.profile_id,
                    "score": candidate_scores[model],
                    "is_truth": candidate.identity_id == source.identity_id,
                }
            )
            serialized_candidates.append(
                {
                    "profile": self._serialize_profile(candidate),
                    "scores": candidate_scores,
                    "features": feature_map[candidate.profile_id],
                    "is_truth": candidate.identity_id == source.identity_id,
                    "won_models": [name for name, winner_id in leader_by_model.items() if winner_id == candidate.profile_id],
                }
            )

        return {
            "source": self._serialize_profile(source),
            "target_platform": target_platform,
            "selected_model": model,
            "available_models": model_names,
            "leaders": leader_by_model,
            "candidates": serialized_candidates,
            "graph": {"nodes": nodes, "edges": edges},
        }


def create_app(service: DemoService | None = None, config: ExperimentConfig | None = None) -> FastAPI:
    app = FastAPI(title="LINKSOCIAL Research Studio")
    root = Path(__file__).parent / "web"
    app.mount("/assets", StaticFiles(directory=root), name="assets")

    @lru_cache(maxsize=1)
    def get_service() -> DemoService:
        if service is not None:
            return service
        return DemoService(config or ExperimentConfig())

    @app.get("/api/overview")
    def api_overview() -> dict[str, Any]:
        return get_service().overview()

    @app.get("/api/profiles")
    def api_profiles(
        platform: str | None = Query(default=None),
        q: str | None = Query(default=None),
        limit: int = Query(default=25, ge=1, le=100),
    ) -> list[dict[str, Any]]:
        return get_service().search_profiles(platform=platform, query=q, limit=limit)

    @app.get("/api/profiles/{profile_id:path}")
    def api_profile_detail(profile_id: str) -> dict[str, Any]:
        try:
            return get_service().profile_detail(profile_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Profile not found") from exc

    @app.get("/api/link/{profile_id:path}")
    def api_link_profile(
        profile_id: str,
        target_platform: str,
        model: str = "semantic_hybrid_gbdt",
        top_k: int = Query(default=8, ge=3, le=20),
    ) -> dict[str, Any]:
        try:
            return get_service().link_profile(profile_id=profile_id, target_platform=target_platform, model=model, top_k=top_k)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Profile not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/")
    def root_index() -> FileResponse:
        return FileResponse(root / "index.html")

    return app


def run_server(config: ExperimentConfig, host: str = "127.0.0.1", port: int = 8000) -> None:
    uvicorn.run(create_app(config=config), host=host, port=port)


def main() -> None:
    run_server(config=ExperimentConfig())
