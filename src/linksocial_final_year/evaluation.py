from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .candidates import CandidateIndex
from .data import build_pair_task, group_profiles_by_identity
from .features import CLASSICAL_FEATURES, FeatureStore
from .models import (
    TrainedRanker,
    build_lexical_modern_ranker,
    build_logreg_ranker,
    build_rf_ranker,
    build_semantic_hybrid_ranker,
    build_sgd_ranker,
)
from .schema import ExperimentConfig, PairTask, ProfileRecord


@dataclass
class PairExperimentResult:
    source_platform: str
    target_platform: str
    train_groups: int
    test_groups: int
    metrics: dict[str, float]

    def as_json(self) -> dict[str, object]:
        return {
            "source_platform": self.source_platform,
            "target_platform": self.target_platform,
            "train_groups": self.train_groups,
            "test_groups": self.test_groups,
            "metrics": self.metrics,
        }


def _stack_feature_rows(rows: list[np.ndarray]) -> np.ndarray:
    if not rows:
        return np.empty((0, 0), dtype=float)
    return np.vstack(rows)


def _build_training_examples(
    task: PairTask,
    store: FeatureStore,
    config: ExperimentConfig,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    rng = random.Random(config.seed)
    train_groups = task.train_group_profiles
    source_profiles = [profiles[task.source_platform] for profiles in train_groups.values()]
    target_profiles = [profiles[task.target_platform] for profiles in train_groups.values()]
    target_by_identity = {profile.identity_id: profile for profile in target_profiles}
    candidate_index = CandidateIndex(target_profiles, store)

    feature_names_by_set = {
        "classical": store.feature_names_for("classical"),
        "lexical_modern": store.feature_names_for("lexical_modern"),
    }
    if store.semantic_available:
        feature_names_by_set["hybrid"] = store.feature_names_for("hybrid")

    rows_by_set: dict[str, list[np.ndarray]] = {feature_set: [] for feature_set in feature_names_by_set}
    labels: list[int] = []

    for source in source_profiles:
        positive_target = target_by_identity[source.identity_id]
        for feature_set, feature_names in feature_names_by_set.items():
            rows_by_set[feature_set].append(store.pair_feature_vector(source, positive_target, feature_names))
        labels.append(1)

        hard_candidates = [
            candidate
            for candidate in candidate_index.top_candidates(source, config.candidate_pool_size(len(target_profiles)))
            if candidate.identity_id != source.identity_id
        ][: config.hard_negatives_per_positive]

        random_candidates = [
            candidate
            for candidate in rng.sample(target_profiles, k=min(len(target_profiles), config.random_negatives_per_positive * 3))
            if candidate.identity_id != source.identity_id
        ][: config.random_negatives_per_positive]

        for negative in [*hard_candidates, *random_candidates]:
            for feature_set, feature_names in feature_names_by_set.items():
                rows_by_set[feature_set].append(store.pair_feature_vector(source, negative, feature_names))
            labels.append(0)

    matrices = {feature_set: _stack_feature_rows(rows) for feature_set, rows in rows_by_set.items()}
    return matrices, np.asarray(labels, dtype=int)


def _train_models(
    feature_matrices: dict[str, np.ndarray],
    labels: np.ndarray,
    seed: int,
) -> dict[str, TrainedRanker]:
    models = {
        "linksocial_sgd": build_sgd_ranker(seed),
        "linksocial_logreg": build_logreg_ranker(seed),
        "linksocial_rf": build_rf_ranker(seed),
        "lexical_modern_gbdt": build_lexical_modern_ranker(seed),
    }
    if "hybrid" in feature_matrices:
        models["semantic_hybrid_gbdt"] = build_semantic_hybrid_ranker(seed)
    for model in models.values():
        model.estimator.fit(feature_matrices[model.feature_set], labels)
    return models


def _baseline_score(feature_vector: np.ndarray, feature_names: list[str]) -> float:
    feature_map = {name: value for name, value in zip(feature_names, feature_vector)}
    return (
        feature_map["username_jaro_winkler"]
        + feature_map["full_name_jaro_winkler"]
        + feature_map["bio_tfidf_cosine"]
    )


def _rank_candidates(
    source: ProfileRecord,
    candidates: list[ProfileRecord],
    store: FeatureStore,
    models: dict[str, TrainedRanker],
) -> dict[str, str]:
    feature_names_by_set = {
        "classical": store.feature_names_for("classical"),
        "lexical_modern": store.feature_names_for("lexical_modern"),
    }
    if store.semantic_available:
        feature_names_by_set["hybrid"] = store.feature_names_for("hybrid")

    rows_by_set: dict[str, list[np.ndarray]] = {feature_set: [] for feature_set in feature_names_by_set}
    for candidate in candidates:
        for feature_set, feature_names in feature_names_by_set.items():
            rows_by_set[feature_set].append(store.pair_feature_vector(source, candidate, feature_names))
    matrices = {feature_set: _stack_feature_rows(rows) for feature_set, rows in rows_by_set.items()}

    predictions: dict[str, str] = {}
    baseline_scores = np.array(
        [_baseline_score(row, feature_names_by_set["classical"]) for row in rows_by_set["classical"]],
        dtype=float,
    )
    predictions["baseline"] = candidates[int(np.argmax(baseline_scores))].identity_id
    if store.semantic_available:
        hybrid_feature_names = feature_names_by_set["hybrid"]
        semantic_idx = hybrid_feature_names.index("semantic_profile_cosine")
        semantic_scores = matrices["hybrid"][:, semantic_idx]
        predictions["semantic_cosine"] = candidates[int(np.argmax(semantic_scores))].identity_id

    for model_name, model in models.items():
        scores = model.predict_scores(matrices[model.feature_set])
        predictions[model_name] = candidates[int(np.argmax(scores))].identity_id
    return predictions


def _prediction_model_names(models: dict[str, TrainedRanker], store: FeatureStore) -> list[str]:
    names = ["baseline"]
    if store.semantic_available:
        names.append("semantic_cosine")
    names.extend(models.keys())
    return names


def model_names_for_store(models: dict[str, TrainedRanker], store: FeatureStore) -> list[str]:
    return _prediction_model_names(models, store)


def train_pair_models(
    profiles: list[ProfileRecord],
    store: FeatureStore,
    config: ExperimentConfig,
    source_platform: str,
    target_platform: str,
) -> tuple[dict[str, TrainedRanker], PairTask]:
    task = build_pair_task(profiles, source_platform, target_platform, config.train_fraction, config.seed)
    feature_matrices, labels = _build_training_examples(task, store, config)
    models = _train_models(feature_matrices, labels, config.seed)
    return models, task


def score_candidate_set(
    source: ProfileRecord,
    candidates: list[ProfileRecord],
    store: FeatureStore,
    models: dict[str, TrainedRanker],
) -> tuple[list[str], dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    feature_names_by_set = {
        "classical": store.feature_names_for("classical"),
        "lexical_modern": store.feature_names_for("lexical_modern"),
    }
    if store.semantic_available:
        feature_names_by_set["hybrid"] = store.feature_names_for("hybrid")

    rows_by_set: dict[str, list[np.ndarray]] = {feature_set: [] for feature_set in feature_names_by_set}
    feature_dicts: dict[str, dict[str, float]] = {}
    for candidate in candidates:
        feature_dict = store.pair_feature_dict(source, candidate)
        feature_dicts[candidate.profile_id] = feature_dict
        for feature_set, feature_names in feature_names_by_set.items():
            rows_by_set[feature_set].append(
                np.array([feature_dict.get(name, 0.0) for name in feature_names], dtype=float)
            )
    matrices = {feature_set: _stack_feature_rows(rows) for feature_set, rows in rows_by_set.items()}

    score_map: dict[str, dict[str, float]] = {candidate.profile_id: {} for candidate in candidates}
    baseline_scores = np.array(
        [_baseline_score(row, feature_names_by_set["classical"]) for row in rows_by_set["classical"]],
        dtype=float,
    )
    for idx, candidate in enumerate(candidates):
        score_map[candidate.profile_id]["baseline"] = float(baseline_scores[idx])

    if store.semantic_available:
        hybrid_feature_names = feature_names_by_set["hybrid"]
        semantic_idx = hybrid_feature_names.index("semantic_profile_cosine")
        semantic_scores = matrices["hybrid"][:, semantic_idx]
        for idx, candidate in enumerate(candidates):
            score_map[candidate.profile_id]["semantic_cosine"] = float(semantic_scores[idx])

    for model_name, model in models.items():
        scores = model.predict_scores(matrices[model.feature_set])
        for idx, candidate in enumerate(candidates):
            score_map[candidate.profile_id][model_name] = float(scores[idx])

    return _prediction_model_names(models, store), score_map, feature_dicts


def run_pair_experiment(
    profiles: list[ProfileRecord],
    store: FeatureStore,
    config: ExperimentConfig,
    source_platform: str,
    target_platform: str,
) -> tuple[PairExperimentResult, dict[str, TrainedRanker], PairTask]:
    task = build_pair_task(profiles, source_platform, target_platform, config.train_fraction, config.seed)
    feature_matrices, labels = _build_training_examples(task, store, config)
    models = _train_models(feature_matrices, labels, config.seed)

    test_profiles = task.test_group_profiles
    source_profiles = [profiles[source_platform] for profiles in test_profiles.values()]
    target_profiles = [profiles[target_platform] for profiles in test_profiles.values()]
    candidate_index = CandidateIndex(target_profiles, store)
    pool_size = config.candidate_pool_size(len(target_profiles))
    model_names = _prediction_model_names(models, store)

    totals = defaultdict(int)
    for source in source_profiles:
        candidates = candidate_index.top_candidates(source, pool_size)
        if not candidates:
            continue
        totals["evaluated"] += 1
        if any(candidate.identity_id == source.identity_id for candidate in candidates):
            totals["candidate_recall_hits"] += 1
        predictions = _rank_candidates(source, candidates, store, models)
        for model_name in model_names:
            predicted_identity = predictions[model_name]
            if predicted_identity == source.identity_id:
                totals[f"{model_name}_hits"] += 1

    denominator = max(1, totals["evaluated"])
    metrics = {
        "candidate_recall_at_pool": totals["candidate_recall_hits"] / denominator,
        "train_examples": int(len(labels)),
    }
    for model_name in model_names:
        metrics[f"{model_name}_accuracy"] = totals[f"{model_name}_hits"] / denominator
    return (
        PairExperimentResult(
            source_platform=source_platform,
            target_platform=target_platform,
            train_groups=len(task.train_groups),
            test_groups=len(task.test_groups),
            metrics=metrics,
        ),
        models,
        task,
    )


def run_multi_platform_experiment(
    profiles: list[ProfileRecord],
    store: FeatureStore,
    config: ExperimentConfig,
    pair_models: dict[tuple[str, str], dict[str, TrainedRanker]],
) -> dict[str, float]:
    grouped = group_profiles_by_identity(profiles)
    eligible = [
        identity_id
        for identity_id, platform_map in grouped.items()
        if all(platform in platform_map for platform in ("google_plus", "instagram", "twitter"))
    ]
    eligible = sorted(eligible)
    rng = random.Random(config.seed)
    rng.shuffle(eligible)
    cutoff = max(1, int(len(eligible) * config.train_fraction))
    test_ids = eligible[cutoff:]
    if not test_ids:
        test_ids = eligible[-max(1, len(eligible) // 3) :]

    test_groups = {group_id: grouped[group_id] for group_id in test_ids}
    indexes = {
        platform: CandidateIndex(
            [platform_map[platform] for platform_map in test_groups.values()],
            store,
        )
        for platform in ("google_plus", "instagram", "twitter")
    }

    per_anchor_hits = defaultdict(int)
    per_anchor_total = defaultdict(int)
    reference_models = next(iter(pair_models.values()))
    tracked_models = _prediction_model_names(reference_models, store)

    for anchor in ("google_plus", "instagram", "twitter"):
        others = [platform for platform in ("google_plus", "instagram", "twitter") if platform != anchor]
        for group_id, platform_map in test_groups.items():
            anchor_profile = platform_map[anchor]
            per_anchor_total[anchor] += 1
            success_by_model = {model_name: True for model_name in tracked_models}
            for target in others:
                model_key = tuple(sorted((anchor, target)))
                models = pair_models.get(model_key)
                if models is None:
                    for model_name in tracked_models:
                        success_by_model[model_name] = False
                    break
                candidates = indexes[target].top_candidates(
                    anchor_profile,
                    config.candidate_pool_size(len(test_groups)),
                )
                if not candidates:
                    for model_name in tracked_models:
                        success_by_model[model_name] = False
                    break
                predictions = _rank_candidates(anchor_profile, candidates, store, models)
                for model_name in tracked_models:
                    if predictions[model_name] != group_id:
                        success_by_model[model_name] = False
            for model_name, success in success_by_model.items():
                if success:
                    per_anchor_hits[(model_name, anchor)] += 1

    results = {
        f"{model_name}_{anchor}_multi_accuracy": per_anchor_hits[(model_name, anchor)] / max(1, per_anchor_total[anchor])
        for model_name in tracked_models
        for anchor in ("google_plus", "instagram", "twitter")
    }
    for model_name in tracked_models:
        anchor_values = [results[f"{model_name}_{anchor}_multi_accuracy"] for anchor in ("google_plus", "instagram", "twitter")]
        results[f"{model_name}_mean_multi_accuracy"] = float(np.mean(anchor_values))
    results["test_groups"] = float(len(test_groups))
    return results


def write_results(
    config: ExperimentConfig,
    pair_results: list[PairExperimentResult],
    multi_result: dict[str, float],
) -> None:
    config.results_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = config.results_dir / "metrics.json"
    summary_path = config.results_dir / "summary.md"

    payload = {
        "pairwise": [result.as_json() for result in pair_results],
        "multi_platform": multi_result,
    }
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = ["# Experiment Summary", "", "## Pairwise Results", ""]
    for result in pair_results:
        lines.append(f"### {result.source_platform} -> {result.target_platform}")
        for metric_name, metric_value in result.metrics.items():
            lines.append(f"- {metric_name}: {metric_value:.4f}" if isinstance(metric_value, float) else f"- {metric_name}: {metric_value}")
        lines.append("")
    lines.append("## Multi-Platform Results")
    lines.append("")
    for metric_name, metric_value in multi_result.items():
        lines.append(f"- {metric_name}: {metric_value:.4f}" if isinstance(metric_value, float) else f"- {metric_name}: {metric_value}")
    lines.append("")
    summary_path.write_text("\n".join(lines), encoding="utf-8")
