from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .schema import ProfileRecord
from .semantic import load_or_create_semantic_embeddings
from .utils import (
    char_probability_vector,
    compact_alnum,
    generate_bigrams,
    jaccard_set_similarity,
    jaro_winkler_similarity,
    normalize_handle,
    normalize_text,
    safe_float,
    symmetric_kl_similarity,
    token_set,
    url_tokens,
)


CLASSICAL_FEATURES = [
    "username_jaro_winkler",
    "full_name_jaro_winkler",
    "bio_tfidf_cosine",
    "username_bigram_jaccard",
    "name_bigram_jaccard",
    "merged_bigram_jaccard",
    "username_char_similarity",
    "name_char_similarity",
    "merged_char_similarity",
    "external_url_jaccard",
    "username_exact",
    "full_name_exact",
]

MODERN_EXTRA_FEATURES = [
    "profile_text_char_cosine",
    "bio_char_cosine",
    "token_jaccard",
]

SEMANTIC_EXTRA_FEATURES = [
    "semantic_profile_cosine",
    "semantic_profile_l2_similarity",
]


@dataclass(frozen=True)
class PreparedProfile:
    profile: ProfileRecord
    username: str
    full_name: str
    bio: str
    combined_text: str
    username_bigrams: tuple[str, ...]
    name_bigrams: tuple[str, ...]
    merged_bigrams: tuple[str, ...]
    username_char_probs: np.ndarray
    name_char_probs: np.ndarray
    merged_char_probs: np.ndarray
    url_tokens: set[str]
    text_tokens: set[str]


class FeatureStore:
    def __init__(
        self,
        profiles: list[ProfileRecord],
        semantic_model_name: str | None = None,
        semantic_cache_dir: Path | None = None,
        semantic_batch_size: int = 128,
    ) -> None:
        self.profile_ids = [profile.profile_id for profile in profiles]
        self.id_to_index = {profile_id: idx for idx, profile_id in enumerate(self.profile_ids)}
        self.prepared: dict[str, PreparedProfile] = {}
        self.semantic_model_name = semantic_model_name
        self.semantic_device = "disabled"
        self.semantic_embeddings: np.ndarray | None = None

        bios: list[str] = []
        combined_texts: list[str] = []
        bio_char_texts: list[str] = []

        for profile in profiles:
            username = normalize_handle(profile.username)
            full_name = normalize_text(profile.full_name)
            bio = normalize_text(profile.bio)
            username_bigrams = profile.raw_bigrams or generate_bigrams(profile.username)
            name_bigrams = generate_bigrams(profile.full_name)
            merged_bigrams = tuple(dict.fromkeys([*username_bigrams, *name_bigrams]))
            combined_text = " ".join(part for part in [username, full_name, bio, " ".join(profile.external_urls)] if part).strip()

            prepared = PreparedProfile(
                profile=profile,
                username=username,
                full_name=full_name,
                bio=bio,
                combined_text=combined_text,
                username_bigrams=username_bigrams,
                name_bigrams=name_bigrams,
                merged_bigrams=merged_bigrams,
                username_char_probs=char_probability_vector(username),
                name_char_probs=char_probability_vector(full_name),
                merged_char_probs=char_probability_vector(f"{username} {full_name}"),
                url_tokens=url_tokens(profile.external_urls),
                text_tokens=token_set(combined_text),
            )
            self.prepared[profile.profile_id] = prepared
            bios.append(bio or "emptybio")
            combined_texts.append(combined_text or "emptyprofile")
            bio_char_texts.append(compact_alnum(bio) or "emptybio")

        self.bio_vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1, 2))
        self.bio_matrix = self.bio_vectorizer.fit_transform(bios)

        self.profile_char_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
        self.profile_char_matrix = self.profile_char_vectorizer.fit_transform(combined_texts)

        self.bio_char_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
        self.bio_char_matrix = self.bio_char_vectorizer.fit_transform(bio_char_texts)

        if semantic_model_name:
            embeddings, device = load_or_create_semantic_embeddings(
                profile_ids=self.profile_ids,
                texts=combined_texts,
                model_name=semantic_model_name,
                cache_dir=semantic_cache_dir or Path("data/processed/semantic_cache"),
                batch_size=semantic_batch_size,
            )
            self.semantic_embeddings = embeddings
            self.semantic_device = device

    def _cosine(self, matrix, left_id: str, right_id: str) -> float:
        left_idx = self.id_to_index[left_id]
        right_idx = self.id_to_index[right_id]
        return safe_float(matrix[left_idx].multiply(matrix[right_idx]).sum())

    def _semantic_cosine(self, left_id: str, right_id: str) -> float:
        if self.semantic_embeddings is None:
            return 0.0
        left_idx = self.id_to_index[left_id]
        right_idx = self.id_to_index[right_id]
        return safe_float(float(np.dot(self.semantic_embeddings[left_idx], self.semantic_embeddings[right_idx])))

    def _semantic_l2_similarity(self, left_id: str, right_id: str) -> float:
        if self.semantic_embeddings is None:
            return 0.0
        left_idx = self.id_to_index[left_id]
        right_idx = self.id_to_index[right_id]
        distance = float(np.linalg.norm(self.semantic_embeddings[left_idx] - self.semantic_embeddings[right_idx]))
        return 1.0 / (1.0 + distance)

    @property
    def semantic_available(self) -> bool:
        return self.semantic_embeddings is not None

    def get_prepared(self, profile: ProfileRecord) -> PreparedProfile:
        return self.prepared[profile.profile_id]

    def pair_feature_dict(self, left: ProfileRecord, right: ProfileRecord) -> dict[str, float]:
        left_p = self.get_prepared(left)
        right_p = self.get_prepared(right)
        features = {
            "username_jaro_winkler": jaro_winkler_similarity(left_p.username, right_p.username),
            "full_name_jaro_winkler": jaro_winkler_similarity(left_p.full_name, right_p.full_name),
            "bio_tfidf_cosine": self._cosine(self.bio_matrix, left.profile_id, right.profile_id),
            "username_bigram_jaccard": jaccard_set_similarity(left_p.username_bigrams, right_p.username_bigrams),
            "name_bigram_jaccard": jaccard_set_similarity(left_p.name_bigrams, right_p.name_bigrams),
            "merged_bigram_jaccard": jaccard_set_similarity(left_p.merged_bigrams, right_p.merged_bigrams),
            "username_char_similarity": symmetric_kl_similarity(left_p.username_char_probs, right_p.username_char_probs),
            "name_char_similarity": symmetric_kl_similarity(left_p.name_char_probs, right_p.name_char_probs),
            "merged_char_similarity": symmetric_kl_similarity(left_p.merged_char_probs, right_p.merged_char_probs),
            "external_url_jaccard": jaccard_set_similarity(left_p.url_tokens, right_p.url_tokens),
            "username_exact": float(left_p.username == right_p.username and bool(left_p.username)),
            "full_name_exact": float(left_p.full_name == right_p.full_name and bool(left_p.full_name)),
            "profile_text_char_cosine": self._cosine(self.profile_char_matrix, left.profile_id, right.profile_id),
            "bio_char_cosine": self._cosine(self.bio_char_matrix, left.profile_id, right.profile_id),
            "token_jaccard": jaccard_set_similarity(left_p.text_tokens, right_p.text_tokens),
        }
        if self.semantic_available:
            features["semantic_profile_cosine"] = self._semantic_cosine(left.profile_id, right.profile_id)
            features["semantic_profile_l2_similarity"] = self._semantic_l2_similarity(left.profile_id, right.profile_id)
        return features

    def feature_names_for(self, feature_set: str) -> list[str]:
        if feature_set == "classical":
            return CLASSICAL_FEATURES
        if feature_set == "lexical_modern":
            return CLASSICAL_FEATURES + MODERN_EXTRA_FEATURES
        if feature_set == "hybrid":
            return CLASSICAL_FEATURES + MODERN_EXTRA_FEATURES + (SEMANTIC_EXTRA_FEATURES if self.semantic_available else [])
        raise ValueError(f"Unsupported feature set: {feature_set}")

    def pair_feature_vector(self, left: ProfileRecord, right: ProfileRecord, feature_names: list[str]) -> np.ndarray:
        feature_dict = self.pair_feature_dict(left, right)
        return np.array([safe_float(feature_dict.get(name, 0.0)) for name in feature_names], dtype=float)
