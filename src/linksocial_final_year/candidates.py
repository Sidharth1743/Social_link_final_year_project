from __future__ import annotations

from collections import defaultdict

from .features import FeatureStore
from .schema import ProfileRecord
from .utils import jaccard_set_similarity


class CandidateIndex:
    def __init__(self, targets: list[ProfileRecord], store: FeatureStore) -> None:
        self.targets = targets
        self.store = store
        self.target_map = {profile.profile_id: profile for profile in targets}
        self.bigram_to_target_ids: dict[str, set[str]] = defaultdict(set)
        for target in targets:
            merged = set(store.get_prepared(target).merged_bigrams)
            for bigram in merged:
                self.bigram_to_target_ids[bigram].add(target.profile_id)

    def top_candidates(self, source: ProfileRecord, top_n: int) -> list[ProfileRecord]:
        merged = set(self.store.get_prepared(source).merged_bigrams)
        overlap_counts: dict[str, int] = defaultdict(int)
        for bigram in merged:
            for target_id in self.bigram_to_target_ids.get(bigram, ()):
                overlap_counts[target_id] += 1

        if not overlap_counts:
            return self.targets[:top_n]

        scored: list[tuple[float, str]] = []
        for target_id, overlap in overlap_counts.items():
            target = self.target_map[target_id]
            target_merged = set(self.store.get_prepared(target).merged_bigrams)
            union_size = len(merged | target_merged)
            score = 0.0 if union_size == 0 else overlap / union_size
            scored.append((score, target_id))

        scored.sort(key=lambda item: (-item[0], item[1]))
        selected_ids = [target_id for _, target_id in scored[:top_n]]
        return [self.target_map[target_id] for target_id in selected_ids]

    def contains_truth(self, source: ProfileRecord, truth_identity_id: str, top_n: int) -> bool:
        return any(candidate.identity_id == truth_identity_id for candidate in self.top_candidates(source, top_n))
