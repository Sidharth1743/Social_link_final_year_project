from __future__ import annotations

from fastapi.testclient import TestClient

from linksocial_final_year.webapp import create_app


class StubService:
    def overview(self):
        return {
            "dataset_counts": {"google_plus": 1, "instagram": 1, "twitter": 1},
            "identity_counts": {"single_platform_groups": 1, "pair_platform_groups": 1, "triple_platform_groups": 1},
            "models": ["baseline", "semantic_hybrid_gbdt"],
            "semantic_device": "cpu",
            "metrics": {"pairwise": [], "multi_platform": {}},
        }

    def search_profiles(self, platform, query, limit):
        return [{"profile_id": "p1", "platform": "twitter", "username": "@demo", "full_name": "Demo User", "bio": "demo", "identity_id": "g1", "external_urls": [], "source_path": "", "platform_count": 2}]

    def profile_detail(self, profile_id):
        return {"profile": self.search_profiles(None, None, 1)[0], "linked_profiles": []}

    def link_profile(self, profile_id, target_platform, model, top_k):
        profile = self.search_profiles(None, None, 1)[0]
        return {
            "source": profile,
            "target_platform": target_platform,
            "selected_model": model,
            "available_models": ["baseline", "semantic_hybrid_gbdt"],
            "leaders": {"baseline": "p2"},
            "candidates": [
                {
                    "profile": {**profile, "profile_id": "p2", "platform": target_platform},
                    "scores": {"baseline": 0.7, "semantic_hybrid_gbdt": 0.9},
                    "features": {"username_jaro_winkler": 0.8},
                    "is_truth": True,
                    "won_models": ["semantic_hybrid_gbdt"],
                }
            ],
            "graph": {
                "nodes": [
                    {"id": "p1", "kind": "source", "platform": "twitter", "label": "@demo", "score": 1.0},
                    {"id": "p2", "kind": "candidate", "platform": target_platform, "label": "@demo2", "score": 0.9},
                ],
                "edges": [{"source": "p1", "target": "p2", "score": 0.9, "is_truth": True}],
            },
        }


def test_webapp_endpoints_with_stub_service():
    app = create_app(service=StubService())
    client = TestClient(app)

    overview = client.get("/api/overview")
    assert overview.status_code == 200
    assert overview.json()["models"] == ["baseline", "semantic_hybrid_gbdt"]

    profiles = client.get("/api/profiles")
    assert profiles.status_code == 200
    assert profiles.json()[0]["profile_id"] == "p1"

    linked = client.get("/api/link/p1", params={"target_platform": "instagram", "model": "semantic_hybrid_gbdt"})
    assert linked.status_code == 200
    assert linked.json()["selected_model"] == "semantic_hybrid_gbdt"
