from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class TrainedRanker:
    name: str
    estimator: object
    feature_set: str

    def predict_scores(self, features: np.ndarray) -> np.ndarray:
        if hasattr(self.estimator, "predict_proba"):
            proba = self.estimator.predict_proba(features)
            if proba.ndim == 2 and proba.shape[1] >= 2:
                return proba[:, 1]
        predictions = self.estimator.predict(features)
        return np.asarray(predictions, dtype=float)


def build_sgd_ranker(seed: int) -> TrainedRanker:
    estimator = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                SGDRegressor(
                    loss="squared_error",
                    penalty="l2",
                    alpha=0.0001,
                    max_iter=1000,
                    learning_rate="constant",
                    eta0=0.001,
                    random_state=seed,
                    tol=1e-4,
                ),
            ),
        ]
    )
    return TrainedRanker(name="linksocial_sgd", estimator=estimator, feature_set="classical")


def build_logreg_ranker(seed: int) -> TrainedRanker:
    estimator = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2500,
                    random_state=seed,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    return TrainedRanker(name="linksocial_logreg", estimator=estimator, feature_set="classical")


def build_rf_ranker(seed: int) -> TrainedRanker:
    estimator = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=1,
        random_state=seed,
        n_jobs=-1,
    )
    return TrainedRanker(name="linksocial_rf", estimator=estimator, feature_set="classical")


def build_lexical_modern_ranker(seed: int) -> TrainedRanker:
    estimator = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=6,
        max_iter=300,
        random_state=seed,
    )
    return TrainedRanker(name="lexical_modern_gbdt", estimator=estimator, feature_set="lexical_modern")


def build_semantic_hybrid_ranker(seed: int) -> TrainedRanker:
    estimator = HistGradientBoostingClassifier(
        learning_rate=0.04,
        max_depth=7,
        max_iter=350,
        random_state=seed,
    )
    return TrainedRanker(name="semantic_hybrid_gbdt", estimator=estimator, feature_set="hybrid")
