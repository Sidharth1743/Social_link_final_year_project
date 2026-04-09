from __future__ import annotations

import math
import re
from urllib.parse import urlparse

import numpy as np


ALNUM_RE = re.compile(r"[^a-z0-9]+")
SPACE_RE = re.compile(r"\s+")


def normalize_text(text: str | None) -> str:
    if not text:
        return ""
    return SPACE_RE.sub(" ", str(text).strip().lower())


def normalize_handle(text: str | None) -> str:
    cleaned = normalize_text(text).lstrip("@")
    return cleaned


def compact_alnum(text: str | None) -> str:
    return ALNUM_RE.sub("", normalize_text(text))


def generate_bigrams(text: str | None) -> tuple[str, ...]:
    cleaned = compact_alnum(text)
    if not cleaned:
        return ()
    if len(cleaned) == 1:
        return (cleaned,)
    return tuple(cleaned[idx : idx + 2] for idx in range(len(cleaned) - 1))


def token_set(text: str | None) -> set[str]:
    compact = normalize_text(text)
    if not compact:
        return set()
    return {token for token in compact.split(" ") if token}


def jaccard_set_similarity(left: set[str] | tuple[str, ...], right: set[str] | tuple[str, ...]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set and not right_set:
        return 1.0
    if not left_set or not right_set:
        return 0.0
    union = left_set | right_set
    if not union:
        return 0.0
    return len(left_set & right_set) / len(union)


def char_probability_vector(text: str | None) -> np.ndarray:
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"
    cleaned = compact_alnum(text)
    counts = np.full(len(alphabet), 1e-6, dtype=float)
    if cleaned:
        for char in cleaned:
            idx = alphabet.find(char)
            if idx >= 0:
                counts[idx] += 1.0
    counts /= counts.sum()
    return counts


def symmetric_kl_similarity(left: np.ndarray, right: np.ndarray) -> float:
    kl_lr = float(np.sum(left * np.log(left / right)))
    kl_rl = float(np.sum(right * np.log(right / left)))
    divergence = 0.5 * (kl_lr + kl_rl)
    return 1.0 / (1.0 + divergence)


def url_tokens(urls: tuple[str, ...]) -> set[str]:
    tokens: set[str] = set()
    for raw in urls:
        parsed = urlparse(raw if "://" in raw else f"https://{raw}")
        host = parsed.netloc.lower().replace("www.", "")
        path = parsed.path.strip("/").lower()
        if host:
            tokens.add(host)
        if path:
            tokens.update(part for part in path.split("/") if part)
    return tokens


def safe_float(value: float | int | None) -> float:
    if value is None:
        return 0.0
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return 0.0
    return float(value)


def jaro_winkler_similarity(left: str | None, right: str | None, scaling: float = 0.1) -> float:
    s1 = normalize_text(left)
    s2 = normalize_text(right)
    if s1 == s2:
        return 1.0 if s1 else 0.0
    if not s1 or not s2:
        return 0.0

    len1 = len(s1)
    len2 = len(s2)
    match_distance = max(len1, len2) // 2 - 1

    s1_matches = [False] * len1
    s2_matches = [False] * len2

    matches = 0
    for idx in range(len1):
        start = max(0, idx - match_distance)
        end = min(idx + match_distance + 1, len2)
        for jdx in range(start, end):
            if s2_matches[jdx]:
                continue
            if s1[idx] != s2[jdx]:
                continue
            s1_matches[idx] = True
            s2_matches[jdx] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    s1_match_chars = [s1[idx] for idx, matched in enumerate(s1_matches) if matched]
    s2_match_chars = [s2[idx] for idx, matched in enumerate(s2_matches) if matched]
    transpositions = sum(left_char != right_char for left_char, right_char in zip(s1_match_chars, s2_match_chars)) / 2.0

    jaro = (
        (matches / len1)
        + (matches / len2)
        + ((matches - transpositions) / matches)
    ) / 3.0

    prefix = 0
    for left_char, right_char in zip(s1, s2):
        if left_char != right_char:
            break
        prefix += 1
        if prefix == 4:
            break
    return jaro + prefix * scaling * (1.0 - jaro)
