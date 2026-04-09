from __future__ import annotations

from pathlib import Path

import numpy as np


def sanitize_model_name(model_name: str) -> str:
    return model_name.replace("/", "__").replace("-", "_").replace(".", "_")


def load_or_create_semantic_embeddings(
    profile_ids: list[str],
    texts: list[str],
    model_name: str,
    cache_dir: Path,
    batch_size: int = 128,
) -> tuple[np.ndarray, str]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{sanitize_model_name(model_name)}.npz"
    if cache_path.exists():
        payload = np.load(cache_path, allow_pickle=True)
        cached_ids = payload["profile_ids"].tolist()
        if cached_ids == profile_ids:
            return payload["embeddings"], str(payload["device"])

    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)
    np.savez_compressed(
        cache_path,
        profile_ids=np.array(profile_ids, dtype=object),
        embeddings=embeddings,
        device=np.array(device),
    )
    return embeddings, device
