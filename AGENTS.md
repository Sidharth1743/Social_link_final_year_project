# AGENTS.md

## Project Goal

Reproduce the LINKSOCIAL user-profile-linkage pipeline on the public LinkSocial dataset, then compare it with one stronger modern text-profile baseline.

## Dataset

- Raw source: `data/raw/Dataset-LinkSocial`
- Supported platforms: Google+, Instagram, Twitter
- Identity groups come from directory structure:
  - `1.profile.data`: single-platform groups
  - `2.profile.data`: pair-platform groups
  - `3.profile.data`: three-platform groups

## Pipeline

1. Prepare canonical profile records from raw JSON.
2. Build pairwise candidate pools with bigram Jaccard pruning.
3. Compute classical LINKSOCIAL-style similarities.
4. Train pairwise rankers and evaluate pairwise plus multi-platform linkage.
5. Compare against a stronger text-representation baseline.

## Working Rules

- Use `uv` for dependency management and execution.
- Keep documentation minimal: `README.md` and this file only.
- Prefer deterministic behavior with explicit random seeds.
- Treat dataset parsing carefully; raw files have mild schema variation.

## Common Commands

- `uv sync`
- `uv run linksocial prepare-data`
- `uv run linksocial run-experiments`
- `uv run pytest`
