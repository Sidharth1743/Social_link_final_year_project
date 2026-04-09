# LINKSOCIAL Implementation Details

## 1. Document Purpose

This document explains the full implementation of the final-year project in detail. It is written to support:

- project submission
- technical review
- viva or presentation preparation
- future extension of the codebase

The goal of this project is to reproduce the core ideas of the LINKSOCIAL paper and then extend them with stronger modern baselines that are still compatible with the publicly available LinkSocial dataset.

## 2. Project Objective

The system links user profiles belonging to the same real-world identity across three social media platforms:

- Google+
- Instagram
- Twitter

The implementation uses only public profile metadata available in the dataset, such as:

- username
- full name
- bio
- external URLs
- precomputed or derived bigrams

The project does not depend on follower graphs, post history, mobility traces, or multimodal post-level signals. That is a deliberate design choice because the public LinkSocial release does not fully support those richer modalities.

## 3. High-Level System Flow

The implemented pipeline follows these stages:

1. Read raw JSON profiles from the LinkSocial dataset.
2. Convert them into a canonical internal schema.
3. Normalize textual fields.
4. Build candidate pools using merged-bigram Jaccard pruning.
5. Generate classical, lexical-modern, and semantic features.
6. Train supervised rankers on matched and non-matched profile pairs.
7. Evaluate pairwise linkage across platform pairs.
8. Evaluate multi-platform linkage across all three platforms.
9. Expose the trained logic through a lightweight FastAPI frontend backend.
10. Allow the user to search profiles and inspect candidate matches interactively in the frontend.

## 4. Dataset Handling

### 4.1 Dataset source

The raw dataset is expected at:

`data/raw/Dataset-LinkSocial`

The code assumes the dataset has three top-level sections:

- `1.profile.data`
- `2.profile.data`
- `3.profile.data`

These correspond to:

- identities with one platform profile
- identities with two linked platform profiles
- identities with three linked platform profiles

### 4.2 Raw directory interpretation

Each identity is represented as a directory. Inside that directory there are JSON files for each platform profile. Examples include:

- `twitter-<name>.json`
- `instagram-<name>.json`
- `googlePlus- <name>.json`

Some directories also contain:

- `filename.json`
- `score_file.json`

These are ignored by the loader because they are not user profile records.

### 4.3 Loader implementation

File:

- `/home/sidharth/Desktop/final_project/src/linksocial_final_year/data.py`

Main responsibilities:

- traverse all valid dataset partitions
- detect the platform from file name
- load JSON content
- handle schema variation across platforms
- convert external URLs into a uniform tuple structure
- create a canonical `ProfileRecord`

### 4.4 Canonical schema

File:

- `/home/sidharth/Desktop/final_project/src/linksocial_final_year/schema.py`

The canonical record is `ProfileRecord`, with fields:

- `profile_id`
- `identity_id`
- `dataset_partition`
- `platform`
- `username`
- `full_name`
- `bio`
- `external_urls`
- `raw_bigrams`
- `source_path`

This makes later pipeline stages independent of the original JSON structure.

### 4.5 Processed output

After parsing, profiles are written to:

- `data/processed/profiles.jsonl`

Each line is one canonical profile record. This makes repeated experiments much faster because the raw dataset does not need to be reparsed every time.

## 5. Configuration System

File:

- `/home/sidharth/Desktop/final_project/src/linksocial_final_year/schema.py`

The project configuration is stored in `ExperimentConfig`.

Important fields:

- `raw_dir`
- `processed_path`
- `results_dir`
- `semantic_cache_dir`
- `demo_models_dir`
- `train_fraction`
- `seed`
- `cluster_ratio`
- `max_candidates`
- `min_candidates`
- `random_negatives_per_positive`
- `hard_negatives_per_positive`
- `semantic_model_name`
- `semantic_batch_size`

### 5.1 Candidate pool sizing

`ExperimentConfig.candidate_pool_size(target_size)` computes the candidate pool size dynamically:

1. Multiply total target size by `cluster_ratio`.
2. Ensure the result is at least `min_candidates`.
3. Cap the result at `max_candidates`.

This allows the candidate pool size to scale with dataset size while staying computationally manageable.

## 6. Text and Similarity Utilities

File:

- `/home/sidharth/Desktop/final_project/src/linksocial_final_year/utils.py`

This file contains the reusable utilities used throughout the pipeline.

### 6.1 Normalization

- `normalize_text`: lowercases, trims, and collapses whitespace
- `normalize_handle`: normalizes usernames and removes `@`
- `compact_alnum`: keeps only lowercase alphanumeric characters

These functions ensure that similarity features are not distorted by case differences, punctuation, or extra spaces.

### 6.2 Token and bigram utilities

- `generate_bigrams`
- `token_set`
- `jaccard_set_similarity`

These are used for username, name, and merged-bigram comparisons, as well as token-level overlap.

### 6.3 Character-distribution features

- `char_probability_vector`
- `symmetric_kl_similarity`

These functions construct a smoothed character-distribution vector over the alphabet plus digits and compare two distributions using symmetric KL divergence converted into a bounded similarity score.

This supports the paper-style idea that string composition can be useful even when exact tokens differ.

### 6.4 URL tokenization

- `url_tokens`

This extracts host and path tokens from external URLs and allows URL similarity to be used as a feature.

### 6.5 String similarity

- `jaro_winkler_similarity`

This is implemented manually rather than relying on an external library. It compares usernames and full names using a similarity function that is tolerant to small edits and prefix agreement.

## 7. Candidate Generation

File:

- `/home/sidharth/Desktop/final_project/src/linksocial_final_year/candidates.py`

The class `CandidateIndex` is responsible for pruning the search space before ranking.

### 7.1 Why candidate pruning is needed

If every source profile were compared with every target profile, computation would become expensive, especially during evaluation. Candidate pruning reduces the number of pairwise comparisons while trying to keep the true match in the retained candidate set.

### 7.2 Data structure

For each target profile:

1. obtain the merged bigram set from the prepared profile
2. add every bigram to an inverted index mapping bigram to target profile IDs

### 7.3 Candidate scoring

For a source profile:

1. compute its merged bigram set
2. collect all target profiles sharing any bigram
3. count overlap occurrences
4. compute Jaccard score between source and target merged bigram sets
5. sort by descending score
6. keep the top `n` candidates

This follows the same broad intuition as the pruning stage described in LINKSOCIAL.

## 8. Feature Engineering

File:

- `/home/sidharth/Desktop/final_project/src/linksocial_final_year/features.py`

The `FeatureStore` prepares all features used by the models.

### 8.1 Prepared profile representation

For each profile, the code constructs a `PreparedProfile` containing:

- normalized username
- normalized full name
- normalized bio
- combined text
- username bigrams
- full-name bigrams
- merged bigrams
- character probability vectors
- URL token sets
- general text token sets

This avoids recomputing preprocessing repeatedly.

### 8.2 Combined text

The combined text is formed from:

- normalized username
- normalized full name
- normalized bio
- external URLs joined as text

This text is used for semantic embedding generation and lexical character-vector similarity.

### 8.3 Classical LINKSOCIAL-style features

The list `CLASSICAL_FEATURES` includes:

- `username_jaro_winkler`
- `full_name_jaro_winkler`
- `bio_tfidf_cosine`
- `username_bigram_jaccard`
- `name_bigram_jaccard`
- `merged_bigram_jaccard`
- `username_char_similarity`
- `name_char_similarity`
- `merged_char_similarity`
- `external_url_jaccard`
- `username_exact`
- `full_name_exact`

These features closely follow the classical profile-attribute matching logic in the original paper.

### 8.4 Lexical-modern features

The list `MODERN_EXTRA_FEATURES` includes:

- `profile_text_char_cosine`
- `bio_char_cosine`
- `token_jaccard`

These features strengthen the feature space without leaving the profile-only setting.

### 8.5 Semantic features

The list `SEMANTIC_EXTRA_FEATURES` includes:

- `semantic_profile_cosine`
- `semantic_profile_l2_similarity`

These are derived from transformer embeddings of the full combined profile text.

### 8.6 Vectorizers used

The `FeatureStore` creates:

- a word-level TF-IDF vectorizer for bio similarity
- a character n-gram TF-IDF vectorizer for profile-level lexical similarity
- a character n-gram TF-IDF vectorizer for bio-level lexical similarity

### 8.7 Feature-set organization

The project supports three feature sets:

- `classical`
- `lexical_modern`
- `hybrid`

Mapping:

- `classical` = only `CLASSICAL_FEATURES`
- `lexical_modern` = `CLASSICAL_FEATURES + MODERN_EXTRA_FEATURES`
- `hybrid` = `CLASSICAL_FEATURES + MODERN_EXTRA_FEATURES + SEMANTIC_EXTRA_FEATURES`

This design allows different models to consume different levels of feature richness.

## 9. Semantic Embedding System

File:

- `/home/sidharth/Desktop/final_project/src/linksocial_final_year/semantic.py`

### 9.1 Model used

The default semantic encoder is:

- `BAAI/bge-small-en-v1.5`

### 9.2 Workflow

1. Build a model-specific cache file name using `sanitize_model_name`.
2. Check whether cached embeddings exist.
3. If cached IDs match the current profile order, reuse the cache.
4. Otherwise:
   - load `SentenceTransformer`
   - choose `cuda` if available, otherwise `cpu`
   - encode all profile combined texts
   - L2-normalize the embeddings
   - save embeddings and profile IDs to a compressed `.npz` file

### 9.3 Cache output

Embeddings are stored under:

- `data/processed/semantic_cache/`

This avoids repeated transformer inference across runs.

## 10. Model Definitions

File:

- `/home/sidharth/Desktop/final_project/src/linksocial_final_year/models.py`

The project defines a `TrainedRanker` wrapper with:

- `name`
- `estimator`
- `feature_set`

### 10.1 `linksocial_sgd`

- implemented with `SGDRegressor`
- wrapped in a standardization pipeline
- uses the classical feature set

This approximates the weighted linear ranking idea in LINKSOCIAL.

### 10.2 `linksocial_logreg`

- implemented with `LogisticRegression`
- wrapped in a standardization pipeline
- uses balanced class weighting
- uses the classical feature set

This serves as a stronger linear comparator against SGD.

### 10.3 `linksocial_rf`

- implemented with `RandomForestClassifier`
- uses the classical feature set

This is the closest paper-style tree-based comparison model in the current codebase.

### 10.4 `lexical_modern_gbdt`

- implemented with `HistGradientBoostingClassifier`
- uses the lexical-modern feature set

This is a stronger modern non-neural baseline using only profile-derived signals.

### 10.5 `semantic_hybrid_gbdt`

- implemented with `HistGradientBoostingClassifier`
- uses the hybrid feature set

This combines classical, lexical, and semantic signals and is the strongest model currently implemented.

## 11. Training Data Construction

File:

- `/home/sidharth/Desktop/final_project/src/linksocial_final_year/evaluation.py`

The training pair construction is handled by `_build_training_examples`.

### 11.1 Pair task generation

Before training, `build_pair_task`:

1. groups profiles by identity
2. selects identities containing both required platforms
3. shuffles with a fixed seed
4. applies a 60/40 train-test split

### 11.2 Positive examples

For each source profile in the training identities:

- the matching target profile from the same identity becomes one positive example

### 11.3 Negative examples

Two kinds of negatives are created:

- hard negatives
- random negatives

#### Hard negatives

Hard negatives come from candidate pruning:

1. generate candidate list for the source
2. remove the true identity
3. take the top `hard_negatives_per_positive`

These are difficult negatives because they are already similar under the pruning logic.

#### Random negatives

Random negatives are sampled randomly from target profiles, excluding the true match.

This combination gives the model both:

- strong confusing negatives
- general contrastive negatives

### 11.4 Multi-feature matrix construction

The code builds separate matrices for each feature set:

- classical
- lexical-modern
- hybrid if semantic features are available

All models are then trained from the same matched and non-matched identity pairs, but on their respective feature spaces.

## 12. Pairwise Evaluation

File:

- `/home/sidharth/Desktop/final_project/src/linksocial_final_year/evaluation.py`

The main function is `run_pair_experiment`.

### 12.1 Platform pairs used

The implemented experiment suite trains and evaluates:

- Google+ -> Instagram
- Google+ -> Twitter
- Instagram -> Twitter

These cover the three pairwise combinations.

### 12.2 Test-time evaluation process

For each source profile in the test split:

1. build candidate set using `CandidateIndex`
2. compute scores for all candidate profiles
3. choose the top candidate for each model
4. compare the predicted identity to the true identity

### 12.3 Reported pairwise metrics

Each pairwise task records:

- `candidate_recall_at_pool`
- `train_examples`
- one accuracy value per model

Candidate recall measures whether the true identity remains in the pruned candidate set. This is important because no ranking model can recover a match that candidate pruning removed.

## 13. Multi-Platform Evaluation

File:

- `/home/sidharth/Desktop/final_project/src/linksocial_final_year/evaluation.py`

The main function is `run_multi_platform_experiment`.

### 13.1 Eligible identities

Only identities present on all three platforms are considered.

### 13.2 Evaluation process

For each anchor platform:

1. use one platform profile as the anchor
2. generate candidate pools in the other two platforms
3. rank candidates using the model corresponding to each platform pair
4. mark success only if both non-anchor platforms return the correct identity

This produces a stricter multi-platform metric than pairwise matching.

### 13.3 Reported metrics

For each model:

- Google+ anchor multi accuracy
- Instagram anchor multi accuracy
- Twitter anchor multi accuracy
- mean multi accuracy

This gives a clearer picture of how stable a method is when identity linking must work consistently across all platforms.

## 14. Result Writing

File:

- `/home/sidharth/Desktop/final_project/src/linksocial_final_year/evaluation.py`

After experiments complete, `write_results` stores:

- `results/latest/metrics.json`
- `results/latest/summary.md`

### 14.1 `metrics.json`

This machine-readable file stores:

- pairwise task metadata
- pairwise metric values
- multi-platform metric values

### 14.2 `summary.md`

This human-readable file summarizes:

- all pairwise results
- all multi-platform results

## 15. Command-Line Interface

File:

- `/home/sidharth/Desktop/final_project/src/linksocial_final_year/cli.py`

The CLI is the main user-facing interface for running the project.

Available commands:

- `prepare-data`
- `run-experiments`
- `serve-web`

### 15.1 `prepare-data`

Reads raw dataset and writes:

- `data/processed/profiles.jsonl`

### 15.2 `run-experiments`

Loads processed profiles, builds feature store, trains the model suite, evaluates pairwise tasks, evaluates multi-platform tasks, and writes final results.

### 15.3 `serve-web`

Starts the FastAPI frontend backend on a given host and port.

## 16. Frontend and Web Backend

Backend file:

- `/home/sidharth/Desktop/final_project/src/linksocial_final_year/webapp.py`

Frontend files:

- `/home/sidharth/Desktop/final_project/src/linksocial_final_year/web/index.html`
- `/home/sidharth/Desktop/final_project/src/linksocial_final_year/web/styles.css`
- `/home/sidharth/Desktop/final_project/src/linksocial_final_year/web/app.js`

### 16.1 Backend architecture

The backend is built with FastAPI and centered around `DemoService`.

### 16.2 `DemoService` responsibilities

When initialized, the service:

1. loads processed profiles
2. creates a profile lookup by ID
3. groups profiles by identity
4. builds a `FeatureStore`
5. loads saved experiment metrics if present
6. builds platform-specific candidate indexes
7. loads cached pair models or trains them for demo use

### 16.3 Demo model caching

Demo pairwise models are cached at:

- `data/processed/demo_models/pair_models.joblib`

This prevents re-training every time the UI starts.

### 16.4 Backend endpoints

Implemented endpoints:

- `GET /`
- `GET /api/overview`
- `GET /api/profiles`
- `GET /api/profiles/{profile_id}`
- `GET /api/link/{profile_id}`

### 16.5 Endpoint behavior

#### `/api/overview`

Returns:

- dataset counts by platform
- identity counts by group size
- available model names
- semantic device used
- metrics from the latest experiment run

#### `/api/profiles`

Supports:

- platform filtering
- query string filtering
- result limit

It searches across username, full name, bio, and identity ID.

#### `/api/profiles/{profile_id}`

Returns:

- selected profile
- other profiles from the same identity group

#### `/api/link/{profile_id}`

This is the main interactive demo endpoint.

It:

1. validates source and target platforms
2. selects the correct pairwise model family
3. builds a candidate pool in the target platform
4. scores all candidates
5. sorts by the selected model
6. returns:
   - ranked candidates
   - model scores
   - feature values
   - graph nodes and edges
   - model winners for each candidate

### 16.6 Frontend behavior

The frontend is intentionally lightweight and does not use a heavy JavaScript framework.

The JavaScript file:

1. loads overview statistics
2. fetches searchable profile lists
3. lets the user choose source profile, target platform, and model
4. requests candidate linkage results
5. renders:
   - hero statistics
   - result tables
   - candidate graph
   - evidence inspector

This keeps deployment simple while still providing an interactive research demo.

## 17. Testing

Test files:

- `/home/sidharth/Desktop/final_project/tests/test_pipeline.py`
- `/home/sidharth/Desktop/final_project/tests/test_webapp.py`

### 17.1 Pipeline tests

The pipeline tests validate:

- raw dataset parsing
- processed JSONL round-trip behavior
- pairwise experiment execution on a synthetic fixture

### 17.2 Web tests

The web test validates:

- `/api/overview`
- `/api/profiles`
- `/api/link`

using a stubbed service so endpoint behavior can be checked without running the full model pipeline.

## 18. File-by-File Responsibilities

### Core data and configuration

- `schema.py`: dataclasses and configuration
- `data.py`: dataset traversal, parsing, grouping, and JSONL I/O
- `utils.py`: normalization and similarity primitives

### Retrieval and feature engineering

- `candidates.py`: candidate pruning using merged bigram Jaccard
- `features.py`: feature store and feature vectors
- `semantic.py`: semantic embedding creation and caching

### Training and evaluation

- `models.py`: ranker definitions
- `evaluation.py`: negative sampling, training, scoring, pairwise evaluation, multi-platform evaluation, and result writing
- `cli.py`: command-line entrypoints

### Demo system

- `webapp.py`: FastAPI backend and interactive demo service
- `web/index.html`: page structure
- `web/styles.css`: visual design
- `web/app.js`: frontend data fetching and rendering logic

## 19. Important Design Decisions

### 19.1 Why profile-only modeling was chosen

This project uses profile-only data because that is the richest reliably available signal in the public LinkSocial dataset.

### 19.2 Why the project includes both classical and modern models

The final-year-project goal is not only to reproduce the original paper, but also to analyze how far profile-only linkage can be improved with stronger modern methods.

### 19.3 Why candidate pruning is retained

Even with stronger models, candidate pruning remains important because:

- it matches the original problem framing
- it reduces compute
- it lets the project report both retrieval quality and ranking quality

### 19.4 Why the frontend uses FastAPI plus static assets

This keeps the system:

- simple to run
- aligned with `uv`
- easier to document
- easier to demonstrate during evaluation

## 20. Current Limitations

The implementation has several important limitations.

### 20.1 Dataset limitations

The public dataset does not fully support:

- graph-based UIL
- multimodal post-aware UIL
- temporal or mobility-based UIL
- post-level stylometry

### 20.2 Image limitation

The original paper discusses image similarity, but the current executable project does not include a complete image-feature pipeline because the dataset release is not structured for that end-to-end setup.

### 20.3 Frontend limitation

The frontend is designed for research demonstration and inspection, not production deployment.

### 20.4 Pairwise direction simplification

The experiments train one model family per unordered platform pair for practical efficiency rather than maintaining separate directional models for every direction.

## 21. Current Best Performing Model

Based on the current experiment outputs:

- best pairwise model varies slightly by task
- best overall multi-platform model is `semantic_hybrid_gbdt`

This means the strongest practical method in the current codebase is a combination of:

- LINKSOCIAL-style classical attributes
- stronger lexical similarity
- transformer-based semantic profile understanding

## 22. How To Run The Full Project

### Step 1

Install dependencies:

```bash
uv sync --dev
```

### Step 2

Prepare processed profiles:

```bash
uv run linksocial prepare-data
```

### Step 3

Run experiments:

```bash
uv run linksocial run-experiments \
  --max-candidates 40 \
  --min-candidates 20 \
  --cluster-ratio 0.005 \
  --semantic-model-name BAAI/bge-small-en-v1.5 \
  --semantic-batch-size 128
```

### Step 4

Start the frontend:

```bash
uv run linksocial serve-web --host 127.0.0.1 --port 8000
```

### Step 5

Open:

```text
http://127.0.0.1:8000
```

### Step 6

Run tests if needed:

```bash
uv run pytest
```

## 23. Final Summary

This implementation is a complete research-oriented identity-linkage system built around the public LinkSocial dataset. It covers:

- data ingestion
- canonical transformation
- candidate generation
- classical feature engineering
- lexical-modern feature engineering
- semantic embedding construction
- supervised ranking
- pairwise evaluation
- multi-platform evaluation
- result export
- interactive frontend-based demonstration

For a final-year project, this gives both:

- a faithful technical base connected to the original LINKSOCIAL paper
- a stronger modern extension with clear comparative results
