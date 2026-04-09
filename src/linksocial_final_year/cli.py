from __future__ import annotations

import argparse
from pathlib import Path

from .data import load_raw_profiles, read_profiles_jsonl, write_profiles_jsonl
from .evaluation import run_multi_platform_experiment, run_pair_experiment, write_results
from .features import FeatureStore
from .schema import ExperimentConfig


def prepare_data(config: ExperimentConfig) -> None:
    profiles = load_raw_profiles(config.raw_dir)
    write_profiles_jsonl(profiles, config.processed_path)
    print(f"Prepared {len(profiles)} profiles into {config.processed_path}")


def run_experiments(config: ExperimentConfig) -> None:
    if not config.processed_path.exists():
        prepare_data(config)

    profiles = read_profiles_jsonl(config.processed_path)
    store = FeatureStore(
        profiles,
        semantic_model_name=config.semantic_model_name,
        semantic_cache_dir=config.semantic_cache_dir,
        semantic_batch_size=config.semantic_batch_size,
    )

    pair_results = []
    pair_models = {}
    pair_specs = [
        ("google_plus", "instagram"),
        ("google_plus", "twitter"),
        ("instagram", "twitter"),
    ]
    for source_platform, target_platform in pair_specs:
        result, models, task = run_pair_experiment(profiles, store, config, source_platform, target_platform)
        pair_results.append(result)
        pair_models[tuple(sorted((source_platform, target_platform)))] = models
        print(
            f"{source_platform} -> {target_platform}: "
            f"baseline={result.metrics['baseline_accuracy']:.4f}, "
            f"sgd={result.metrics['linksocial_sgd_accuracy']:.4f}, "
            f"logreg={result.metrics['linksocial_logreg_accuracy']:.4f}, "
            f"rf={result.metrics['linksocial_rf_accuracy']:.4f}, "
            f"lexical_gbdt={result.metrics['lexical_modern_gbdt_accuracy']:.4f}, "
            + (
                f"semantic_cosine={result.metrics['semantic_cosine_accuracy']:.4f}, "
                f"semantic_hybrid={result.metrics['semantic_hybrid_gbdt_accuracy']:.4f}"
                if 'semantic_hybrid_gbdt_accuracy' in result.metrics
                else ""
            ),
            flush=True,
        )

    multi_result = run_multi_platform_experiment(profiles, store, config, pair_models)
    write_results(config, pair_results, multi_result)
    print(f"Saved results to {config.results_dir}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="linksocial")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_shared_arguments(command_parser: argparse.ArgumentParser) -> None:
        command_parser.add_argument("--raw-dir", type=Path, default=ExperimentConfig.raw_dir)
        command_parser.add_argument("--processed-path", type=Path, default=ExperimentConfig.processed_path)
        command_parser.add_argument("--results-dir", type=Path, default=ExperimentConfig.results_dir)
        command_parser.add_argument("--semantic-cache-dir", type=Path, default=ExperimentConfig.semantic_cache_dir)
        command_parser.add_argument("--seed", type=int, default=ExperimentConfig.seed)
        command_parser.add_argument("--cluster-ratio", type=float, default=ExperimentConfig.cluster_ratio)
        command_parser.add_argument("--max-candidates", type=int, default=ExperimentConfig.max_candidates)
        command_parser.add_argument("--min-candidates", type=int, default=ExperimentConfig.min_candidates)
        command_parser.add_argument("--semantic-model-name", type=str, default=ExperimentConfig.semantic_model_name)
        command_parser.add_argument("--semantic-batch-size", type=int, default=ExperimentConfig.semantic_batch_size)

    prepare_parser = subparsers.add_parser("prepare-data")
    add_shared_arguments(prepare_parser)

    experiment_parser = subparsers.add_parser("run-experiments")
    add_shared_arguments(experiment_parser)

    web_parser = subparsers.add_parser("serve-web")
    add_shared_arguments(web_parser)
    web_parser.add_argument("--host", type=str, default="127.0.0.1")
    web_parser.add_argument("--port", type=int, default=8000)

    return parser


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        raw_dir=args.raw_dir,
        processed_path=args.processed_path,
        results_dir=args.results_dir,
        semantic_cache_dir=args.semantic_cache_dir,
        seed=args.seed,
        cluster_ratio=args.cluster_ratio,
        max_candidates=args.max_candidates,
        min_candidates=args.min_candidates,
        semantic_model_name=args.semantic_model_name,
        semantic_batch_size=args.semantic_batch_size,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = config_from_args(args)
    if args.command == "prepare-data":
        prepare_data(config)
        return
    if args.command == "run-experiments":
        run_experiments(config)
        return
    if args.command == "serve-web":
        from .webapp import run_server

        run_server(config=config, host=args.host, port=args.port)
        return
    raise SystemExit(f"Unsupported command: {args.command}")
