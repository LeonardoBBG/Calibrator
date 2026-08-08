"""Dependency-light command line interface for individual stages and full runs."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence

from .orchestrator import PipelineOrchestrator

LOGGER = logging.getLogger("sol_fable.cli")


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero (uncapped) or a positive integer")
    return parsed


def _add_global_arguments(parser: argparse.ArgumentParser, *, on_subcommand: bool = False) -> None:
    """Add invocation-wide options to the root parser or one of its subcommands.

    Argparse normally stops accepting root-parser options after it encounters a
    subcommand.  Mirroring the options on each subparser lets users put them in
    either position.  Suppressed subparser defaults are important: when an
    option appears only before the subcommand, an absent duplicate must not
    overwrite the value already parsed by the root parser.
    """

    default = argparse.SUPPRESS if on_subcommand else None
    parser.add_argument(
        "--project-root",
        type=Path,
        default=default,
        help=(
            "Project/data root (defaults to SOL_FABLE_DATA_DIR, the source checkout, "
            "or a writable user data directory)"
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=default,
        help="Alternative settings YAML",
    )
    parser.add_argument(
        "--run-id",
        default=default,
        help="Run to continue; defaults to latest",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=argparse.SUPPRESS if on_subcommand else False,
    )
    parser.add_argument(
        "--analysis-backend",
        choices=["deterministic-v1", "ollama", "openai", "anthropic", "dual-api"],
        default=default,
        help="Override config/settings.yaml's analysis_backend for this invocation.",
    )
    parser.add_argument(
        "--ollama-model",
        default=default,
        help="Ollama model tag, e.g. gemma3:27b",
    )
    parser.add_argument(
        "--ollama-host",
        default=default,
        help="Ollama host URL",
    )
    parser.add_argument(
        "--ollama-num-gpu",
        type=_non_negative_int,
        default=default,
        help=(
            "GPU layers requested from Ollama; a high value such as 999 requests full "
            "offload and is clamped to the model's layer count. Zero forces CPU."
        ),
    )
    parser.add_argument(
        "--openai-model",
        default=default,
        help="OpenAI model name (API key is read from OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--anthropic-model",
        default=default,
        help="Anthropic model name (API key is read from ANTHROPIC_API_KEY)",
    )
    parser.add_argument(
        "--ollama-argument-cap",
        type=_non_negative_int,
        default=default,
        help=(
            "When using the ollama backend, only this many arguments (highest importance "
            "first) get the real LLM debate; the rest are auto-filled deterministically. "
            "Deprecated alias for --live-argument-cap; zero means uncapped."
        ),
    )
    parser.add_argument(
        "--live-argument-cap",
        type=_non_negative_int,
        default=default,
        help=(
            "Maximum arguments sent to any live provider; remaining arguments are completed "
            "deterministically. Defaults to the safe configured cap; zero explicitly uncaps."
        ),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sol-fable",
        description="Auditable WS argument calibrator",
    )
    _add_global_arguments(parser)
    sub = parser.add_subparsers(dest="command", required=True)

    def command_parser(command: str, **kwargs: object) -> argparse.ArgumentParser:
        item = sub.add_parser(command, **kwargs)
        _add_global_arguments(item, on_subcommand=True)
        return item

    def files(command: str) -> argparse.ArgumentParser:
        item = command_parser(command)
        item.add_argument("--et1", required=True, type=Path)
        item.add_argument("--et3", required=True, type=Path)
        item.add_argument("--ws", required=True, type=Path)
        return item

    files("ingest")
    run_all = files("run-all")
    run_all.add_argument("--n-rounds", type=int, default=None)
    run_all.add_argument("--claimant-barrister", choices=["SOL", "FABLE"], default=None)
    command_parser("parse-references")
    command_parser("build-issues")
    command_parser("extract-propositions")
    command_parser("match-pleadings")
    command_parser("build-arguments")
    assess = command_parser("assess")
    assess.add_argument("--n-rounds", type=int, default=None)
    assess.add_argument("--claimant-barrister", choices=["SOL", "FABLE"], default=None)
    command_parser("rank")
    command_parser("generate-search-packages")
    command_parser("report")
    command_parser("status")
    command_parser("list-models", help="List models pulled in the local Ollama instance")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    command = args.command

    if command == "list-models":
        from .llm_backend import list_ollama_models

        host = args.ollama_host or "http://localhost:11434"
        print(json.dumps(list_ollama_models(host), indent=2))
        return 0

    explicit_backend = None
    effective_settings = None
    overrides = {
        **({"analysis_backend": args.analysis_backend} if args.analysis_backend else {}),
        **({"ollama_model": args.ollama_model} if args.ollama_model else {}),
        **({"ollama_host": args.ollama_host} if args.ollama_host else {}),
        **(
            {"ollama_num_gpu": args.ollama_num_gpu}
            if args.ollama_num_gpu is not None
            else {}
        ),
        **({"openai_model": args.openai_model} if args.openai_model else {}),
        **({"anthropic_model": args.anthropic_model} if args.anthropic_model else {}),
    }
    if overrides:
        from .config import load_settings

        settings = load_settings(args.config)
        effective_settings = type(settings).model_validate(
            {**settings.model_dump(mode="python"), **overrides}
        )
        if command in {"assess", "run-all"}:
            from .llm_backend import build_backend

            explicit_backend = build_backend(effective_settings)
    orchestrator = PipelineOrchestrator(args.project_root, args.config, backend=explicit_backend)
    if effective_settings is not None:
        # The backend override is part of the run's audit configuration, not
        # merely a construction detail.  Paths are unchanged by these updates.
        orchestrator.settings = effective_settings

    def _progress(done: int, total: int) -> None:
        LOGGER.info("assess_progress: %d/%d arguments", done, total)

    if command == "run-all":
        run_id = orchestrator.run_all(
            args.et1,
            args.et3,
            args.ws,
            n_rounds=args.n_rounds,
            claimant_barrister=args.claimant_barrister,
            ollama_argument_cap=args.ollama_argument_cap,
            progress_callback=_progress,
            live_argument_cap=args.live_argument_cap,
        )
    elif command == "ingest":
        run_id = orchestrator.ingest(args.et1, args.et3, args.ws, args.run_id)
    elif command == "parse-references":
        run_id = orchestrator.parse_references(args.run_id)
    elif command == "build-issues":
        run_id = orchestrator.build_issues(args.run_id)
    elif command == "extract-propositions":
        run_id = orchestrator.extract_propositions(args.run_id)
    elif command == "match-pleadings":
        run_id = orchestrator.match_pleadings(args.run_id)
    elif command == "build-arguments":
        run_id = orchestrator.build_arguments(args.run_id)
    elif command == "assess":
        run_id = orchestrator.assess(
            args.run_id,
            n_rounds=args.n_rounds,
            claimant_barrister=args.claimant_barrister,
            ollama_argument_cap=args.ollama_argument_cap,
            progress_callback=_progress,
            live_argument_cap=args.live_argument_cap,
        )
    elif command == "rank":
        run_id = orchestrator.rank(args.run_id)
    elif command == "generate-search-packages":
        run_id = orchestrator.generate_search_packages(args.run_id)
    elif command == "report":
        run_id = orchestrator.report(args.run_id)
    else:
        run_id = orchestrator._require_run(args.run_id)
        payload = {
            "run": orchestrator.store.load_run(run_id).model_dump(mode="json"),
            "stages": orchestrator.store.stage_events(run_id),
        }
        print(json.dumps(payload, indent=2))
        return 0
    print(run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
