"""Main entry point for the option scanner using NautilusTrader and IBKR."""
from __future__ import annotations

import argparse
import asyncio
import importlib
import os
import pkgutil
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml
from loguru import logger

from ai_agents import SignalExplainAgent, SignalValidationAgent
from docker_utils import DockerController, DockerControllerError
from dotenv import load_dotenv
from logging_utils import configure_logging
from notifications import SlackNotifier
from option_data import IBKRDataFetcher, MARKET_DATA_TYPE_CODES
from portfolio.manager import PortfolioManager
from runner import run_once, run_scheduler
from strategies.base import BaseOptionStrategy


class RunMode(str, Enum):
    """Supported execution modes for the scanner."""

    LOCAL_IMMEDIATE = "local"
    DOCKER_IMMEDIATE = "docker-immediate"
    DOCKER_SCHEDULED = "docker-scheduled"


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def discover_strategies() -> List[BaseOptionStrategy]:
    strategy_dir = Path(__file__).parent / "strategies"
    strategies: List[BaseOptionStrategy] = []
    for module_info in pkgutil.iter_modules([str(strategy_dir)]):
        if not module_info.name.startswith("strategy_"):
            continue
        module = importlib.import_module(f"strategies.{module_info.name}")
        for attr in dir(module):
            obj = getattr(module, attr)
            if (
                isinstance(obj, type)
                and issubclass(obj, BaseOptionStrategy)
                and obj is not BaseOptionStrategy
            ):
                try:
                    strategies.append(obj())
                except TypeError as exc:
                    logger.error(
                        "Failed to instantiate strategy {name}: {error}",
                        name=obj.__name__,
                        error=exc,
                    )
    logger.info("Loaded {count} strategies", count=len(strategies))
    return strategies


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Nautilus option scanner.")
    parser.add_argument(
        "--run-mode",
        choices=[mode.value for mode in RunMode],
        default=RunMode.LOCAL_IMMEDIATE.value,
        help="Select how the scanner executes: local (no Docker), docker-immediate, or docker-scheduled.",
    )
    parser.add_argument(
        "--market-data",
        choices=sorted(MARKET_DATA_TYPE_CODES.keys()),
        default="LIVE",
        help="Market data type requested from IBKR when using live data fetchers (LIVE or FROZEN).",
    )
    parser.add_argument(
        "--compose-file",
        type=Path,
        default=Path("docker-compose.yml"),
        help="Path to the docker-compose file used when starting Docker services.",
    )
    parser.add_argument(
        "--docker-service",
        default="ib-gateway",
        help="Name of the docker-compose service that hosts the IBKR gateway.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Configuration file for the scanner.",
    )
    return parser.parse_args(argv)


def execute_portfolio_manager(
    fetcher: IBKRDataFetcher,
    portfolio_config: Dict[str, Any],
) -> None:
    """Run the portfolio manager workflow with the provided fetcher."""
    try:
        slack_config = portfolio_config.get("slack") or portfolio_config.get("notifications")
        enable_gemini = bool(portfolio_config.get("enable_gemini", True))
        manager = PortfolioManager(
            fetcher.ib,
            config_path=portfolio_config.get("config_path", "risk.yaml"),
            slack_config=slack_config,
            enable_gemini=enable_gemini,
        )
        manager.load_positions()
        manager.compute_greeks()
        manager.evaluate_rules()
        manager.generate_actions()
        message = manager.notify()
        logger.info("Portfolio manager summary:\n{message}", message=message)
    except Exception:
        logger.exception("Portfolio manager execution failed")


def main(argv: Optional[List[str]] = None) -> None:
    load_dotenv()
    args = parse_args(argv)
    config = load_config(args.config)
    log_dir = Path(config.get("log_dir", "./logs"))
    configure_logging(log_dir, "strategy_signals")

    enable_gemini = bool(config.get("enable_gemini", True))
    results_dir = Path(config.get("results_dir", "./results"))
    slack_notifier = SlackNotifier(config.get("slack"))
    strategies = discover_strategies()
    symbols: Sequence[str] = config.get("tickers", [])
    run_mode = RunMode(args.run_mode)
    data_dir = Path(config.get("data_dir", "./data"))
    ibkr_settings = config.get("ibkr") or {}
    portfolio_settings = dict(config.get("portfolio", {}))
    portfolio_settings.setdefault("enable_gemini", enable_gemini)
    if "slack" not in portfolio_settings and config.get("slack"):
        portfolio_settings["slack"] = config.get("slack")
    host = ibkr_settings.get("host", "127.0.0.1")
    port = ibkr_settings.get("port")
    client_id = ibkr_settings.get("client_id", 1)
    disable_portfolio_manager = (
        os.getenv("DISABLE_PORTFOLIO_MANAGER", "").strip().lower() in {"1", "true", "yes", "on"}
    )

    explain_agent = SignalExplainAgent(enable_gemini=enable_gemini)
    validation_agent = SignalValidationAgent(enable_gemini=enable_gemini)

    def maybe_run_portfolio_manager(fetcher: IBKRDataFetcher) -> None:
        if disable_portfolio_manager:
            logger.info("Portfolio manager disabled via DISABLE_PORTFOLIO_MANAGER")
            return
        execute_portfolio_manager(fetcher, portfolio_settings)

    if port is None:
        raise ValueError("IBKR port is not configured. Set 'ibkr.port' in the configuration.")

    if run_mode is RunMode.LOCAL_IMMEDIATE:
        fetcher = IBKRDataFetcher(
            host=host,
            port=int(port),
            client_id=int(client_id),
            data_dir=data_dir,
            market_data_type=args.market_data,
        )
        try:
            asyncio.run(
                run_once(
                    fetcher,
                    strategies,
                    symbols,
                    results_dir,
                    explain_agent=explain_agent,
                    validation_agent=validation_agent,
                    slack_notifier=slack_notifier,
                    enable_gemini=enable_gemini,
                )
            )
            maybe_run_portfolio_manager(fetcher)
        except KeyboardInterrupt:
            logger.info("Shutdown requested by user")
        return

    
    docker_controller = DockerController(Path(args.compose_file))
    try:
        docker_controller.start_service(args.docker_service)
    except DockerControllerError as exc:
        logger.error("Unable to start Docker service: {error}", error=exc)
        raise SystemExit(1) from exc

    fetcher = IBKRDataFetcher(
        host=host,
        port=int(port),
        client_id=int(client_id),
        data_dir=data_dir,
        market_data_type=args.market_data,
    )

    try:
        if run_mode is RunMode.DOCKER_IMMEDIATE:
            asyncio.run(
                run_once(
                    fetcher,
                    strategies,
                    symbols,
                    results_dir,
                    explain_agent=explain_agent,
                    validation_agent=validation_agent,
                    slack_notifier=slack_notifier,
                    enable_gemini=enable_gemini,
                )
            )
            maybe_run_portfolio_manager(fetcher)
        else:
            asyncio.run(
                run_scheduler(
                    config,
                    fetcher,
                    strategies,
                    symbols,
                    results_dir,
                    slack_notifier,
                    enable_gemini=enable_gemini,
                    post_run=lambda: maybe_run_portfolio_manager(fetcher),
                )
            )
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")


if __name__ == "__main__":
    main()
