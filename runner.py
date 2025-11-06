"""Execution helpers for running the option scanner."""
from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from loguru import logger
from zoneinfo import ZoneInfo

from ai_agents import SignalExplainAgent, SignalValidationAgent
from notifications import SlackNotifier
from option_data import BaseDataFetcher, OptionChainSnapshot
from scheduling import compute_next_run, parse_schedule_times
from strategies.base import BaseOptionStrategy, TradeSignal


async def run_once(
    fetcher: BaseDataFetcher,
    strategies: List[BaseOptionStrategy],
    symbols: Iterable[str],
    results_dir: Path,
    *,
    explain_agent: Optional[SignalExplainAgent] = None,
    validation_agent: Optional[SignalValidationAgent] = None,
    slack_notifier: Optional[SlackNotifier] = None,
) -> None:
    snapshots = await fetcher.fetch_all(symbols)
    aggregated_signals: List[Tuple[str, TradeSignal]] = []
    for strategy in strategies:
        try:
            signals = strategy.on_data(snapshots)
            for signal in signals:
                aggregated_signals.append((strategy.name, signal))
        except Exception:
            logger.exception("Strategy {name} failed", name=strategy.name)
    if not aggregated_signals:
        logger.info("No trade signals generated in this iteration")
        return
    results_dir.mkdir(parents=True, exist_ok=True)
    explain_agent = explain_agent or SignalExplainAgent()
    validation_agent = validation_agent or SignalValidationAgent()
    snapshot_by_symbol = {snapshot.symbol: snapshot for snapshot in snapshots}
    rows: List[Dict[str, Any]] = []
    signals_only = [signal for _strategy_name, signal in aggregated_signals]
    for strategy_name, signal in aggregated_signals:
        snapshot = snapshot_by_symbol.get(signal.symbol)
        explanation = explain_agent.explain(signal, snapshot)
        validation = validation_agent.review(signal, snapshot, signals_only)
        row = signal.__dict__.copy()
        row.update(
            {
                "explanation": explanation,
                "validation": validation,
                "strategy": strategy_name,
            }
        )
        rows.append(row)
    df = pd.DataFrame(rows)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    file_path = results_dir / f"signals_{timestamp}.csv"
    df.to_csv(file_path, index=False)
    logger.info("Saved {count} signals to {path}", count=len(df), path=str(file_path))

    if slack_notifier:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, slack_notifier.send_signals, df, file_path)


async def run_scheduler(
    config: Dict[str, Any],
    fetcher: BaseDataFetcher,
    strategies: List[BaseOptionStrategy],
    symbols: Sequence[str],
    results_dir: Path,
    slack_notifier: Optional[SlackNotifier],
) -> None:
    schedule_config = config.get("schedule", {})
    scheduled_times = parse_schedule_times(schedule_config)
    timezone_name = schedule_config.get("timezone", "America/Los_Angeles")

    try:
        tz = ZoneInfo(timezone_name)
    except Exception:
        logger.warning("Invalid timezone '{timezone}', defaulting to UTC", timezone=timezone_name)
        tz = ZoneInfo("UTC")

    logger.info(
        "Scheduling runs at {times} ({timezone})",
        times=", ".join(scheduled_time.strftime("%H:%M") for scheduled_time in scheduled_times),
        timezone=getattr(tz, "key", str(tz)),
    )

    explain_agent = SignalExplainAgent()
    validation_agent = SignalValidationAgent()

    while True:
        now = datetime.now(tz)
        next_run = compute_next_run(now, scheduled_times)
        sleep_seconds = max((next_run - now).total_seconds(), 0.0)
        logger.info(
            "Next run scheduled at {next_time} (sleeping {seconds:.2f}s)",
            next_time=next_run.isoformat(),
            seconds=sleep_seconds,
        )
        await asyncio.sleep(sleep_seconds)
        start = datetime.now(tz)
        logger.info("Starting scheduled run at {start}", start=start.isoformat())
        try:
            await run_once(
                fetcher,
                strategies,
                symbols,
                results_dir,
                explain_agent=explain_agent,
                validation_agent=validation_agent,
                slack_notifier=slack_notifier,
            )
        except Exception:
            logger.exception("Run failed")
        duration = (datetime.now(tz) - start).total_seconds()
        logger.info("Run completed in {duration:.2f}s", duration=duration)


__all__ = ["run_once", "run_scheduler"]
