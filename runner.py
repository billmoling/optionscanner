"""Execution helpers for running the option scanner."""
from __future__ import annotations

import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from loguru import logger
from zoneinfo import ZoneInfo

from ai_agents import SignalExplainAgent, SignalValidationAgent
from notifications import SlackNotifier
from option_data import BaseDataFetcher, OptionChainSnapshot
from position_cache import ExitRecommendation, PositionCache
from stock_data import StockDataFetcher
from technical_indicators import TechnicalIndicatorProcessor
from market_state import DictMarketStateProvider, MarketStateClassifier, MarketStateResult
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
    enable_gemini: bool = True,
    stock_fetcher: Optional[StockDataFetcher] = None,
    indicator_processor: Optional[TechnicalIndicatorProcessor] = None,
    stock_history_kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    symbol_list = list(symbols)
    underlying_context: Dict[str, Dict[str, Any]] = {}
    market_state_results: Dict[str, MarketStateResult] = {}
    if stock_fetcher is not None:
        underlying_context, market_state_results = await _fetch_underlying_context(
            stock_fetcher,
            symbol_list,
            indicator_processor=indicator_processor,
            history_kwargs=stock_history_kwargs or {},
        )
    snapshots = await fetcher.fetch_all(symbol_list)
    if underlying_context:
        for snapshot in snapshots:
            context = underlying_context.get(snapshot.symbol.upper())
            if context:
                if snapshot.context is None:
                    snapshot.context = {}
                snapshot.context.update(context)
                state_result = market_state_results.get(snapshot.symbol.upper())
                if state_result:
                    snapshot.context.setdefault("market_state", state_result.state.value)
                    snapshot.context.setdefault("market_state_as_of", state_result.as_of.isoformat())
    snapshot_by_symbol = {snapshot.symbol.upper(): snapshot for snapshot in snapshots}
    cache = PositionCache(results_dir / "position_cache.json")
    state_provider = None
    if market_state_results:
        state_provider = DictMarketStateProvider(
            {symbol: result.state for symbol, result in market_state_results.items()}
        )
        logger.info(
            "Market state provider initialized | symbols={symbols}",
            symbols=",".join(sorted(market_state_results.keys())),
        )
        for strategy in strategies:
            if hasattr(strategy, "market_state_provider"):
                strategy.market_state_provider = state_provider
    aggregated_signals: List[Tuple[str, TradeSignal]] = []
    for strategy in strategies:
        try:
            signals = strategy.on_data(snapshots)
            for signal in signals:
                aggregated_signals.append((strategy.name, signal))
                cache.record_signal(strategy.name, signal, snapshot_by_symbol.get((signal.symbol or "").upper()))
        except Exception:
            logger.exception("Strategy {name} failed", name=strategy.name)
    exit_recommendations = cache.evaluate_exits(snapshot_by_symbol)
    cache.save()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if exit_recommendations:
        _export_exit_recommendations(results_dir, exit_recommendations, timestamp)
    if not aggregated_signals:
        logger.info("No trade signals generated in this iteration")
        return
    results_dir.mkdir(parents=True, exist_ok=True)
    if enable_gemini and explain_agent is None:
        explain_agent = SignalExplainAgent(enable_gemini=True)
    if enable_gemini and validation_agent is None:
        validation_agent = SignalValidationAgent(enable_gemini=True)
    rows: List[Dict[str, Any]] = []
    signals_only = [signal for _strategy_name, signal in aggregated_signals]
    for strategy_name, signal in aggregated_signals:
        snapshot = snapshot_by_symbol.get((signal.symbol or "").upper())
        row = asdict(signal)
        row.update(
            {
                "strategy": strategy_name,
            }
        )
        if explain_agent:
            row["explanation"] = explain_agent.explain(signal, snapshot)
        if validation_agent:
            row["validation"] = validation_agent.review(signal, snapshot, signals_only)
        rows.append(row)
    df = pd.DataFrame(rows)
    preferred_order = [
        "symbol",
        "expiry",
        "strike",
        "option_type",
        "strategy",
        "direction",
        "rationale",
        "explanation",
        "validation",
    ]
    ordered_columns = [col for col in preferred_order if col in df.columns]
    remaining_columns = [col for col in df.columns if col not in ordered_columns]
    df = df[ordered_columns + remaining_columns]
    file_path = results_dir / f"signals_{timestamp}.csv"
    df.to_csv(file_path, index=False)
    logger.info("Saved {count} signals to {path}", count=len(df), path=str(file_path))

    if slack_notifier:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, slack_notifier.send_signals, df, file_path)


async def _fetch_underlying_context(
    stock_fetcher: StockDataFetcher,
    symbols: Sequence[str],
    *,
    indicator_processor: Optional[TechnicalIndicatorProcessor] = None,
    history_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, MarketStateResult]]:
    history_kwargs = history_kwargs or {}
    try:
        histories = await stock_fetcher.fetch_history_many(symbols, **history_kwargs)
    except Exception:
        logger.exception("Failed to download underlying stock history")
        return {}, {}

    processor = indicator_processor or TechnicalIndicatorProcessor()
    processor.ensure_default_moving_averages()
    classifier = MarketStateClassifier()
    context: Dict[str, Dict[str, Any]] = {}
    state_results: Dict[str, MarketStateResult] = {}

    for symbol, history in histories.items():
        if history is None or history.empty:
            continue
        try:
            enriched = processor.process(history)
        except Exception:
            logger.exception("Failed to compute indicators for {symbol}", symbol=symbol)
            continue
        enriched = enriched.dropna(subset=["close"])
        if enriched.empty:
            continue
        latest = enriched.iloc[-1]
        metrics: Dict[str, Any] = {}
        close_value = latest.get("close")
        if pd.notna(close_value):
            metrics["close"] = float(close_value)
        for column in ("ma5", "ma10", "ma30", "ma50"):
            value = latest.get(column)
            if pd.notna(value):
                metrics[column] = float(value)
                if column == "ma30":
                    metrics["moving_average_30"] = float(value)
                elif column == "ma50":
                    metrics["moving_average_50"] = float(value)
        timestamp_value = latest.get("timestamp")
        if pd.notna(timestamp_value):
            timestamp = pd.to_datetime(timestamp_value, utc=True)
            metrics["indicator_timestamp"] = timestamp.isoformat()

        state_result = classifier.classify(enriched, symbol=symbol)
        if state_result:
            state_results[symbol.upper()] = state_result
            metrics["market_state"] = state_result.state.value
            metrics["market_state_as_of"] = state_result.as_of.isoformat()

        if not metrics:
            continue
        context[symbol.upper()] = metrics
    return context, state_results


def _export_exit_recommendations(
    results_dir: Path,
    recommendations: List[ExitRecommendation],
    timestamp: str,
) -> None:
    rows = [
        {
            "symbol": rec.symbol,
            "strategy": rec.strategy,
            "direction": rec.direction,
            "strike": rec.strike,
            "expiry": rec.expiry,
            "reason": rec.reason,
            "action": rec.action,
        }
        for rec in recommendations
    ]
    if not rows:
        return
    results_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    path = results_dir / f"exits_{timestamp}.csv"
    df.to_csv(path, index=False)
    logger.info(
        "Generated {count} exit recommendations | path={path}",
        count=len(df),
        path=str(path),
    )


async def run_scheduler(
    config: Dict[str, Any],
    fetcher: BaseDataFetcher,
    strategies: List[BaseOptionStrategy],
    symbols: Sequence[str],
    results_dir: Path,
    slack_notifier: Optional[SlackNotifier],
    *,
    enable_gemini: bool = True,
    stock_fetcher: Optional[StockDataFetcher] = None,
    indicator_processor: Optional[TechnicalIndicatorProcessor] = None,
    stock_history_kwargs: Optional[Dict[str, Any]] = None,
    post_run: Optional[Callable[[], None]] = None,
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

    explain_agent = SignalExplainAgent(enable_gemini=enable_gemini) if enable_gemini else None
    validation_agent = SignalValidationAgent(enable_gemini=enable_gemini) if enable_gemini else None

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
                enable_gemini=enable_gemini,
                stock_fetcher=stock_fetcher,
                indicator_processor=indicator_processor,
                stock_history_kwargs=stock_history_kwargs,
            )
            if post_run is not None:
                try:
                    post_run()
                except Exception:
                    logger.exception("Post-run callback failed")
        except Exception:
            logger.exception("Run failed")
        duration = (datetime.now(tz) - start).total_seconds()
        logger.info("Run completed in {duration:.2f}s", duration=duration)


__all__ = ["run_once", "run_scheduler"]
