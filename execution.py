"""Automated execution helpers for AI-directed signals and portfolio actions."""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
from ib_async import IB, LimitOrder, MarketOrder, Option, Stock
from loguru import logger

from notifications import SlackNotifier
from option_data import OptionChainSnapshot
from strategies.base import TradeSignal


@dataclass(slots=True)
class TradeExecutionConfig:
    """Settings controlling finalist-to-order execution."""

    enabled: bool = False
    default_quantity: int = 1
    limit_padding_pct: float = 0.05
    max_orders: int = 5
    max_spread_pct: float = 0.6
    allow_market_fallback: bool = True


@dataclass(slots=True)
class PortfolioExecutionConfig:
    """Settings controlling execution of AI portfolio adjustments."""

    enabled: bool = False
    max_positions: int = 5
    use_market_orders: bool = False
    limit_padding_pct: float = 0.03


@dataclass(slots=True)
class OrderPlan:
    """Normalized instruction for an order submission."""

    description: str
    contract: object
    order: object
    context: Dict[str, object]


class TradeExecutor:
    """Translate AI-selected finalists into IBKR orders."""

    def __init__(
        self,
        ib: IB,
        config: TradeExecutionConfig,
        slack: Optional[SlackNotifier] = None,
    ) -> None:
        self._ib = ib
        self._config = config
        self._slack = slack

    def execute_finalists(
        self,
        finalists: Sequence[Tuple[str, TradeSignal, Optional[float], Optional[str]]],
        snapshots: Dict[str, OptionChainSnapshot],
    ) -> List[str]:
        logger.info(
            "Starting finalist execution run | enabled={enabled} finalists={count} max_orders={max_orders}",
            enabled=self._config.enabled,
            count=len(finalists),
            max_orders=self._config.max_orders,
        )
        if not self._config.enabled:
            logger.debug("Trade executor disabled; skipping finalist execution")
            return []
        reports: List[str] = []
        for idx, (strategy, signal, score, reason) in enumerate(finalists, start=1):
            if idx > self._config.max_orders:
                logger.info(
                    "Skipping finalist execution due to max order cap | symbol={symbol} strategy={strategy}",
                    symbol=signal.symbol,
                    strategy=strategy,
                )
                break
            snapshot = snapshots.get(signal.symbol.upper())
            plan = self._build_plan(signal, snapshot)
            if plan is None:
                continue
            result = self._submit(plan)
            reports.append(result)
            logger.info(
                "Finalist executed | symbol={symbol} strategy={strategy} result={result}",
                symbol=signal.symbol,
                strategy=strategy,
                result=result,
            )
        if reports and self._slack and self._slack.enabled:
            message = "\n".join(["*AI Finalist Executions*"] + reports)
            payload = self._slack._build_payload(message)
            try:
                self._slack._post(self._slack.settings.webhook_url, payload)
            except Exception:
                logger.warning("Failed to post finalist executions to Slack", exc_info=True)
        logger.info(
            "Finished finalist execution run | executed={executed} finalists={count}",
            executed=len(reports),
            count=len(finalists),
        )
        return reports

    def _build_plan(
        self, signal: TradeSignal, snapshot: Optional[OptionChainSnapshot]
    ) -> Optional[OrderPlan]:
        try:
            expiry = signal.expiry if isinstance(signal.expiry, datetime) else None
            expiry_str = expiry.strftime("%Y%m%d") if expiry else str(signal.expiry)
            right = self._resolve_right(signal.option_type)
            if right is None:
                logger.warning("Unable to resolve option right for signal | signal={signal}", signal=signal)
                return None
            contract = Option(signal.symbol, expiry_str, float(signal.strike), right, "SMART", currency="USD")
            qualified = self._ib.qualifyContracts(contract)
            if not qualified:
                logger.warning(
                    "Failed to qualify option contract; skipping order | symbol={symbol} expiry={expiry} strike={strike} right={right}",
                    symbol=signal.symbol,
                    expiry=expiry_str,
                    strike=signal.strike,
                    right=right,
                )
                return None
            contract = qualified[0]
            quantity = max(int(self._config.default_quantity), 1)
            limit_price = self._derive_price(signal, snapshot)
            side = self._resolve_side(signal.direction)
            order = self._build_order(side, quantity, limit_price)
            description = (
                f"{signal.symbol} {right}{signal.strike} exp {expiry_str} qty {quantity} "
                f"action {order.action} type {order.orderType}"
            )
            context = {
                "signal_direction": signal.direction,
                "option_type": signal.option_type,
                "limit_price": limit_price,
            }
            return OrderPlan(description=description, contract=contract, order=order, context=context)
        except Exception:
            logger.exception("Failed to assemble order plan | signal={signal}", signal=signal)
            return None

    def _build_order(self, side: str, quantity: int, limit_price: Optional[float]):
        if limit_price is None or limit_price <= 0:
            if self._config.allow_market_fallback:
                return MarketOrder(side, quantity)
            raise ValueError("No viable price for limit order and market fallback disabled")
        return LimitOrder(side, quantity, round(limit_price, 2))

    def _derive_price(
        self, signal: TradeSignal, snapshot: Optional[OptionChainSnapshot]
    ) -> Optional[float]:
        if snapshot is None:
            return None
        try:
            matches = [
                row
                for row in snapshot.options
                if str(row.get("option_type", "")).upper().startswith(signal.option_type[:1].upper())
                and math.isclose(float(row.get("strike", 0.0) or 0.0), float(signal.strike), rel_tol=1e-4)
                and _compare_dates(row.get("expiry"), signal.expiry)
            ]
        except Exception:
            logger.debug("Failed to search snapshot for price | symbol={symbol}", symbol=signal.symbol)
            matches = []
        if not matches:
            return None
        row = matches[0]
        bid = float(row.get("bid", 0.0) or 0.0)
        ask = float(row.get("ask", 0.0) or 0.0)
        mark = float(row.get("mark", 0.0) or 0.0)
        mid = mark or (bid + ask) / 2 if (bid and ask) else max(bid, ask)
        if mid and bid and ask:
            spread_pct = (ask - bid) / mid if mid else 0.0
            if spread_pct > self._config.max_spread_pct:
                logger.warning(
                    "Skipping order due to wide spread | symbol={symbol} bid={bid} ask={ask} spread_pct={spread}",
                    symbol=signal.symbol,
                    bid=bid,
                    ask=ask,
                    spread=spread_pct,
                )
                return None
        if not mid:
            return None
        padding = 1 + self._config.limit_padding_pct
        return mid * padding

    @staticmethod
    def _resolve_right(option_type: str) -> Optional[str]:
        if not option_type:
            return None
        upper = option_type.upper()
        if upper.startswith("C"):
            return "C"
        if upper.startswith("P"):
            return "P"
        return None

    @staticmethod
    def _resolve_side(direction: Optional[str]) -> str:
        if not direction:
            return "BUY"
        text = direction.upper()
        if any(token in text for token in ("SELL", "SHORT")):
            return "SELL"
        return "BUY"

    def _submit(self, plan: OrderPlan) -> str:
        try:
            trade = self._ib.placeOrder(plan.contract, plan.order)
            order_id = getattr(getattr(trade, "order", None), "orderId", None)
            status = getattr(getattr(trade, "orderStatus", None), "status", "submitted")
            return f"{plan.description} | orderId={order_id} status={status}"
        except Exception as exc:
            logger.warning(
                "IBKR order submission failed | description={description} error={error}",
                description=plan.description,
                error=exc,
            )
            return f"{plan.description} | failed: {exc}"


class PortfolioActionExecutor:
    """Execute AI-suggested hedges or exits from Gemini responses."""

    def __init__(
        self, ib: IB, config: PortfolioExecutionConfig, slack: Optional[SlackNotifier] = None
    ) -> None:
        self._ib = ib
        self._config = config
        self._slack = slack

    def execute_ai_response(self, positions: pd.DataFrame, response: Optional[str]) -> List[str]:
        logger.info(
            "Starting portfolio execution run | enabled={enabled} positions={positions} has_response={has_response}",
            enabled=self._config.enabled,
            positions=len(positions) if positions is not None else 0,
            has_response=bool(response),
        )
        if not self._config.enabled:
            logger.debug("Portfolio executor disabled; skipping execution")
            return []
        if not response:
            logger.info("Portfolio executor: no AI response to act on")
            return []
        targets = self._extract_targets(positions, response)
        if not targets:
            logger.info("No actionable portfolio targets detected in AI response")
            return []
        reports: List[str] = []
        for idx, (_, row) in enumerate(targets.iterrows(), start=1):
            if idx > self._config.max_positions:
                logger.info("Skipping further portfolio actions due to cap")
                break
            plan = self._build_close_plan(row)
            if plan is None:
                continue
            result = self._submit(plan)
            reports.append(result)
        if reports and self._slack and self._slack.enabled:
            payload = self._slack._build_payload("\n".join(["*AI Portfolio Actions*"] + reports))
            try:
                self._slack._post(self._slack.settings.webhook_url, payload)
            except Exception:
                logger.warning("Failed to post portfolio executions to Slack", exc_info=True)
        logger.info(
            "Finished portfolio execution run | executed={executed} targets={targets}",
            executed=len(reports),
            targets=len(targets),
        )
        return reports

    def _extract_targets(self, positions: pd.DataFrame, response: str) -> pd.DataFrame:
        response_upper = response.upper()
        symbols = positions.get("underlying") or positions.get("symbol")
        if symbols is None:
            return positions.iloc[0:0]
        mask = []
        for symbol in symbols:
            symbol_str = str(symbol or "").upper()
            if not symbol_str:
                mask.append(False)
                continue
            should_close = symbol_str in response_upper and any(
                keyword in response_upper for keyword in ("CLOSE", "EXIT", "REDUCE", "TRIM")
            )
            mask.append(should_close)
        if not any(mask):
            return positions.iloc[0:0]
        return positions.loc[mask]

    def _build_close_plan(self, row: pd.Series) -> Optional[OrderPlan]:
        try:
            quantity = float(row.get("quantity", 0.0) or 0.0)
            if quantity == 0:
                return None
            side = "SELL" if quantity > 0 else "BUY"
            abs_qty = max(int(abs(quantity)), 1)
            sec_type = str(row.get("sec_type", "OPT") or "OPT").upper()
            if sec_type == "STK":
                contract = Stock(str(row.get("underlying")), "SMART", "USD")
            else:
                expiry_raw = row.get("expiry") or ""
                expiry_str = expiry_raw if isinstance(expiry_raw, str) else pd.to_datetime(expiry_raw).strftime("%Y%m%d")
                right = str(row.get("right", "")).upper() or "C"
                strike = float(row.get("strike", 0.0) or 0.0)
                contract = Option(str(row.get("underlying")), expiry_str, strike, right[:1], "SMART", currency="USD")
            price = self._derive_exit_price(row)
            order = self._build_portfolio_order(side, abs_qty, price)
            description = (
                f"Close {abs_qty} {row.get('underlying')} {row.get('right')}{row.get('strike')} "
                f"exp {row.get('expiry')} action {order.action} type {order.orderType}"
            )
            return OrderPlan(description=description, contract=contract, order=order, context={})
        except Exception:
            logger.exception("Failed to assemble portfolio close plan | row={row}", row=row)
            return None

    def _derive_exit_price(self, row: pd.Series) -> Optional[float]:
        try:
            mark = float(row.get("market_price", 0.0) or 0.0)
        except Exception:
            mark = 0.0
        if mark <= 0:
            return None
        factor = 1 + self._config.limit_padding_pct if row.get("quantity", 0.0) > 0 else 1 - self._config.limit_padding_pct
        return max(mark * factor, 0.01)

    def _build_portfolio_order(self, side: str, quantity: int, price: Optional[float]):
        if self._config.use_market_orders or price is None or price <= 0:
            return MarketOrder(side, quantity)
        return LimitOrder(side, quantity, round(price, 2))

    def _submit(self, plan: OrderPlan) -> str:
        try:
            trade = self._ib.placeOrder(plan.contract, plan.order)
            order_id = getattr(getattr(trade, "order", None), "orderId", None)
            status = getattr(getattr(trade, "orderStatus", None), "status", "submitted")
            return f"{plan.description} | orderId={order_id} status={status}"
        except Exception as exc:
            logger.warning(
                "IBKR portfolio order submission failed | description={description} error={error}",
                description=plan.description,
                error=exc,
            )
            return f"{plan.description} | failed: {exc}"


def _compare_dates(option_expiry: object, signal_expiry: object) -> bool:
    try:
        option_date = pd.to_datetime(option_expiry).date()
        signal_date = pd.to_datetime(signal_expiry).date()
        return option_date == signal_date
    except Exception:
        return False


__all__ = [
    "TradeExecutionConfig",
    "PortfolioExecutionConfig",
    "TradeExecutor",
    "PortfolioActionExecutor",
]
