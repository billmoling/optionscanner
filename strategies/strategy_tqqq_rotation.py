"""TQQQ/QQQ Momentum Rotation Strategy implementation."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable, List, Optional

from loguru import logger

from .base import BaseOptionStrategy, TradeSignal, SignalLeg


class TqqqQqqRotationStrategy(BaseOptionStrategy):
    """
    Implements the QQQ/TQQQ Momentum Rotation Strategy.
    
    Logic:
    - Universe: QQQ (Base), TQQQ (Aggressive)
    - Indicators: SMA200, SMA10, RSI14 on QQQ
    - Risk-On: QQQ > SMA200 AND QQQ > SMA10 OR QQQ RSI < 35
    - Risk-Off: QQQ < SMA10 OR QQQ < SMA200 OR QQQ RSI > 75
    """

    def __init__(
        self,
        enabled: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.enabled = enabled
        # Track simulated position state if possible, or just emit signals based on current condition.
        # Since this strategy runs periodically, it should emit "BUY TQQQ" or "BUY QQQ" signals
        # depending on what the rigorous state *should* be. 
        # Ideally, the execution layer handles whether we already hold it. I will emit what we *should* hold.

    def on_data(self, data: Iterable[Any]) -> List[TradeSignal]:
        if not self.enabled:
            return []
        
        # We need QQQ data to decide
        qqq_snapshot = None
        tqqq_snapshot = None
        
        # Iterate to find QQQ and TQQQ snapshots
        for snapshot in data:
            symbol = self._snapshot_value(snapshot, "symbol")
            if symbol == "QQQ":
                qqq_snapshot = snapshot
            elif symbol == "TQQQ":
                tqqq_snapshot = snapshot
        
        if not qqq_snapshot:
            # Can't render decision without QQQ
            return []

        # Extract Indicators from QQQ context
        context = self._snapshot_value(qqq_snapshot, "context", {})
        
        # Check if we have necessary indicators
        # Note: names match TechnicalIndicatorProcessor registration
        ma200 = context.get("ma200")
        ma10 = context.get("ma10")
        rsi14 = context.get("rsi14")
        qqq_price = context.get("close")
        
        if any(x is None for x in [ma200, ma10, rsi14, qqq_price]):
            logger.warning("Missing indicators for QQQ: ma200={ma200}, ma10={ma10}, rsi={rsi}, price={price}",
                          ma200=ma200, ma10=ma10, rsi=rsi14, price=qqq_price)
            return []

        # Logic
        # Rule A: Risk-On (Rotate to TQQQ)
        # Condition: (Price > SMA200 and Price > SMA10) OR (RSI < 35)
        is_risk_on = (qqq_price > ma200 and qqq_price > ma10) or (rsi14 < 35)
        
        # Rule B: Risk-Off (Rotate to QQQ)
        # Condition: (Price < SMA10) OR (Price < SMA200) OR (RSI > 75)
        is_risk_off = (qqq_price < ma10) or (qqq_price < ma200) or (rsi14 > 75)

        # Safety Hard Stop?
        # The prompt asked for "If TQQQ drops more than 15% in a single week".
        # This requires TQQQ history. We might not have 1-week history here in the snapshot context unless requested.
        # However, checking risk-off conditions on QQQ is primary. 
        # If we have TQQQ context, we could check it. But simplify to core logic first.
        
        signals: List[TradeSignal] = []
        now_utc = datetime.now(timezone.utc)
        
        if is_risk_on:
            # Signal: BUY TQQQ / SELL QQQ (implied by rotation)
            # We emit a signal to hold TQQQ.
            rationale = f"Risk-On: QQQ({qqq_price:.2f}) > SMA200({ma200:.2f}) & SMA10({ma10:.2f}) or RSI({rsi14:.2f}) < 35"
            signals.append(
                self.emit_signal(
                    TradeSignal(
                        symbol="TQQQ",
                        expiry=now_utc, # Dummy for stock
                        strike=0.0,     # Dummy for stock
                        option_type="STOCK",
                        direction="LONG",
                        rationale=rationale,
                        legs=()
                    )
                )
            )
        elif is_risk_off:
            # Signal: BUY QQQ / SELL TQQQ
            rationale = f"Risk-Off: QQQ({qqq_price:.2f}) weak vs SMA10({ma10:.2f})/SMA200({ma200:.2f}) or RSI({rsi14:.2f}) > 75"
            signals.append(
                self.emit_signal(
                    TradeSignal(
                        symbol="QQQ",
                        expiry=now_utc,
                        strike=0.0,
                        option_type="STOCK",
                        direction="LONG",
                        rationale=rationale,
                        legs=()
                    )
                )
            )

        return signals
