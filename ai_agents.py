"""Utility agents for batching and validating trade signals."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from statistics import mean
from typing import Iterable, List, Optional

from loguru import logger

from gemini_client import GeminiClient, GeminiClientError
from option_data import OptionChainSnapshot
from strategies.base import TradeSignal


@dataclass(slots=True)
class GeminiSelection:
    """Represents a Gemini-chosen finalist from a batch review."""

    id: int
    score: Optional[float] = None
    reason: Optional[str] = None


@dataclass(slots=True)
class BatchSelectionResult:
    """Outcome of a batch Gemini selection call."""

    selections: List[GeminiSelection]
    prompt: Optional[str]
    response: Optional[str]


@dataclass(slots=True)
class SignalBatchSelector:
    """Rank a batch of signals at once and select the top candidates."""

    client: Optional[GeminiClient] = None
    enable_gemini: bool = True
    top_k: int = 5

    def select(self, signals: List[tuple[str, TradeSignal]]) -> BatchSelectionResult:
        if not self.enable_gemini or not signals:
            return BatchSelectionResult([], None, None)
        if self.client is None:
            self.client = GeminiClient()

        system_prompt = (
            "You are an options desk lead. Review a list of candidate option trades and pick the strongest ideas. "
            f"Select at most {self.top_k} finalists that balance risk/reward and liquidity. "
            "Return ONLY JSON with shape: "
            '{"finalists":[{"id":<number>,"score":<0-10 optional>,"reason":"concise why this idea wins"}],'
            '"summary":"one-sentence portfolio note"}. '
            "Use only IDs that appear in the list."
        )
        user_prompt = self._build_user_prompt(signals)
        try:
            response = self.client.generate(system_prompt=system_prompt, user_prompt=user_prompt)
        except GeminiClientError as exc:
            logger.warning("Batch Gemini selection failed | reason={error}", error=exc)
            return BatchSelectionResult([], user_prompt, None)

        selections = self._parse_response(response)
        if not selections:
            logger.warning("Gemini selection returned no finalists; response may be unstructured")
        return BatchSelectionResult(selections, user_prompt, response)

    def _build_user_prompt(self, signals: List[tuple[str, TradeSignal]]) -> str:
        lines = [
            "Evaluate these option trade signals. Consider directional edge, risk, and simplicity.",
            "Pick the top ideas only; skip low-quality trades.",
            "",
            "Signals:",
        ]
        for idx, (strategy, signal) in enumerate(signals, start=1):
            expiry_text = signal.expiry.date().isoformat() if hasattr(signal.expiry, "date") else str(signal.expiry)
            lines.append(
                f"{idx}. Strategy: {strategy} | Symbol: {signal.symbol} | Direction: {signal.direction} | "
                f"Option: {signal.option_type} {signal.strike:.2f} exp {expiry_text} | "
                f"Rationale: {signal.rationale}"
            )
        lines.append("")
        lines.append("Return JSON only.")
        return "\n".join(lines)

    def _parse_response(self, response: str) -> List[GeminiSelection]:
        if not response:
            return []
        candidates = [response]
        if "```" in response:
            candidates.extend(part for part in response.split("```") if part and "{" in part)

        for candidate in candidates:
            try:
                cleaned = candidate.strip()
                if cleaned.startswith("json"):
                    cleaned = cleaned[len("json") :].strip()
                start = cleaned.find("{")
                end = cleaned.rfind("}")
                if start != -1 and end != -1 and end > start:
                    cleaned = cleaned[start : end + 1]
                data = json.loads(cleaned)
                selections = self._extract_selections(data)
                if selections:
                    return selections
            except json.JSONDecodeError:
                continue
        return []

    def _extract_selections(self, data: object) -> List[GeminiSelection]:
        if not isinstance(data, dict):
            return []
        finalists = data.get("finalists")
        if not isinstance(finalists, list):
            return []
        selections: List[GeminiSelection] = []
        for item in finalists:
            if not isinstance(item, dict):
                continue
            idx_raw = item.get("id")
            try:
                idx = int(idx_raw)
            except (TypeError, ValueError):
                continue
            score_raw = item.get("score")
            try:
                score = float(score_raw) if score_raw is not None else None
            except (TypeError, ValueError):
                score = None
            reason = item.get("reason")
            if isinstance(reason, str):
                reason = reason.strip()
            selections.append(GeminiSelection(id=idx, score=score, reason=reason or None))
        return selections


@dataclass(slots=True)
class SignalValidationAgent:
    """Review a signal against lightweight market context to offer guidance."""

    lookback_limit: int = 20
    client: Optional[GeminiClient] = None
    enable_gemini: bool = True
    cooldown_seconds: int = 60
    _rate_limited_until: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        if self.enable_gemini and self.client is None:
            self.client = GeminiClient()

    def review(
        self,
        signal: TradeSignal,
        snapshot: Optional[OptionChainSnapshot],
        peer_signals: Iterable[TradeSignal],
    ) -> str:
        peer_list = list(peer_signals)
        trend = self._infer_trend(snapshot)
        alignment = self._assess_alignment(signal.direction, trend)
        peer_view = self._peer_context(signal, peer_list)

        system_prompt = (
            "You are an options risk manager. Evaluate the robustness of a trading signal "
            "using trend, peer signals, and option data, then provide validation guidance."
        )
        peer_summary = self._peer_summary(peer_list, signal)
        user_prompt = (
            "Assess whether the signal aligns with current market context. Provide 3 concise bullet points: "
            "market trend assessment, alignment with the signal, and risk/positioning advice.\n"
            f"Signal:\n"
            f"  Symbol: {signal.symbol}\n"
            f"  Direction: {signal.direction}\n"
            f"  Option type: {signal.option_type}\n"
            f"  Strike: {signal.strike:.2f}\n"
            f"  Expiry: {signal.expiry.isoformat()}\n"
            f"  Rationale: {signal.rationale or 'N/A'}\n"
            f"Derived context:\n"
            f"  Trend inference: {trend or 'Unavailable'}\n"
            f"  Alignment note: {alignment or 'None'}\n"
            f"  Peer view: {peer_view or 'No peer context'}\n"
            f"Market snapshot summary:\n{self._snapshot_overview(snapshot) or 'No snapshot data supplied.'}\n"
            f"Peer signal summary:\n{peer_summary or 'No other signals for this batch.'}"
        )
        if not self.enable_gemini:
            logger.info(
                "Gemini validation disabled via configuration; using heuristic | symbol={symbol}",
                symbol=signal.symbol,
            )
            return self._fallback_review(signal, trend, alignment, peer_view)
        if time.time() < self._rate_limited_until:
            logger.info(
                "Gemini validation temporarily disabled due to prior rate limit; using heuristic | symbol={symbol}",
                symbol=signal.symbol,
            )
            return self._fallback_review(signal, trend, alignment, peer_view)
        try:
            review_text = self.client.generate(system_prompt=system_prompt, user_prompt=user_prompt)
        except GeminiClientError as exc:
            self._handle_rate_limit(exc)
            logger.warning(
                "Falling back to heuristic validation | symbol={symbol} reason={error}",
                symbol=signal.symbol,
                error=exc,
            )
            return self._fallback_review(signal, trend, alignment, peer_view)
        logger.debug(
            "Validation agent (Gemini) output | symbol={symbol} text={text}",
            symbol=signal.symbol,
            text=review_text,
        )
        return review_text

    def _handle_rate_limit(self, exc: GeminiClientError) -> None:
        message = str(exc).lower()
        if "429" in message or "resource exhausted" in message or "rate" in message:
            self._rate_limited_until = time.time() + max(self.cooldown_seconds, 1)

    def _fallback_review(
        self,
        signal: TradeSignal,
        trend: Optional[str],
        alignment: Optional[str],
        peer_view: Optional[str],
    ) -> str:
        commentary_parts: List[str] = []
        if trend:
            commentary_parts.append(f"Observed option order flow appears {trend} for {signal.symbol}.")
        if alignment:
            commentary_parts.append(alignment)
        if peer_view:
            commentary_parts.append(peer_view)
        if not commentary_parts:
            commentary_parts.append(
                "Limited market context available; consider validating the idea with broader trend analysis."
            )
        return " ".join(commentary_parts)

    def _snapshot_overview(self, snapshot: Optional[OptionChainSnapshot]) -> str:
        if snapshot is None or not snapshot.options:
            return ""
        option_count = len(snapshot.options)
        avg_mark = sum(float(opt.get("mark", 0.0) or 0.0) for opt in snapshot.options) / max(option_count, 1)
        return (
            f"Underlying price: {snapshot.underlying_price:.2f} (UTC {snapshot.timestamp.isoformat()})\n"
            f"Options captured: {option_count} | average mark {avg_mark:.2f}"
        )

    def _peer_summary(self, peer_signals: List[TradeSignal], target: TradeSignal) -> str:
        if not peer_signals:
            return ""
        total = len(peer_signals)
        same_symbol = [s for s in peer_signals if s.symbol == target.symbol]
        if not same_symbol:
            return f"Total signals evaluated: {total}; none share the symbol {target.symbol}."
        directional_breakdown: dict[str, int] = {}
        for sig in same_symbol:
            directional_breakdown.setdefault(sig.direction, 0)
            directional_breakdown[sig.direction] += 1
        breakdown_str = ", ".join(f"{direction}: {count}" for direction, count in directional_breakdown.items())
        return (
            f"Total signals evaluated: {total}; matching symbol signals: {len(same_symbol)}.\n"
            f"Directional breakdown: {breakdown_str}"
        )

    def _infer_trend(self, snapshot: Optional[OptionChainSnapshot]) -> Optional[str]:
        if snapshot is None or not snapshot.options:
            return None

        call_marks: List[float] = []
        put_marks: List[float] = []
        for option in snapshot.options[: self.lookback_limit]:
            mark = float(option.get("mark", 0.0) or 0.0)
            if mark <= 0.0:
                continue
            if option.get("option_type") == "CALL":
                call_marks.append(mark)
            elif option.get("option_type") == "PUT":
                put_marks.append(mark)

        if not call_marks and not put_marks:
            return None

        avg_call = mean(call_marks) if call_marks else 0.0
        avg_put = mean(put_marks) if put_marks else 0.0

        if avg_call > avg_put * 1.05:
            return "bullish"
        if avg_put > avg_call * 1.05:
            return "bearish"
        return "balanced"

    def _assess_alignment(self, direction: str, trend: Optional[str]) -> Optional[str]:
        if trend is None:
            return None
        direction_upper = (direction or "").upper()
        if trend == "bullish" and any(term in direction_upper for term in ("CALL", "LONG", "BULL")):
            return "The idea aligns with the bullish skew in option pricing."
        if trend == "bearish" and any(term in direction_upper for term in ("PUT", "SHORT", "BEAR")):
            return "The idea aligns with the bearish skew in option pricing."
        if trend == "balanced":
            return "Option pricing looks balanced; position sizing discipline is important."
        return "The signal runs counter to the detected skew, so ensure risk controls are strict."

def _peer_context(
    self,
    signal: TradeSignal,
    peer_signals: Iterable[TradeSignal],
) -> Optional[str]:
    similar = [s for s in peer_signals if s.symbol == signal.symbol and s is not signal]
    if not similar:
        return None
    same_direction = [s for s in similar if s.direction == signal.direction]
    if len(same_direction) == len(similar):
        return "Multiple strategies share this direction, adding conviction."
    if not same_direction:
        return "Other strategies disagree on direction; double-check assumptions."
    return "Some strategies agree while others differ; weigh conviction before acting."


__all__ = [
    "SignalValidationAgent",
    "SignalBatchSelector",
    "GeminiSelection",
    "BatchSelectionResult",
    "GeminiClient",
    "GeminiClientError",
]
