"""Utility agents for explaining and validating trade signals."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from textwrap import dedent
from typing import Iterable, List, Optional, Sequence, TYPE_CHECKING

from loguru import logger
import yaml

from strategies.base import TradeSignal

try:  # pragma: no cover - optional dependency resolved at runtime
    import google.generativeai as genai
except ImportError:  # pragma: no cover - handled gracefully in GeminiClient
    genai = None  # type: ignore[assignment]

if TYPE_CHECKING:  # pragma: no cover - imported for type checking only
    from main import OptionChainSnapshot


class GeminiClientError(RuntimeError):
    """Raised when the Gemini client cannot fulfill a request."""


@dataclass(slots=True)
class GeminiClient:
    """Lightweight wrapper for Google Gemini text generation."""

    model_name: str = "gemini-1.5-flash"
    api_key_env_vars: Sequence[str] = ("GOOGLE_API_KEY", "GEMINI_API_KEY")
    config_path_env_var: str = "GEMINI_CONFIG_PATH"
    config_file_candidates: Sequence[str] = (
        "config/secrets.yaml",
        "secrets.yaml",
        ".secrets.yaml",
    )
    temperature: float = 0.6
    top_p: float = 0.9
    max_output_tokens: int = 512
    _model: Optional["genai.GenerativeModel"] = field(init=False, default=None)
    _configured: bool = field(init=False, default=False)

    def _ensure_model(self) -> "genai.GenerativeModel":
        if self._configured and self._model is not None:
            return self._model
        if genai is None:
            raise GeminiClientError(
                "google-generativeai is not installed; install it or disable Gemini usage."
            )
        api_key = self._resolve_api_key()
        if not api_key:
            raise GeminiClientError(
                "Gemini API key not found. Set one of: " + ", ".join(self.api_key_env_vars)
            )
        try:
            genai.configure(api_key=api_key)
            generation_config = genai.GenerationConfig(
                temperature=self.temperature,
                top_p=self.top_p,
                max_output_tokens=self.max_output_tokens,
            )
            self._model = genai.GenerativeModel(
                self.model_name,
                generation_config=generation_config,
            )
            self._configured = True
            return self._model
        except Exception as exc:  # pragma: no cover - network/runtime failure
            raise GeminiClientError(f"Failed to initialise Gemini client: {exc}") from exc

    def _resolve_api_key(self) -> Optional[str]:
        for env_var in self.api_key_env_vars:
            api_key = os.getenv(env_var)
            if api_key:
                return api_key
        config_api_key = self._resolve_api_key_from_config()
        if config_api_key:
            return config_api_key
        return None

    def _resolve_api_key_from_config(self) -> Optional[str]:
        explicit_path = os.getenv(self.config_path_env_var)
        candidate_paths: List[Path] = []
        if explicit_path:
            candidate_paths.append(Path(explicit_path).expanduser())
        candidate_paths.extend(Path(path) for path in self.config_file_candidates)

        for path in candidate_paths:
            if not path.exists():
                continue
            try:
                with path.open("r", encoding="utf-8") as fh:
                    data = yaml.safe_load(fh) or {}
            except Exception as exc:  # pragma: no cover - defensive parsing
                logger.warning(
                    "Unable to read Gemini secret file | path={path} reason={error}",
                    path=str(path),
                    error=exc,
                )
                continue
            api_key = self._extract_api_key(data)
            if api_key:
                return api_key
        return None

    def _extract_api_key(self, data: object) -> Optional[str]:
        if not isinstance(data, dict):
            return None

        key_paths = (
            ("google_api_key",),
            ("gemini_api_key",),
            ("google", "api_key"),
            ("gemini", "api_key"),
        )

        for path in key_paths:
            value: object = data
            for part in path:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    break
            else:
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return None

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        model = self._ensure_model()
        prompt = dedent(
            f"""
            {system_prompt.strip()}

            {user_prompt.strip()}
            """
        ).strip()
        try:
            response = model.generate_content([{"role": "user", "parts": [{"text": prompt}]}])
        except Exception as exc:  # pragma: no cover - API/runtime failure
            raise GeminiClientError(f"Gemini generation failed: {exc}") from exc

        text = getattr(response, "text", None)
        if text:
            return text.strip()
        try:
            candidates = getattr(response, "candidates", None) or []
            parts = candidates[0].content.parts if candidates else []
            combined = " ".join(part.text for part in parts if getattr(part, "text", ""))
        except Exception as exc:  # pragma: no cover - defensive parsing
            raise GeminiClientError(f"Unable to parse Gemini response: {exc}") from exc
        if not combined.strip():
            raise GeminiClientError("Gemini response did not contain text content")
        return combined.strip()


@dataclass(slots=True)
class SignalExplainAgent:
    """Generate a natural language explanation for a trade signal."""

    client: Optional[GeminiClient] = None
    bullish_templates: Sequence[str] = (
        "The setup anticipates strength in {symbol}, so the {direction} {option_type} looks to capture upside once the price clears the {strike:.2f} strike.",
        "A supportive backdrop suggests momentum could build above {strike:.2f}, making a {direction.lower()} on the {option_type.lower()} attractive for upside participation.",
    )
    bearish_templates: Sequence[str] = (
        "The thesis expects weakness in {symbol}; the {direction} {option_type} aims to profit if price slips beneath {strike:.2f}.",
        "Downside pressure is the primary risk in focus, so {direction.lower()} the {option_type.lower()} offers protection should price drop through {strike:.2f}.",
    )

    def __post_init__(self) -> None:
        if self.client is None:
            self.client = GeminiClient()

    def explain(
        self,
        signal: TradeSignal,
        snapshot: Optional["OptionChainSnapshot"],
    ) -> str:
        """Return a concise explanation of what the signal attempts to capture."""

        snapshot_summary = self._snapshot_summary(snapshot)
        system_prompt = (
            "You are an expert options strategist. Explain to a trader why an options signal "
            "was generated, covering upside, downside, and how market context supports the idea."
        )
        user_prompt = (
            "Provide a focused explanation (3-5 sentences) of the following trade signal. "
            "Highlight what happens if price rises or falls, and reference relevant option market context.\n"
            f"Signal:\n"
            f"  Symbol: {signal.symbol}\n"
            f"  Direction: {signal.direction}\n"
            f"  Option type: {signal.option_type}\n"
            f"  Strike: {signal.strike:.2f}\n"
            f"  Expiry: {signal.expiry.isoformat()}\n"
            f"  Rationale: {signal.rationale or 'N/A'}\n"
            f"Market snapshot summary:\n{snapshot_summary or 'No recent market data provided.'}"
        )
        try:
            explanation = self.client.generate(system_prompt=system_prompt, user_prompt=user_prompt)
            logger.debug(
                "Explain agent (Gemini) output | symbol={symbol} text={text}",
                symbol=signal.symbol,
                text=explanation,
            )
            return explanation
        except GeminiClientError as exc:
            logger.warning(
                "Falling back to template explanation | symbol={symbol} reason={error}",
                symbol=signal.symbol,
                error=exc,
            )
            fallback = self._fallback_explanation(signal, snapshot)
            logger.debug(
                "Explain agent (fallback) output | symbol={symbol} text={text}",
                symbol=signal.symbol,
                text=fallback,
            )
            return fallback

    def _snapshot_summary(self, snapshot: Optional["OptionChainSnapshot"]) -> str:
        if snapshot is None or not snapshot.options:
            return ""
        lines = [
            f"Underlying price: {snapshot.underlying_price:.2f}",
            f"Snapshot timestamp (UTC): {snapshot.timestamp.isoformat()}",
        ]
        sorted_options = sorted(
            snapshot.options,
            key=lambda option: abs(float(option.get("strike", 0.0)) - snapshot.underlying_price),
        )[:5]
        for option in sorted_options:
            strike = float(option.get("strike", 0.0))
            option_type = option.get("option_type", "?")
            mark = float(option.get("mark", 0.0) or 0.0)
            delta = float(option.get("delta", 0.0) or 0.0)
            lines.append(
                f"  {option_type} strike {strike:.2f} | mark {mark:.2f} | delta {delta:.2f}"
            )
        return "\n".join(lines)

    def _fallback_explanation(
        self,
        signal: TradeSignal,
        snapshot: Optional["OptionChainSnapshot"],
    ) -> str:
        template = self._choose_template(signal.direction)
        base = template.format(
            symbol=signal.symbol,
            direction=signal.direction,
            option_type=signal.option_type,
            strike=signal.strike,
        )
        rationale_note = f" Rationale: {signal.rationale}." if signal.rationale else ""
        market_note = self._market_scenarios(signal, snapshot)
        return f"{base}{market_note}{rationale_note}".strip()

    def _choose_template(self, direction: str) -> str:
        direction_upper = (direction or "").upper()
        if any(keyword in direction_upper for keyword in ("CALL", "BULL", "LONG")):
            return self.bullish_templates[0]
        if any(keyword in direction_upper for keyword in ("PUT", "BEAR", "SHORT")):
            return self.bearish_templates[0]
        return "The strategy produced a {direction} idea on {symbol} around the {strike:.2f} level."  # fallback

    def _market_scenarios(
        self,
        signal: TradeSignal,
        snapshot: Optional["OptionChainSnapshot"],
    ) -> str:
        if snapshot is None or not snapshot.options:
            return ""
        price = snapshot.underlying_price
        upside = price * 1.02
        downside = price * 0.98
        return (
            f" If price rallies toward {upside:.2f}, the position should benefit as it moves deeper in-the-money."
            f" If price retreats toward {downside:.2f}, reassess the thesis because the option could lose premium."
        )


@dataclass(slots=True)
class SignalValidationAgent:
    """Review a signal against lightweight market context to offer guidance."""

    lookback_limit: int = 20
    client: Optional[GeminiClient] = None

    def __post_init__(self) -> None:
        if self.client is None:
            self.client = GeminiClient()

    def review(
        self,
        signal: TradeSignal,
        snapshot: Optional["OptionChainSnapshot"],
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
        try:
            review_text = self.client.generate(system_prompt=system_prompt, user_prompt=user_prompt)
            logger.debug(
                "Validation agent (Gemini) output | symbol={symbol} text={text}",
                symbol=signal.symbol,
                text=review_text,
            )
            return review_text
        except GeminiClientError as exc:
            logger.warning(
                "Falling back to heuristic validation | symbol={symbol} reason={error}",
                symbol=signal.symbol,
                error=exc,
            )
            fallback = self._fallback_review(signal, trend, alignment, peer_view)
            logger.debug(
                "Validation agent (fallback) output | symbol={symbol} text={text}",
                symbol=signal.symbol,
                text=fallback,
            )
            return fallback

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

    def _snapshot_overview(self, snapshot: Optional["OptionChainSnapshot"]) -> str:
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
        directional_breakdown = {}
        for signal in same_symbol:
            directional_breakdown.setdefault(signal.direction, 0)
            directional_breakdown[signal.direction] += 1
        breakdown_str = ", ".join(f"{direction}: {count}" for direction, count in directional_breakdown.items())
        return (
            f"Total signals evaluated: {total}; matching symbol signals: {len(same_symbol)}.\n"
            f"Directional breakdown: {breakdown_str}"
        )

    def _infer_trend(self, snapshot: Optional["OptionChainSnapshot"]) -> Optional[str]:
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


__all__ = ["SignalExplainAgent", "SignalValidationAgent", "GeminiClient", "GeminiClientError"]

