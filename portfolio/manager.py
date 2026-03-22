"""Portfolio manager orchestrating position aggregation, risk checks and reporting."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml
from ib_async import IB
from loguru import logger

from .greeks import GreekCalculator, GreekSummary, compute_concentration
from .playbooks import PlaybookContext, PlaybookEngine
from .positions import PositionLoader, PositionSource
from .report import PortfolioReporter, ReporterConfig
from .rules import RiskBreach, RiskEvaluator, RiskLimitConfig


class PortfolioManager:
    """High level coordinator for portfolio risk management."""

    def __init__(
        self,
        ib: IB,
        config_path: str = "risk.yaml",
        *,
        slack_config: Optional[dict] = None,
        enable_gemini: bool = True,
    ) -> None:
        self._ib = ib
        self._config_path = Path(config_path)
        self._config_data = self._load_config()
        self._risk_config = RiskLimitConfig(
            limits=self._config_data.get("limits", {}),
            roll_rules=self._config_data.get("roll_rules", {}),
        )
        logs_dir = Path(self._config_data.get("logs_dir", "./logs"))
        results_dir = Path(self._config_data.get("results_dir", "./results"))
        self._position_loader = PositionLoader(
            PositionSource(ib=self._ib, log_dir=logs_dir, results_dir=results_dir)
        )
        reporter_config = ReporterConfig(
            results_dir=results_dir,
            logs_dir=logs_dir,
            slack_config=slack_config,
            enable_gemini=enable_gemini,
        )
        self._reporter = PortfolioReporter(reporter_config)
        self.positions: pd.DataFrame = pd.DataFrame()
        self.greek_summary: GreekSummary = GreekSummary(
            per_symbol=pd.DataFrame(), totals={}
        )
        self.concentration: pd.DataFrame = pd.DataFrame()
        self.breaches: List[RiskBreach] = []
        self.actions: List[str] = []
        self.last_gemini_response: Optional[str] = None

    def _load_config(self) -> Dict[str, dict]:
        if not self._config_path.exists():
            logger.warning(
                "Risk configuration file not found at {path}; using defaults",
                path=str(self._config_path),
            )
            return {"limits": {}, "roll_rules": {}}
        with self._config_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        logger.debug("Loaded risk configuration from {path}", path=str(self._config_path))
        return data

    def load_positions(self) -> pd.DataFrame:
        logger.info(
            "Loading portfolio positions",
            component="portfolio_manager",
            event_type="load_positions_start",
        )
        self.positions = self._position_loader.load()
        logger.info(
            "Loaded {count} positions",
            count=len(self.positions),
            component="portfolio_manager",
            event_type="load_positions_complete",
        )
        return self.positions

    def compute_greeks(self) -> GreekSummary:
        logger.info(
            "Computing portfolio greeks",
            component="portfolio_manager",
            event_type="compute_greeks_start",
        )
        calculator = GreekCalculator(self._ib)
        self.greek_summary = calculator.compute(self.positions)
        concentration_df, _ = compute_concentration(self.positions)
        self.concentration = concentration_df
        logger.info(
            "Portfolio greeks computed | delta={delta} gamma={gamma} theta={theta} vega={vega}",
            delta=self.greek_summary.totals.get("delta"),
            gamma=self.greek_summary.totals.get("gamma"),
            theta=self.greek_summary.totals.get("theta"),
            vega=self.greek_summary.totals.get("vega"),
            component="portfolio_manager",
            event_type="compute_greeks_complete",
        )
        return self.greek_summary

    def evaluate_rules(self) -> List[RiskBreach]:
        logger.info(
            "Evaluating risk limits",
            component="portfolio_manager",
            event_type="evaluate_rules_start",
        )
        evaluator = RiskEvaluator(self._risk_config)
        self.breaches = evaluator.evaluate(
            self.greek_summary.per_symbol,
            self.greek_summary.totals,
            self.concentration,
            self.positions,
        )
        if self.breaches:
            logger.warning(
                "Risk limits breached | count={count}",
                count=len(self.breaches),
                component="portfolio_manager",
                event_type="risk_breach_detected",
            )
        else:
            logger.info(
                "No risk limits breached",
                component="portfolio_manager",
                event_type="evaluate_rules_complete",
            )
        return self.breaches

    def generate_actions(self) -> List[str]:
        logger.info(
            "Generating playbook actions | breaches={count}",
            count=len(self.breaches),
            component="portfolio_manager",
            event_type="generate_actions_start",
        )
        context = PlaybookContext(roll_rules=self._risk_config.roll_rules)
        engine = PlaybookEngine(context)
        self.actions = engine.generate(self.positions, self.breaches)
        logger.info(
            "Generated {count} playbook actions",
            count=len(self.actions),
            component="portfolio_manager",
            event_type="generate_actions_complete",
        )
        return self.actions

    def notify(self) -> str:
        logger.info(
            "Preparing portfolio notifications",
            component="portfolio_manager",
            event_type="notify_start",
        )
        message = self._reporter.build_summary_message(
            self.greek_summary.totals,
            self.concentration,
            self.breaches,
            self.actions,
        )
        csv_path = None
        json_path = None
        try:
            csv_path, _json_unused, timestamp = self._reporter.write_outputs(
                self.positions, self.greek_summary.per_symbol
            )
            logger.info(
                "Portfolio outputs written | csv={path}",
                path=csv_path,
                component="portfolio_manager",
                event_type="outputs_written",
            )
            self.last_gemini_response = self._reporter.evaluate_positions_with_gemini(
                self.positions, timestamp
            )
            if self.last_gemini_response:
                logger.info(
                    "Gemini portfolio evaluation completed",
                    component="portfolio_manager",
                    event_type="gemini_evaluation_complete",
                )
        except Exception as exc:
            logger.warning("Failed to write portfolio CSV | reason={error}", error=exc)
            csv_path = None
        self._reporter.log_details(message)
        self._reporter.send_notifications(message, csv_path)
        logger.info(
            "Portfolio notification sent",
            component="portfolio_manager",
            event_type="notify_complete",
        )
        return message


__all__ = ["PortfolioManager"]
