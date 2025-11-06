import os
import unittest
from pathlib import Path

import pandas as pd

from notifications.slack import SlackNotifier


class SlackNotifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.sent = []

        def fake_post(url, payload):
            self.sent.append((url, payload))

        self.fake_post = fake_post

    def test_disabled_notifier_does_not_post(self):
        notifier = SlackNotifier({"enabled": False}, post=self.fake_post)
        df = pd.DataFrame([{"symbol": "NVDA"}])

        notifier.send_signals(df)

        self.assertEqual(self.sent, [])

    def test_empty_dataframe_skips_post(self):
        notifier = SlackNotifier({"enabled": True, "webhook_url": "https://hooks.test"}, post=self.fake_post)
        df = pd.DataFrame()

        notifier.send_signals(df)

        self.assertEqual(self.sent, [])

    def test_each_signal_sends_individual_message(self):
        notifier = SlackNotifier(
            {
                "enabled": True,
                "webhook_url": "https://hooks.test",
                "title": "Daily Signals",
                "username": "Scanner",
                "channel": "#alerts",
                "max_rows": 2,
            },
            post=self.fake_post,
        )
        df = pd.DataFrame(
            [
                {"symbol": "NVDA", "strategy": "Momentum", "action": "BUY", "confidence": 0.9},
                {"symbol": "AAPL", "strategy": "Reversal", "action": "SELL", "confidence": 0.7},
                {"symbol": "TSLA", "strategy": "Breakout", "action": "BUY", "confidence": 0.8},
            ]
        )
        csv_path = Path("results/signals.csv")

        notifier.send_signals(df, csv_path)

        self.assertEqual(len(self.sent), 3)
        for (url, payload), symbol in zip(self.sent, ["NVDA", "AAPL", "TSLA"]):
            self.assertEqual(url, "https://hooks.test")
            self.assertEqual(payload.get("username"), "Scanner")
            self.assertEqual(payload.get("channel"), "#alerts")
            text = payload["text"]
            self.assertIn("Daily Signals", text)
            self.assertIn(symbol, text)
            self.assertIn("CSV saved to results/signals.csv", text)

    def test_environment_variable_used_when_config_missing(self):
        os.environ["SLACK_WEBHOOK_URL"] = "https://hooks.env"
        self.addCleanup(lambda: os.environ.pop("SLACK_WEBHOOK_URL", None))

        notifier = SlackNotifier({"enabled": True}, post=self.fake_post)
        df = pd.DataFrame([{"symbol": "NVDA"}])

        notifier.send_signals(df)

        self.assertEqual(len(self.sent), 1)
        url, _payload = self.sent[0]
        self.assertEqual(url, "https://hooks.env")

    def test_secrets_file_used_when_config_and_env_missing(self):
        # Ensure env var is absent
        os.environ.pop("SLACK_WEBHOOK_URL", None)

        temp_dir = Path("config")
        temp_dir.mkdir(exist_ok=True)
        secret_path = temp_dir / "secrets.yaml"
        secret_path.write_text("slack:\n  webhook_url: https://hooks.secret\n", encoding="utf-8")
        self.addCleanup(lambda: secret_path.unlink(missing_ok=True))

        notifier = SlackNotifier({"enabled": True}, post=self.fake_post)
        df = pd.DataFrame([{"symbol": "NVDA"}])

        notifier.send_signals(df)

        self.assertEqual(len(self.sent), 1)
        url, _payload = self.sent[0]
        self.assertEqual(url, "https://hooks.secret")


if __name__ == "__main__":
    unittest.main()
