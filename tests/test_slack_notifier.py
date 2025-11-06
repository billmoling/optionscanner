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

    def test_payload_includes_summary_information(self):
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

        self.assertEqual(len(self.sent), 1)
        url, payload = self.sent[0]
        self.assertEqual(url, "https://hooks.test")
        self.assertEqual(payload.get("username"), "Scanner")
        self.assertEqual(payload.get("channel"), "#alerts")
        text = payload["text"]
        self.assertIn("Daily Signals", text)
        self.assertIn("Total signals: 3", text)
        self.assertIn("NVDA", text)
        self.assertIn("AAPL", text)
        self.assertIn("…and 1 more signal.", text)
        self.assertIn("CSV saved to results/signals.csv", text)


if __name__ == "__main__":
    unittest.main()
