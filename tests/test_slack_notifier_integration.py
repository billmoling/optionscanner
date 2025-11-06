import os
import unittest
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

from notifications.slack import SlackNotifier

load_dotenv()


def _is_integration_enabled() -> bool:
    enabled = bool(os.getenv("SLACK_WEBHOOK_URL"))
    if not enabled:
        print("Skipping Slack integration test: SLACK_WEBHOOK_URL not set.")
    return enabled


@unittest.skipUnless(_is_integration_enabled(), "SLACK_WEBHOOK_URL not set; skipping Slack integration test.")
class SlackNotifierIntegrationTests(unittest.TestCase):
    def test_slack_notifier_posts_message(self) -> None:
        notifier = SlackNotifier(
            {
                "enabled": True,
                "title": "Option Scanner Integration Test",
                "username": "OptionScannerBot",
                "channel": os.getenv("SLACK_TEST_CHANNEL"),
                "icon_emoji": ":robot_face:",
            }
        )
        df = pd.DataFrame(
            [
                {
                    "symbol": "NVDA",
                    "strategy": "IntegrationTest",
                    "action": "PING",
                    "option_type": "CALL",
                    "strike": 0,
                    "expiry": datetime.utcnow().date(),
                    "confidence": 1.0,
                    "explanation": "Integration test message.",
                }
            ]
        )

        notifier.send_signals(df)

        # If send_signals doesn't raise, we consider the post successful.


if __name__ == "__main__":
    unittest.main()
