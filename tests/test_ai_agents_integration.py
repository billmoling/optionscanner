import logging
import os
import sys
import unittest
from datetime import datetime, timezone

from dotenv import load_dotenv

from gemini_client import GeminiClient, GeminiClientError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)
load_dotenv()


def _is_gemini_enabled() -> bool:
    try:
        import google.generativeai  # noqa: F401
    except ImportError:
        log.warning("Skipping Gemini integration test: google-generativeai not installed.")
        return False

    for env_var in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        if os.getenv(env_var):
            return True

    log.warning("Skipping Gemini integration test: GOOGLE_API_KEY/GEMINI_API_KEY not set.")
    return False


@unittest.skipUnless(_is_gemini_enabled(), "Gemini credentials not configured; skipping integration test.")
class GeminiClientIntegrationTest(unittest.TestCase):
    def test_explain_agent_generates_output(self) -> None:
        system_prompt = "You are an expert options strategist who explains trade signals."
        user_prompt = (
            "Explain the rationale for a long NVDA call option expiring in 30 days with a 500 strike. "
            f"Current UTC time: {datetime.now(timezone.utc).isoformat()}."
        )

        try:
            explanation = GeminiClient().generate(system_prompt=system_prompt, user_prompt=user_prompt)
        except (ValueError, GeminiClientError) as exc:
            message = str(exc)
            if "finish_reason" in message:
                log.warning(f"Skipping Gemini integration test: {message}")
                self.skipTest(f"Gemini returned unfinished response: {message}")
            raise

        log.info("Gemini explanation output:\n%s", explanation)
        self.assertTrue(explanation.strip())


if __name__ == "__main__":
    unittest.main()
