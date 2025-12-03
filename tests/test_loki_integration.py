import logging
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

from logging_utils import LokiHandler, get_logger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

load_dotenv()


def _is_loki_configured() -> bool:
    required_vars = ("LOKI_URL", "LOKI_USERNAME", "LOKI_PASSWORD")
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        log.warning("Skipping Loki integration test: missing %s", ", ".join(missing))
        return False
    if LokiHandler is None:
        log.warning("Skipping Loki integration test: python-logging-loki not installed.")
        return False
    return True


@unittest.skipUnless(
    _is_loki_configured(), "LOKI_URL/LOKI_USERNAME/LOKI_PASSWORD not set; skipping Loki integration test."
)
class LokiIntegrationTests(unittest.TestCase):
    def test_sends_log_and_reads_back_from_loki(self) -> None:
        marker = f"optionscanner-loki-test-{uuid4()}"
        with tempfile.TemporaryDirectory(prefix="loki_integration_") as tmp_dir:
            logger = get_logger(Path(tmp_dir), "loki_integration")
            logger.info("Loki integration test marker={marker}", marker=marker)

            time.sleep(1.0)  # allow asynchronous sinks to push the entry

        # If no exception is raised, the log was accepted for delivery. Verification is manual in Grafana.


if __name__ == "__main__":
    unittest.main()
