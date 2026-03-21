import logging
import os
import sys

import pytest
from dotenv import load_dotenv

pytest.importorskip("google.genai", reason="google generative AI client not installed")
from google import genai

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

REQUEST_MODEL = "gemini-2.5-flash"
PROMPT = "What is the full name of America?"


def _resolve_api_key() -> tuple[str | None, str]:
    for env_var in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        value = os.getenv(env_var)
        if value:
            return value, env_var
    return None, "unset"


api_key, api_key_source = _resolve_api_key()

if not api_key:
    raise RuntimeError("Set GOOGLE_API_KEY or GEMINI_API_KEY in your environment before running this script.")

log.info(f"[Gemini] Loaded .env; using API key from {api_key_source} (length={len(api_key)})")
log.info(f"[Gemini] Model: {REQUEST_MODEL}")
log.info(f"[Gemini] Prompt: {PROMPT}")

client = genai.Client(api_key=api_key)
try:
    response = client.models.generate_content(model=REQUEST_MODEL, contents=PROMPT)
except Exception:  # pragma: no cover - integration runtime
    log.exception("[Gemini] Request failed")
    raise

log.info(f"[Gemini] Raw response type: {type(response).__name__}")
finish_reason = getattr(response, "finish_reason", None)
if finish_reason:
    log.info(f"[Gemini] Finish reason: {finish_reason}")
log.info(f"[Gemini] Response text length: {len(getattr(response, 'text', '') or '')}")
log.info("[Gemini] Response text:")
log.info(response.text)
