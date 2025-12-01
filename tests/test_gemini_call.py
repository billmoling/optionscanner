import os

import pytest
from dotenv import load_dotenv

pytest.importorskip("google.genai", reason="google generative AI client not installed")
from google import genai

load_dotenv()

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

print(f"[Gemini] Loaded .env; using API key from {api_key_source} (length={len(api_key)})")
print(f"[Gemini] Model: {REQUEST_MODEL}")
print(f"[Gemini] Prompt: {PROMPT}")

client = genai.Client(api_key=api_key)
try:
    response = client.models.generate_content(model=REQUEST_MODEL, contents=PROMPT)
except Exception as exc:  # pragma: no cover - integration runtime
    print(f"[Gemini] Request failed: {exc}")
    raise

print(f"[Gemini] Raw response type: {type(response).__name__}")
finish_reason = getattr(response, "finish_reason", None)
if finish_reason:
    print(f"[Gemini] Finish reason: {finish_reason}")
print(f"[Gemini] Response text length: {len(getattr(response, 'text', '') or '')}")
print("[Gemini] Response text:")
print(response.text)
