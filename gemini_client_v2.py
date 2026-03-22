"""Google Gemini client using the new google-genai SDK (Gemini 2.5)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import yaml
from loguru import logger

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None  # type: ignore
    types = None  # type: ignore


class GeminiClientError(RuntimeError):
    """Raised when the Gemini client cannot fulfill a request."""


@dataclass(slots=True)
class GeminiClientV2:
    """Google Gemini client using the new google-genai SDK.

    This is the updated client that uses gemini-2.5-flash model.

    Configuration:
        model_name: Model to use (default: gemini-2.5-flash)
        temperature: Sampling temperature (default: 0.6)
        top_p: Nucleus sampling top_p (default: 0.9)
        max_output_tokens: Maximum output tokens (optional)
    """

    model_name: str = "gemini-2.5-flash"
    model_name_env_var: str = "GEMINI_MODEL_NAME"
    api_key_env_vars: Sequence[str] = ("GOOGLE_API_KEY", "GEMINI_API_KEY")
    config_path_env_var: str = "GEMINI_CONFIG_PATH"
    config_file_candidates: Sequence[str] = (
        "config/secrets.yaml",
        "secrets.yaml",
        ".secrets.yaml",
    )
    temperature: float = 0.6
    top_p: float = 0.9
    max_output_tokens: Optional[int] = None
    _client: Optional["genai.Client"] = field(init=False, default=None)
    _configured: bool = field(init=False, default=False)
    _last_system_prompt: Optional[str] = field(init=False, default=None)

    def __post_init__(self) -> None:
        resolved_model = self._resolve_model_name()
        if resolved_model:
            self.model_name = resolved_model

    def _ensure_client(self, system_prompt: str) -> "genai.Client":
        """Ensure the client is configured."""
        if self._configured and self._client is not None:
            return self._client

        if genai is None:
            raise GeminiClientError(
                "google-genai is not installed; install with: pip install google-genai"
            )

        api_key = self._resolve_api_key()
        if not api_key:
            raise GeminiClientError(
                "Gemini API key not found. Set one of: " + ", ".join(self.api_key_env_vars)
            )

        try:
            self._client = genai.Client(api_key=api_key)
            self._configured = True
            self._last_system_prompt = system_prompt
            return self._client
        except Exception as exc:
            raise GeminiClientError(f"Failed to initialise Gemini client: {exc}") from exc

    def _resolve_api_key(self) -> Optional[str]:
        """Resolve API key from environment or config files."""
        # Try environment variables first
        for env_var in self.api_key_env_vars:
            api_key = os.getenv(env_var)
            if api_key:
                return api_key

        # Try config files
        config_api_key = self._resolve_api_key_from_config()
        if config_api_key:
            return config_api_key

        return None

    def _resolve_model_name(self) -> Optional[str]:
        """Resolve model name from environment or config."""
        env_model = os.getenv(self.model_name_env_var, "").strip()
        if env_model:
            return env_model

        # Try config files
        candidate_paths = [Path(path) for path in ("config.yaml",)]
        for path in candidate_paths:
            if not path.exists():
                continue
            try:
                with path.open("r", encoding="utf-8") as fh:
                    data = yaml.safe_load(fh) or {}
            except Exception:
                continue

            # Check gemini section
            gemini = data.get("gemini")
            if isinstance(gemini, dict):
                model = gemini.get("model_name") or gemini.get("model")
                if isinstance(model, str) and model.strip():
                    return model.strip()

        return None

    def _resolve_api_key_from_config(self) -> Optional[str]:
        """Try to load API key from config files."""
        for path_str in self.config_file_candidates:
            path = Path(path_str)
            if not path.exists():
                continue
            try:
                with path.open("r", encoding="utf-8") as fh:
                    data = yaml.safe_load(fh) or {}
            except Exception as exc:
                logger.debug(
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
        """Extract API key from config data."""
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
        """Generate text using Gemini.

        Args:
            system_prompt: System instruction for the model
            user_prompt: User's prompt

        Returns:
            Generated text response
        """
        if genai is None:
            raise GeminiClientError("google-genai is not installed")

        client = self._ensure_client(system_prompt)

        # Build generation config
        config_dict = {
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        if self.max_output_tokens is not None:
            config_dict["max_output_tokens"] = self.max_output_tokens

        try:
            # Combine system prompt and user prompt
            full_prompt = f"{system_prompt}\n\n{user_prompt}"

            response = client.models.generate_content(
                model=self.model_name,
                contents=full_prompt,
                config=types.GenerateContentConfig(**config_dict) if types else None,
            )

            return self._extract_text_from_response(response)

        except Exception as exc:
            raise GeminiClientError(f"Gemini generation failed: {exc}") from exc

    def _extract_text_from_response(self, response: object) -> str:
        """Extract text from Gemini response."""
        # Try direct text attribute
        text = getattr(response, "text", None)
        if isinstance(text, str) and text.strip():
            return text.strip()

        # Try candidates
        candidates = getattr(response, "candidates", None) or []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            if content is None:
                continue
            parts = getattr(content, "parts", None) or []
            for part in parts:
                part_text = getattr(part, "text", None)
                if isinstance(part_text, str) and part_text.strip():
                    return part_text.strip()

        raise GeminiClientError("Gemini response did not contain text content")


__all__ = ["GeminiClientV2", "GeminiClientError"]
