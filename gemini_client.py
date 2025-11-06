"""Thin wrapper around google-generativeai for Gemini text generation."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from textwrap import dedent
from typing import List, Optional, Sequence

from loguru import logger
import yaml

try:  # pragma: no cover - optional dependency resolved at runtime
    import google.generativeai as genai
except ImportError:  # pragma: no cover - handled gracefully
    genai = None  # type: ignore[assignment]


class GeminiClientError(RuntimeError):
    """Raised when the Gemini client cannot fulfill a request."""


@dataclass(slots=True)
class GeminiClient:
    """Lightweight wrapper for Google Gemini text generation."""
    'TODO: update model to use 2.5, the current API is my own account, which only have 2.0 access'
    model_name: str = "gemini-2.0-flash"
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

        text: Optional[str]
        try:
            text = getattr(response, "text", None)
        except ValueError:
            text = None
        if text:
            return text.strip()
        try:
            candidates = getattr(response, "candidates", None) or []
            parts = candidates[0].content.parts if candidates else []
            combined = " ".join(
                part.text for part in parts if getattr(part, "text", "")
            )
        except Exception as exc:  # pragma: no cover - defensive parsing
            raise GeminiClientError(f"Unable to parse Gemini response: {exc}") from exc
        if not combined.strip():
            raise GeminiClientError("Gemini response did not contain text content")
        return combined.strip()


__all__ = ["GeminiClient", "GeminiClientError"]
