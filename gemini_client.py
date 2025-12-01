"""Thin wrapper around google-generativeai for Gemini text generation."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
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
    model_name_env_var: str = "GEMINI_MODEL_NAME"
    api_key_env_vars: Sequence[str] = ("GOOGLE_API_KEY", "GEMINI_API_KEY")
    config_path_env_var: str = "GEMINI_CONFIG_PATH"
    model_config_candidates: Sequence[str] = ("config.yaml",)
    config_file_candidates: Sequence[str] = (
        "config/secrets.yaml",
        "secrets.yaml",
        ".secrets.yaml",
    )
    temperature: float = 0.6
    top_p: float = 0.9
    max_output_tokens: Optional[int] = None
    _model: Optional["genai.GenerativeModel"] = field(init=False, default=None)
    _configured: bool = field(init=False, default=False)
    _last_system_prompt: Optional[str] = field(init=False, default=None)

    def __post_init__(self) -> None:
        resolved_model = self._resolve_model_name()
        if resolved_model:
            self.model_name = resolved_model

    def _ensure_model(self, system_prompt: str) -> "genai.GenerativeModel":
        if self._configured and self._model is not None and self._last_system_prompt == system_prompt:
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
            generation_config_data = {
                "temperature": self.temperature,
                "top_p": self.top_p,
            }
            if self.max_output_tokens is not None:
                generation_config_data["max_output_tokens"] = self.max_output_tokens

            generation_config = genai.GenerationConfig(**generation_config_data)
            self._model = genai.GenerativeModel(
                self.model_name,
                generation_config=generation_config,
                system_instruction=system_prompt,
            )
            self._configured = True
            self._last_system_prompt = system_prompt
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

    def _resolve_model_name(self) -> Optional[str]:
        env_model = os.getenv(self.model_name_env_var, "").strip()
        if env_model:
            return env_model

        candidate_paths: List[Path] = []
        settings_path = os.getenv("GEMINI_SETTINGS_PATH")
        if settings_path:
            candidate_paths.append(Path(settings_path).expanduser())
        explicit_path = os.getenv(self.config_path_env_var)
        if explicit_path:
            candidate_paths.append(Path(explicit_path).expanduser())
        candidate_paths.extend(Path(path) for path in self.model_config_candidates)

        for path in candidate_paths:
            if not path.exists():
                continue
            try:
                with path.open("r", encoding="utf-8") as fh:
                    data = yaml.safe_load(fh) or {}
            except Exception as exc:  # pragma: no cover - defensive parsing
                logger.debug(
                    "Unable to read Gemini settings file | path={path} reason={error}",
                    path=str(path),
                    error=exc,
                )
                continue
            model = self._extract_model_name(data)
            if model:
                return model
        return None

    def _extract_model_name(self, data: object) -> Optional[str]:
        if not isinstance(data, dict):
            return None
        direct = data.get("gemini_model_name")
        if isinstance(direct, str) and direct.strip():
            return direct.strip()

        gemini = data.get("gemini")
        if isinstance(gemini, dict):
            for key in ("model_name", "model"):
                value = gemini.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
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
        model = self._ensure_model(system_prompt=system_prompt.strip())
        try:
            response = model.generate_content(user_prompt.strip())
        except Exception as exc:  # pragma: no cover - API/runtime failure
            raise GeminiClientError(f"Gemini generation failed: {exc}") from exc
        try:
            return self._extract_text_from_response(response)
        except GeminiClientError:
            raise
        except Exception as exc:  # pragma: no cover - defensive parsing
            raise GeminiClientError(f"Unable to parse Gemini response: {exc}") from exc

    def _extract_text_from_response(self, response: object) -> str:
        """Best-effort extraction of text content from Gemini responses."""
        text: Optional[str]
        try:
            text = getattr(response, "text", None)
        except ValueError:
            text = None
        if isinstance(text, str) and text.strip():
            return text.strip()

        candidates = getattr(response, "candidates", None) or []
        finish_reasons: List[str] = []
        collected_parts: List[str] = []

        for candidate in candidates:
            finish_reason = getattr(candidate, "finish_reason", None)
            if finish_reason is None and isinstance(candidate, dict):
                finish_reason = candidate.get("finish_reason")
            if finish_reason:
                finish_reasons.append(str(finish_reason))

            content = getattr(candidate, "content", None)
            if content is None and isinstance(candidate, dict):
                content = candidate.get("content")

            parts = getattr(content, "parts", None) if content is not None else None
            if parts is None and isinstance(content, dict):
                parts = content.get("parts")

            for part in parts or []:
                part_text = getattr(part, "text", None)
                if part_text is None and isinstance(part, dict):
                    part_text = part.get("text")
                if isinstance(part_text, str) and part_text.strip():
                    collected_parts.append(part_text.strip())
                elif isinstance(part, str) and part.strip():
                    collected_parts.append(part.strip())

        if collected_parts:
            return " ".join(collected_parts).strip()

        block_reason = getattr(getattr(response, "prompt_feedback", None), "block_reason", None)
        details: List[str] = []
        if finish_reasons:
            details.append(f"finish_reason={','.join(finish_reasons)}")
        if block_reason:
            details.append(f"block_reason={block_reason}")
        detail_suffix = f" ({'; '.join(details)})" if details else ""
        raise GeminiClientError(f"Gemini response did not contain text content{detail_suffix}")


__all__ = ["GeminiClient", "GeminiClientError"]
