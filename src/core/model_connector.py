"""Model connector for sequential task execution with persistent sessions."""

from __future__ import annotations

import json
import re
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import requests


def clean_prompt_text(text: str) -> str:
    """Return a single-line version of ``text`` for embedding in prompts."""

    if not isinstance(text, str):
        return ""
    text = re.sub(r"[\n\r\t]+", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


class ModelConnector:
    """Blocking HTTP client that maintains one persistent chat session per model."""

    JSON_ONLY_INSTRUCTION = "Return only the JSON object described. No extra text."

    REQUIRED_KEYS = {
        "core_event",
        "themes",
        "tone",
        "conflict_type",
        "stakes",
        "setting_hint",
        "characters",
        "potential_story_hooks",
    }

    LIST_KEYS = {"themes", "characters", "potential_story_hooks"}

    def __init__(
        self,
        model: str,
        url: str,
        request_timeout: int,
        compliance_interval: int,
        logger,
    ):
        self.model = model
        self.url = url
        self.request_timeout = request_timeout
        self.logger = logger
        self.compliance_interval = max(0, int(compliance_interval or 0))
        self._session_messages: List[Dict[str, str]] = []
        self._history: List[Dict[str, str]] = []
        self._active = False
        self._headline_counter = 0
        self._auto_compliance_reminders = 0
        self._manual_compliance_reminders = 0
        self._array_warning_count = 0

    # ------------------------------------------------------------------
    def start_session(self, prompt_dynamic: str, prompt_formatting: str) -> None:
        """Initialise the persistent message history for this model."""

        self.logger.info("Starting session for model %s", self.model)
        dynamic = self._normalise_prompt_section(prompt_dynamic)
        formatting = self._normalise_prompt_section(prompt_formatting)
        self._session_messages = [{"role": "system", "content": self.JSON_ONLY_INSTRUCTION}]
        if dynamic:
            self._session_messages.append({"role": "user", "content": dynamic})
        if formatting:
            self._session_messages.append({"role": "user", "content": formatting})
        self._history = []
        self._active = True
        self._headline_counter = 0

    def close_session(self) -> None:
        """Reset internal session state."""
        if self._active:
            self.logger.info("Closing session for model %s", self.model)
        self._session_messages = []
        self._history = []
        self._headline_counter = 0
        self._active = False

    # ------------------------------------------------------------------
    def send_headline(self, headline: str) -> Optional[Dict[str, object]]:
        """Send one headline to the model and return the validated JSON payload."""
        if not self._active:
            raise RuntimeError("Session has not been started. Call start_session() first.")

        self._maybe_send_auto_compliance_reminder()

        headline_text = self._format_headline(headline)
        user_message = {"role": "user", "content": headline_text}
        messages = self._session_messages + self._history + [user_message]

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }

        compact_payload = json.dumps(payload, ensure_ascii=False)
        self.logger.debug(
            "Dispatching HTTP request for model=%s with headline #%s",
            self.model,
            self._headline_counter + 1,
        )
        self.logger.debug("SEND → %s", compact_payload)

        try:
            response = requests.post(self.url, json=payload, timeout=self.request_timeout)
            response.raise_for_status()
        except Exception as exc:
            self.logger.error("[%s] HTTP error: %s", self.model, exc)
            return None

        raw_text = response.text.strip()
        compact_response = raw_text.replace("\n", " ").replace("  ", " ")
        self.logger.debug("RECV ← %s", compact_response)

        parsed_response, content_text = self._extract_first_object(raw_text)
        if not parsed_response:
            self.logger.error("[%s] Failed to parse valid JSON object.", self.model)
            return None

        normalized = self._normalize_payload(parsed_response)
        if normalized is None:
            self.logger.error("[%s] Response failed schema validation.", self.model)
            return None

        # Persist successful interaction in the conversation history.
        self._history.extend([user_message, {"role": "assistant", "content": content_text}])
        self._headline_counter += 1

        return normalized

    # ------------------------------------------------------------------
    def reinforce_compliance(self) -> None:
        """Reinforce the JSON-only instruction before retrying a headline."""
        if not self._active:
            return
        reminder = {"role": "user", "content": self.JSON_ONLY_INSTRUCTION}
        self._history.append(reminder)
        self._manual_compliance_reminders += 1
        self.logger.warning("[%s] Re-sent JSON compliance instruction.", self.model)

    # ------------------------------------------------------------------
    def _format_headline(self, headline: str) -> str:
        index = self._headline_counter + 1
        return f"{index}. {clean_prompt_text(headline)}"

    def _extract_first_object(self, raw_text: str) -> Tuple[Optional[Dict[str, object]], str]:
        """Extract the first JSON object from the model response."""
        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            self.logger.debug("[%s] Top-level JSON parse failed: %s", self.model, exc)
            return None, ""

        try:
            content = payload["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            self.logger.error("[%s] Unexpected response envelope: %s", self.model, exc)
            return None, ""

        self.logger.debug("PARSING JSON OBJECT → %s", content[:500])
        repaired = self._repair_and_parse_json(content)

        if isinstance(repaired, dict):
            return repaired, content

        if isinstance(repaired, list):
            dict_items = [item for item in repaired if isinstance(item, dict)]
            if dict_items:
                if len(dict_items) > 1:
                    self._array_warning_count += 1
                    self.logger.warning(
                        "[%s] Received array with %s objects; using the first entry.",
                        self.model,
                        len(dict_items),
                    )
                return dict_items[0], content

        return None, content

    def _repair_and_parse_json(
        self, text: str
    ) -> Optional[Dict[str, object]] | List[Dict[str, object]] | None:
        original = text.strip()
        fence_cleaned = re.sub(r"```(?:json)?", "", original, flags=re.IGNORECASE)
        fence_cleaned = fence_cleaned.replace("```", "").strip()

        # Try direct parse of the cleaned response.
        for candidate in (fence_cleaned, original):
            if candidate:
                normalized = self._trim_to_json(candidate)
                if normalized:
                    try:
                        return json.loads(normalized)
                    except Exception:
                        pass

        # Replace adjacent objects with array-style separators.
        squashed = re.sub(r"}\s*{", "},{", fence_cleaned)
        if squashed != fence_cleaned:
            try:
                return json.loads(f"[{squashed}]")
            except Exception:
                pass

        objs = re.findall(r"\{[^{}]*\}", fence_cleaned)
        result: List[Dict[str, object]] = []
        for obj_text in objs:
            try:
                result.append(json.loads(obj_text))
            except Exception:
                continue
        return result if result else None

    def _trim_to_json(self, text: str) -> Optional[str]:
        """Return the substring spanning the first and last JSON delimiters."""

        stripped = text.strip()
        if not stripped:
            return None

        if stripped[0] in "[{" and stripped[-1] in "]}":
            return stripped

        start_obj = stripped.find("{")
        end_obj = stripped.rfind("}")
        start_arr = stripped.find("[")
        end_arr = stripped.rfind("]")

        candidates = []
        if start_obj != -1 and end_obj > start_obj:
            candidates.append(stripped[start_obj : end_obj + 1])
        if start_arr != -1 and end_arr > start_arr:
            candidates.append(stripped[start_arr : end_arr + 1])

        for candidate in candidates:
            candidate = candidate.strip()
            if candidate:
                return candidate
        return None

    def _normalize_payload(self, payload: Dict[str, object]) -> Optional[Dict[str, object]]:
        """Coerce malformed responses into the expected schema when possible."""

        if not isinstance(payload, dict):
            self.logger.debug("[%s] Payload is not an object: %r", self.model, payload)
            return None

        normalized: Dict[str, object] = {}
        missing_keys = []

        for key in self.REQUIRED_KEYS:
            value = payload.get(key)
            if key in self.LIST_KEYS:
                coerced_list, replaced = self._coerce_to_list(value, key)
                normalized[key] = coerced_list
            else:
                coerced_value, replaced = self._coerce_to_string(value)
                normalized[key] = coerced_value

            if replaced:
                missing_keys.append(key)

        if missing_keys:
            self.logger.debug(
                "[%s] Filled missing keys with defaults: %s",
                self.model,
                ", ".join(sorted(missing_keys)),
            )

        # Preserve any extra keys the model might return.
        for key, value in payload.items():
            if key not in normalized:
                normalized[key] = value

        # Final validation to ensure critical fields are non-empty.
        required_strings = {key for key in self.REQUIRED_KEYS if key not in self.LIST_KEYS}
        for key in required_strings:
            if not isinstance(normalized.get(key), str):
                self.logger.debug(
                    "[%s] Key '%s' could not be coerced to string.",
                    self.model,
                    key,
                )
                return None

        for key in self.LIST_KEYS:
            if not isinstance(normalized.get(key), list):
                self.logger.debug(
                    "[%s] Key '%s' could not be coerced to list.",
                    self.model,
                    key,
                )
                return None

        ordered = OrderedDict()
        if "core_event" in normalized:
            ordered["core_event"] = normalized.pop("core_event")
        for key in sorted(normalized):
            ordered[key] = normalized[key]
        return ordered

    def _normalise_prompt_section(self, section: str) -> str:
        if not isinstance(section, str):
            return ""
        return section.replace("\r\n", "\n").replace("\r", "\n").strip()

    def _coerce_to_string(self, value) -> Tuple[str, bool]:
        replaced = False
        if value is None:
            return "", True
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                replaced = True
            return stripped, replaced
        if isinstance(value, (int, float)):
            return str(value), False
        text = str(value).strip()
        if not text:
            replaced = True
        return text, replaced

    def _coerce_to_list(self, value, key: str) -> Tuple[List[str], bool]:
        replaced = False
        if value is None:
            return [], True
        if isinstance(value, list):
            cleaned = []
            for item in value:
                coerced, item_replaced = self._coerce_to_string(item)
                if coerced:
                    cleaned.append(coerced)
                replaced = replaced or item_replaced
            return cleaned, replaced
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return [], True
            parts = [part.strip() for part in re.split(r"[,;\n]+", stripped) if part.strip()]
            if parts:
                self.logger.debug(
                    "[%s] Coerced string to list for key '%s': %s",
                    self.model,
                    key,
                    parts,
                )
            else:
                replaced = True
            return parts, replaced
        if isinstance(value, (int, float)):
            return [str(value)], False
        coerced, item_replaced = self._coerce_to_string(value)
        return ([coerced] if coerced else []), item_replaced

    def _maybe_send_auto_compliance_reminder(self) -> None:
        if not self._active or self.compliance_interval <= 0:
            return
        if self._headline_counter and self._headline_counter % self.compliance_interval == 0:
            reminder = {"role": "user", "content": self.JSON_ONLY_INSTRUCTION}
            self._history.append(reminder)
            self._auto_compliance_reminders += 1
            self.logger.info(
                "[%s] Automatically re-sent JSON compliance instruction after %s headline(s).",
                self.model,
                self._headline_counter,
            )

    @property
    def auto_compliance_reminders(self) -> int:
        return self._auto_compliance_reminders

    @property
    def manual_compliance_reminders(self) -> int:
        return self._manual_compliance_reminders

    @property
    def array_warning_count(self) -> int:
        return self._array_warning_count
