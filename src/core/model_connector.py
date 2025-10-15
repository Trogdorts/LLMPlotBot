"""Model connector for sequential task execution with persistent sessions."""

import json
import re
from typing import Dict, List, Optional, Tuple

import requests


def clean_prompt_text(text: str) -> str:
    """Remove newlines, tabs, carriage returns, and repeated spaces."""
    if not isinstance(text, str):
        return text
    text = re.sub(r"[\n\r\t]+", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


class ModelConnector:
    """Blocking HTTP client that maintains one persistent chat session per model."""

    JSON_ONLY_INSTRUCTION = "Return only the JSON object described. No extra text."

    REQUIRED_KEYS = {
        "id",
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

    def __init__(self, model: str, url: str, request_timeout: int, logger):
        self.model = model
        self.url = url
        self.request_timeout = request_timeout
        self.logger = logger
        self._session_messages: List[Dict[str, str]] = []
        self._history: List[Dict[str, str]] = []
        self._active = False
        self._headline_counter = 0

    # ------------------------------------------------------------------
    def start_session(self, prompt_text: str) -> None:
        """Initialise the persistent message history for this model."""
        self.logger.info("Starting session for model %s", self.model)
        cleaned_prompt = clean_prompt_text(prompt_text)
        self._session_messages = [
            {"role": "system", "content": self.JSON_ONLY_INSTRUCTION},
            {"role": "user", "content": cleaned_prompt},
        ]
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

        if not self._validate_schema(parsed_response):
            self.logger.error("[%s] Response failed schema validation.", self.model)
            return None

        # Persist successful interaction in the conversation history.
        self._history.extend([user_message, {"role": "assistant", "content": content_text}])
        self._headline_counter += 1

        return parsed_response

    # ------------------------------------------------------------------
    def reinforce_compliance(self) -> None:
        """Reinforce the JSON-only instruction before retrying a headline."""
        if not self._active:
            return
        reminder = {"role": "user", "content": self.JSON_ONLY_INSTRUCTION}
        self._history.append(reminder)
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
                    self.logger.warning(
                        "[%s] Received array with %s objects; using the first entry.",
                        self.model,
                        len(dict_items),
                    )
                return dict_items[0], content

        return None, content

    def _repair_and_parse_json(self, text: str):
        original = text.strip()
        text = re.sub(r"[\n\r\t]+", " ", original)
        text = re.sub(r"\s{2,}", " ", text)

        try:
            return json.loads(text)
        except Exception:
            pass

        text2 = re.sub(r"}\s*{", "},{", text)
        try:
            return json.loads(text2)
        except Exception:
            pass

        try:
            if text.strip().startswith("{") and text.strip().endswith("}"):
                return json.loads(text)
        except Exception:
            pass

        objs = re.findall(r"\{[^{}]*\}", original)
        result = []
        for obj_text in objs:
            try:
                result.append(json.loads(obj_text))
            except Exception:
                continue
        return result if result else None

    def _validate_schema(self, payload: Dict[str, object]) -> bool:
        missing = self.REQUIRED_KEYS - payload.keys()
        if missing:
            self.logger.debug("[%s] Missing keys: %s", self.model, ", ".join(sorted(missing)))
            return False

        for key in self.REQUIRED_KEYS - self.LIST_KEYS:
            if not isinstance(payload.get(key), str):
                self.logger.debug("[%s] Key '%s' should be a string.", self.model, key)
                return False

        for key in self.LIST_KEYS:
            value = payload.get(key)
            if not isinstance(value, list):
                self.logger.debug("[%s] Key '%s' should be a list.", self.model, key)
                return False
            if not all(isinstance(item, str) for item in value):
                self.logger.debug("[%s] Key '%s' contains non-string items.", self.model, key)
                return False

        return True
