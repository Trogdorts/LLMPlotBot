"""Model connector for batched task execution with persistent sessions."""

from __future__ import annotations

import json
import re
import string
import unicodedata
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import requests

from requests import Session


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

    _NON_ASCII_RE = re.compile(r"[^\x00-\x7F]+")
    _LANGUAGE_REPLACEMENTS = {
        "狱警": "prison guard",
        "囚犯": "prisoner",
        "官员": "official",
        "凶手": "killer",
        "监狱": "prison",
    }
    _LANGUAGE_PLACEHOLDER = "unknown"

    def __init__(
        self,
        model: str,
        url: str,
        request_timeout: int,
        compliance_interval: int,
        logger,
        expected_language: str | None = None,
    ):
        self.model = model
        self.url = url
        self.request_timeout = request_timeout
        self.logger = logger
        self.compliance_interval = max(0, int(compliance_interval or 0))
        self.expected_language = (expected_language or "").strip().lower()
        self._language_checker = None
        if self.expected_language == "en":
            self._language_checker = self._looks_english
        elif self.expected_language:
            self.logger.warning(
                "[%s] Language '%s' not supported for validation; skipping check.",
                self.model,
                self.expected_language,
            )
        self._session_messages: List[Dict[str, str]] = []
        self._history: List[Dict[str, str]] = []
        # Limit the amount of conversation state retained in memory to avoid
        # unbounded growth when processing large batches of headlines.
        self._history_limit = 50  # stores up to 25 prompt/response pairs
        self._active = False
        self._headline_counter = 0
        self._auto_compliance_reminders = 0
        self._manual_compliance_reminders = 0
        self._array_warning_count = 0
        self._last_request_error: Optional[requests.RequestException] = None
        self._http: Session = Session()
        self._http.headers.update({"Content-Type": "application/json"})
        self._http.trust_env = False

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
        """Backward-compatible wrapper that processes a single headline."""

        results = self.send_batch([headline])
        if not results:
            return None
        return results[0]

    def send_batch(
        self, headlines: List[str]
    ) -> Optional[List[Optional[Dict[str, object]]]]:
        """Send multiple headlines and return validated payloads in order."""

        if not headlines:
            return []

        if not self._active:
            raise RuntimeError("Session has not been started. Call start_session() first.")

        self._maybe_send_auto_compliance_reminder()

        start_index = self._headline_counter + 1
        numbered_lines = [
            f"{start_index + idx}. {clean_prompt_text(headline)}"
            for idx, headline in enumerate(headlines)
        ]
        batch_text = "\n".join(numbered_lines)
        user_message = {"role": "user", "content": batch_text}
        messages = self._session_messages + self._history + [user_message]

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }

        compact_payload = json.dumps(payload, ensure_ascii=False)
        self.logger.debug(
            "Dispatching HTTP request for model=%s with headlines #%s-%s",
            self.model,
            start_index,
            start_index + len(headlines) - 1,
        )
        self.logger.debug("SEND → %s", compact_payload)

        self._last_request_error = None
        try:
            response = self._http.post(
                self.url, json=payload, timeout=self.request_timeout
            )
        except requests.RequestException as exc:
            self.logger.error("[%s] HTTP request failed: %s", self.model, exc)
            self._last_request_error = exc
            return None

        raw_text = response.text.strip()
        compact_response = raw_text.replace("\n", " ").replace("  ", " ")
        self.logger.debug("RECV ← %s", compact_response)

        if response.status_code >= 400:
            self.logger.error(
                "[%s] HTTP error %s: %s",
                self.model,
                response.status_code,
                response.reason,
            )
            self._last_request_error = requests.HTTPError(
                f"HTTP {response.status_code} {response.reason}"
            )
            return None

        parsed_entries, content_text = self._extract_response_entries(raw_text)
        if parsed_entries is None:
            self.logger.error("[%s] Failed to parse JSON array response.", self.model)
            return None

        results: List[Optional[Dict[str, object]]] = [None] * len(headlines)
        max_index = min(len(parsed_entries), len(headlines))

        for offset in range(max_index):
            normalized = self._normalize_payload(parsed_entries[offset])
            entry_number = start_index + offset
            if normalized is None:
                self.logger.error(
                    "[%s] Entry %s failed schema validation.", self.model, entry_number
                )
                continue

            if self._language_checker:
                sanitized = self._ensure_english_payload(normalized)
                if sanitized is not normalized:
                    normalized = sanitized

            if self._language_checker and not self._validate_language(normalized):
                self.logger.error(
                    "[%s] Entry %s failed language validation for '%s'.",
                    self.model,
                    entry_number,
                    self.expected_language,
                )
                continue

            results[offset] = normalized

        if len(parsed_entries) > len(headlines):
            self._array_warning_count += 1
            self.logger.warning(
                "[%s] Received %s objects for %s headline(s); ignoring extras.",
                self.model,
                len(parsed_entries),
                len(headlines),
            )
        elif len(parsed_entries) < len(headlines):
            missing = len(headlines) - len(parsed_entries)
            if missing:
                self.logger.warning(
                    "[%s] Missing %s response(s) in returned array.",
                    self.model,
                    missing,
                )

        success_count = sum(1 for item in results if item is not None)
        if success_count:
            # Persist successful interaction in the conversation history.
            self._history.extend(
                [user_message, {"role": "assistant", "content": content_text}]
            )
            self._prune_history()
            self._headline_counter += success_count

        return results

    def pop_last_request_error(self) -> Optional[requests.RequestException]:
        """Return and clear the most recent request-level error, if any."""

        error = self._last_request_error
        self._last_request_error = None
        return error

    # ------------------------------------------------------------------
    def reinforce_compliance(self) -> None:
        """Reinforce the JSON-only instruction before retrying a headline."""
        if not self._active:
            return
        reminder = {"role": "user", "content": self.JSON_ONLY_INSTRUCTION}
        self._history.append(reminder)
        self._prune_history()
        self._manual_compliance_reminders += 1
        self.logger.warning("[%s] Re-sent JSON compliance instruction.", self.model)

    # ------------------------------------------------------------------
    def _extract_response_entries(
        self, raw_text: str
    ) -> Tuple[Optional[List[Dict[str, object]]], str]:
        """Extract a list of JSON objects from the model response content."""

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

        self.logger.debug("PARSING JSON ARRAY → %s", content[:500])
        repaired = self._repair_and_parse_json(content)

        entries: Optional[List[Dict[str, object]]] = None
        normalized_content = content

        if isinstance(repaired, dict):
            entries = [repaired]
        elif isinstance(repaired, list):
            dict_items = [item for item in repaired if isinstance(item, dict)]
            entries = dict_items or None

        if entries:
            normalized_content = self._normalise_response_content(entries)
            if normalized_content != content:
                self.logger.debug(
                    "[%s] Normalized assistant content before caching.", self.model
                )
            return entries, normalized_content

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

    def _normalise_response_content(
        self, entries: List[Dict[str, object]]
    ) -> str:
        """Return a compact JSON array string for caching in the conversation."""

        try:
            return json.dumps(entries, ensure_ascii=False, separators=(",", ": "))
        except TypeError:
            serializable = json.loads(json.dumps(entries, default=str, ensure_ascii=False))
            return json.dumps(serializable, ensure_ascii=False, separators=(",", ": "))

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

    def _validate_language(self, payload: Dict[str, object]) -> bool:
        if not self._language_checker:
            return True

        snippets: List[str] = []
        for value in payload.values():
            if isinstance(value, str):
                snippets.append(value)
            elif isinstance(value, list):
                snippets.extend(str(item) for item in value if isinstance(item, str))

        for snippet in snippets:
            if snippet and not self._language_checker(snippet):
                preview = snippet.strip().splitlines()[0][:80]
                self.logger.debug(
                    "[%s] Language check failed for text: %r",
                    self.model,
                    preview,
                )
                return False
        return True

    def _ensure_english_payload(self, payload: Dict[str, object]) -> Dict[str, object]:
        """Return a payload with non-English segments replaced by safe English text."""

        updated: Dict[str, object] = payload.__class__()
        changed = False

        for key, value in payload.items():
            cleaned, mutated = self._sanitize_value_to_english(value)
            if mutated:
                changed = True
            updated[key] = cleaned

        if changed:
            self.logger.debug("[%s] Sanitized non-English text in response.", self.model)
        return updated if changed else payload

    def _sanitize_value_to_english(self, value: Any) -> Tuple[Any, bool]:
        if isinstance(value, str):
            cleaned = self._sanitize_string_to_english(value)
            return cleaned, cleaned != value
        if isinstance(value, list):
            cleaned_list = []
            mutated = False
            for item in value:
                if isinstance(item, str):
                    cleaned_item = self._sanitize_string_to_english(item)
                    if cleaned_item != item:
                        mutated = True
                    cleaned_list.append(cleaned_item)
                else:
                    cleaned_list.append(item)
            return cleaned_list, mutated
        if isinstance(value, dict):
            cleaned_dict = {}
            mutated = False
            for sub_key, sub_value in value.items():
                cleaned_sub_value, mutated_sub = self._sanitize_value_to_english(sub_value)
                if mutated_sub:
                    mutated = True
                cleaned_dict[sub_key] = cleaned_sub_value
            return cleaned_dict, mutated
        return value, False

    def _sanitize_string_to_english(self, text: str) -> str:
        if not text:
            return text

        normalized = unicodedata.normalize("NFKC", text)
        if self._looks_english(normalized):
            return normalized

        def replace_segment(match: re.Match[str]) -> str:
            segment = match.group(0)
            mapped = self._LANGUAGE_REPLACEMENTS.get(segment)
            if not mapped:
                mapped_chars = [self._LANGUAGE_REPLACEMENTS.get(char, "") for char in segment]
                mapped = " ".join(filter(None, mapped_chars)).strip()
            if not mapped:
                transliterated = unicodedata.normalize("NFKD", segment).encode(
                    "ascii", "ignore"
                ).decode("ascii", "ignore").strip()
                mapped = transliterated
            if not mapped:
                mapped = self._LANGUAGE_PLACEHOLDER
            return f" {mapped} "

        cleaned = self._NON_ASCII_RE.sub(replace_segment, normalized)
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()

        return cleaned or self._LANGUAGE_PLACEHOLDER

    def _looks_english(self, text: str) -> bool:
        if not text:
            return True

        alpha_chars = [char for char in text if char.isalpha()]
        if not alpha_chars:
            return True

        ascii_alpha = sum(1 for char in alpha_chars if char in string.ascii_letters)
        ratio = ascii_alpha / len(alpha_chars)
        if ratio < 0.85:
            return False

        non_ascii = sum(1 for char in text if not char.isascii())
        if non_ascii and (non_ascii / len(text)) > 0.2:
            return False

        return True

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
            self._prune_history()
            self._auto_compliance_reminders += 1
            self.logger.info(
                "[%s] Automatically re-sent JSON compliance instruction after %s headline(s).",
                self.model,
                self._headline_counter,
            )

    def _prune_history(self) -> None:
        """Trim stored history to the configured limit."""

        if self._history_limit <= 0:
            return

        overflow = len(self._history) - self._history_limit
        if overflow > 0:
            # Keep only the most recent messages. We slice rather than popping in
            # a loop to avoid quadratic time behaviour.
            self._history = self._history[overflow:]

    @property
    def auto_compliance_reminders(self) -> int:
        return self._auto_compliance_reminders

    @property
    def manual_compliance_reminders(self) -> int:
        return self._manual_compliance_reminders

    @property
    def array_warning_count(self) -> int:
        return self._array_warning_count

    # ------------------------------------------------------------------
    def shutdown(self) -> None:
        """Close the HTTP session and reset conversation state."""

        self.close_session()
        self._http.close()
