"""
ModelConnector
==============
Handles all HTTP interactions with LLM endpoints.
Console → pretty-printed JSON
logs/debug.log → compact single-line JSON
Cleans prompt text and uses aggressive JSON recovery.
"""

import json, re, requests
from typing import List
from .task import Task


def clean_prompt_text(text: str) -> str:
    """Remove newlines, tabs, carriage returns, and repeated spaces."""
    if not isinstance(text, str):
        return text
    text = re.sub(r'[\n\r\t]+', ' ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()


class ModelConnector:
    """Blocking HTTP client for one LLM endpoint."""

    def __init__(self, model: str, url: str, request_timeout: int, logger):
        self.model = model
        self.url = url
        self.request_timeout = request_timeout
        self.logger = logger

    # ------------------------------------------------------------------
    def send_batch(self, batch: List[Task]):
        """Send a batch and return parsed JSON results."""
        user_content = clean_prompt_text(self._build_prompt(batch))

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "Return only the JSON object described. No extra text."},
                {"role": "user", "content": user_content},
            ],
            "stream": False,
        }

        divider = "=" * 100
        compact_payload = json.dumps(payload, ensure_ascii=False)
        pretty_payload = json.dumps(payload, ensure_ascii=False, indent=2)

        # --- outbound ---
        self.logger.debug(f"SEND → {compact_payload}")

        try:
            r = requests.post(self.url, json=payload, timeout=self.request_timeout)
            r.raise_for_status()
            raw_text = r.text.strip()

            # Pretty JSON for console; compact for log
            compact_response = raw_text.replace("\n", "").replace("  ", " ")
            self.logger.debug(f"RECV ← {compact_response}")

            try:
                parsed = json.loads(raw_text)
                pretty_response = json.dumps(parsed, ensure_ascii=False, indent=2)
                # Pretty-print the embedded content field
                content = parsed.get("choices", [{}])[0].get("message", {}).get("content")
            except Exception:
                print(
                    f"\n{divider}\n[DEBUG] MODEL: {self.model}\n"
                    f"[DEBUG] RESPONSE STATUS: {r.status_code}\n{divider}\n"
                    f"[DEBUG] RESPONSE BODY (raw):\n{raw_text}\n{divider}\n"
                )

            try:
                data = json.loads(raw_text)
            except Exception as e:
                self.logger.error(f"[{self.model}] JSON parse error: {e}")
                print(f"[ERROR] {self.model} JSON parse error: {e}")
                return [None] * len(batch)

            return self._parse_batch_response(data, len(batch))

        except Exception as e:
            msg = f"[{self.model}] batch error: {e}"
            self.logger.error(msg)
            return [None] * len(batch)

    # ------------------------------------------------------------------
    def _build_prompt(self, batch: List[Task]) -> str:
        """Combine all headlines into one batch prompt."""
        prompt_text = batch[0].prompt_text
        lines = [f"{i+1}. {t.title}" for i, t in enumerate(batch)]
        return f"{prompt_text}\n\nHeadlines:\n" + "\n".join(lines)

    # ------------------------------------------------------------------
    def _repair_and_parse_json(self, text: str):
        """
        Attempt layered JSON repair and return a valid parsed structure or None.
        """
        original = text
        text = text.strip()

        # Strip non-JSON prefix/suffix
        start, end = text.find("["), text.rfind("]")
        if start != -1 and end != -1:
            text = text[start:end + 1]

        # Common cleanups
        text = text.replace("“", '"').replace("”", '"')
        text = re.sub(r",\s*]", "]", text)
        text = re.sub(r",\s*}", "}", text)
        text = re.sub(r"\n+", " ", text)
        text = re.sub(r"\s{2,}", " ", text)

        # try direct parse first
        try:
            return json.loads(text)
        except Exception:
            pass

        # fix missing commas between }{
        text2 = re.sub(r"}\s*{", "},{", text)
        try:
            return json.loads(text2)
        except Exception:
            pass

        # try to wrap single object
        try:
            if text.strip().startswith("{") and text.strip().endswith("}"):
                return [json.loads(text)]
        except Exception:
            pass

        # remove trailing punctuation/braces
        text3 = re.sub(r"[\],\}]$", "]", text2)
        try:
            return json.loads(text3)
        except Exception:
            pass

        # last-ditch: extract possible object segments
        objs = re.findall(r"\{[^{}]*\}", original)
        result = []
        for o in objs:
            try:
                result.append(json.loads(o))
            except Exception:
                continue
        if result:
            return result

        return None

    # ------------------------------------------------------------------
    def _parse_batch_response(self, data, count: int):
        """
        Extract and repair JSON array returned by the model.
        Only returns a fully valid parsed object; never writes failed output.
        """
        try:
            if isinstance(data, dict) and "choices" in data:
                text = data["choices"][0]["message"]["content"].strip()
                self.logger.debug(f"PARSING JSON ARRAY → {text[:500]}...")

                parsed = self._repair_and_parse_json(text)
                if not parsed:
                    self.logger.error(f"[{self.model}] Unable to recover valid JSON after repair.")
                    return [None] * count

                # sanity check
                if isinstance(parsed, dict):
                    return [parsed] * count
                if isinstance(parsed, list):
                    valid_objs = [obj for obj in parsed if isinstance(obj, dict)]
                    if valid_objs:
                        return valid_objs
        except Exception as e:
            self.logger.error(f"[{self.model}] Unexpected parse error: {e}")

        return [None] * count
