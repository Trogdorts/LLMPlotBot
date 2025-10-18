"""Resolve active LLM endpoints for the processing pipeline."""

from __future__ import annotations

import logging
from typing import Dict, Iterable, Mapping

import requests


def resolve_endpoints(config: Mapping[str, object], logger: logging.Logger | None = None) -> Dict[str, str]:
    """Return a mapping of model name -> endpoint URL based on configuration."""

    blocklist = {m.lower() for m in _normalise_model_list(config.get("LLM_BLOCKLIST"))}
    explicit = config.get("LLM_ENDPOINTS") or {}
    endpoints: Dict[str, str] = {}

    if isinstance(explicit, Mapping) and explicit:
        for model, url in explicit.items():
            if model and isinstance(url, str) and model.lower() not in blocklist:
                endpoints[str(model)] = url
        if logger and not endpoints:
            logger.warning("LLM_ENDPOINTS configuration contained no usable entries.")
        return endpoints

    models = list(_normalise_model_list(config.get("LLM_MODELS")))
    base_url = str(config.get("LLM_BASE_URL") or "http://localhost:1234").rstrip("/")
    if not models:
        models = fetch_lmstudio_models(base_url, logger)
    for model in models:
        if model.lower() in blocklist:
            continue
        endpoints[model] = f"{base_url}/v1/chat/completions"
    return endpoints


def fetch_lmstudio_models(base_url: str, logger: logging.Logger | None = None) -> Iterable[str]:
    """Query LM Studio's REST API for available models."""

    url = f"{base_url.rstrip('/')}/v1/models"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
    except requests.RequestException as exc:  # pragma: no cover - network failure
        if logger:
            logger.warning("Unable to query LM Studio models at %s: %s", url, exc)
        return []

    try:
        payload = response.json()
    except ValueError:  # pragma: no cover - invalid response
        if logger:
            logger.warning("LM Studio returned non-JSON payload from %s", url)
        return []

    data = payload.get("data") if isinstance(payload, Mapping) else None
    if not isinstance(data, Iterable):
        return []

    models: list[str] = []
    for entry in data:
        if not isinstance(entry, Mapping):
            continue
        model_id = entry.get("id")
        if isinstance(model_id, str):
            models.append(model_id)
    if logger and models:
        logger.info("Discovered %d LM Studio model(s).", len(models))
    return models


def _normalise_model_list(value: object) -> Iterable[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, Iterable):
        models: list[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                models.append(item.strip())
        return models
    return []


__all__ = ["resolve_endpoints", "fetch_lmstudio_models"]
