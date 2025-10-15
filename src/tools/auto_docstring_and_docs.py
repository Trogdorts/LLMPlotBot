#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
docstring_updater.py
--------------------------------------------------
Adds or updates ONLY Python function/class docstrings
across the entire repository while preserving all code formatting.

Features:
- Connects to LM Studio (OpenAI-compatible endpoint)
- Smart retries, validation, and resume support
- Hash-based skip for unchanged functions
- Safe atomic writes
- Proper indentation for multiline docstrings
- Progress bar and logging

Run:
  python -m src.tools.docstring_updater \
    --root ./src \
    --llm-url http://localhost:1234/v1/chat/completions \
    --model pranav-pvnn/codellama-7b-python-ai-assistant-full-gguf
"""

import os
import ast
import re
import sys
import json
import time
import hashlib
import requests
import argparse
import logging
from tqdm import tqdm
from requests.exceptions import RequestException

# ===============================================================
# Default Configuration
# ===============================================================
DEFAULT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_LOG_DIR = os.path.abspath(os.path.join(DEFAULT_ROOT, "..", "logs"))
DEFAULT_STATE_FILE = os.path.join(DEFAULT_LOG_DIR, "docstring_state.json")
DEFAULT_LLM_URL = "http://localhost:1234/v1/chat/completions"
DEFAULT_MODEL = "pranav-pvnn/codellama-7b-python-ai-assistant-full-gguf"

# ===============================================================
# Logging
# ===============================================================
def setup_logger(level: str, log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("docstring_updater")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()

    fh = logging.FileHandler(os.path.join(log_dir, "docstring_updater.log"), encoding="utf-8")
    fh.setLevel(getattr(logging, level.upper(), logging.INFO))
    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, level.upper(), logging.INFO))

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

# ===============================================================
# State Persistence
# ===============================================================
def load_state(path: str) -> dict:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"functions": {}}

def save_state(path: str, data: dict):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)

# ===============================================================
# Helpers
# ===============================================================
def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def sanitize_llm_docstring(raw: str) -> str:
    raw = raw.strip()
    raw = re.sub(r"^```[\w-]*", "", raw)
    raw = re.sub(r"```$", "", raw)
    match = re.search(r'"""(.*?)"""', raw, re.DOTALL)
    if not match:
        match = re.search(r"'''(.*?)'''", raw, re.DOTALL)
    text = match.group(1) if match else raw
    text = text.replace('"""', '\\"""').strip()
    return text

def normalize_docstring_indentation(content: str, indent: str) -> str:
    """Indent each line consistently under its triple quotes."""
    lines = content.strip().splitlines()
    if not lines:
        return f'{indent}"""\n{indent}"""'
    formatted = [f'{indent}"""{lines[0].strip()}']
    for line in lines[1:]:
        formatted.append(f'{indent}    {line.strip()}')
    formatted.append(f'{indent}"""')
    return "\n".join(formatted) + "\n"

def build_minimal_docstring(name: str, args: list) -> str:
    arg_section = ""
    if args:
        formatted = "\n".join([f"    {a}: Description." for a in args])
        arg_section = f"\n\nArgs:\n{formatted}"
    return f"{name}.\n{arg_section}"

# ===============================================================
# LLM Request
# ===============================================================
def call_llm(llm_url, model, code, retries, backoff, logger):
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a Python documentation assistant. "
                    "Return ONLY a Python docstring literal using triple double quotes. "
                    "Use Summary, Args, Returns, Raises as needed."
                ),
            },
            {"role": "user", "content": code},
        ],
        "temperature": 0.2,
        "stream": False,
    }

    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(llm_url, json=payload, timeout=180)
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"]
            return sanitize_llm_docstring(text)
        except (RequestException, KeyError, ValueError) as e:
            logger.warning(f"LLM attempt {attempt}/{retries} failed: {e}")
            if attempt < retries:
                time.sleep(backoff * attempt)
    return ""

# ===============================================================
# AST Helpers
# ===============================================================
def collect_targets(code: str, path: str):
    """Return AST nodes for all function/class definitions."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    nodes = []
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            nodes.append(n)
    return nodes

def extract_args(node: ast.AST) -> list:
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return []
    args = [a.arg for a in node.args.args if a.arg not in ("self", "cls")]
    return args

# ===============================================================
# Core Function
# ===============================================================
def process_file(path, state, llm_url, model, retries, backoff, logger, dry_run=False):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    code = "".join(lines)

    nodes = collect_targets(code, path)
    if not nodes:
        return (0, 0)

    changed = 0
    skipped = 0

    for node in sorted(nodes, key=lambda n: n.lineno, reverse=True):
        name = getattr(node, "name", "<unknown>")
        key = f"{path}:{name}"
        snippet = code[node.lineno - 1 : node.end_lineno]
        func_hash = sha256(snippet)

        if state["functions"].get(key, {}).get("hash") == func_hash:
            skipped += 1
            continue

        doc = ast.get_docstring(node, clean=False)
        indent = " " * (node.col_offset + 4)
        args = extract_args(node)
        logger.info(f"Processing {key}")

        new_doc = ""
        if not dry_run:
            result = call_llm(llm_url, model, snippet, retries, backoff, logger)
            new_doc = result or build_minimal_docstring(name, args)
        else:
            new_doc = build_minimal_docstring(name, args)

        literal = normalize_docstring_indentation(new_doc, indent)

        # Determine insert or replace
        doc_node = (
            node.body[0]
            if node.body and isinstance(node.body[0], ast.Expr)
            and isinstance(getattr(node.body[0], "value", None), ast.Constant)
            and isinstance(node.body[0].value.value, str)
            else None
        )

        if doc_node:
            start, end = doc_node.lineno - 1, doc_node.end_lineno
            lines[start:end] = [literal]
        else:
            insert_line = node.body[0].lineno - 1 if node.body else node.lineno
            lines.insert(insert_line, literal)

        state["functions"][key] = {"hash": func_hash, "updated_at": time.time()}
        changed += 1

    if changed and not dry_run:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.writelines(lines)
        os.replace(tmp, path)
    return (changed, skipped)

# ===============================================================
# Main
# ===============================================================
def main():
    ap = argparse.ArgumentParser(description="Smart docstring updater for Python projects.")
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--llm-url", default=DEFAULT_LLM_URL)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--log-level", default="INFO")
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--backoff", type=float, default=2.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--state", default=DEFAULT_STATE_FILE)
    args = ap.parse_args()

    logger = setup_logger(args.log_level, DEFAULT_LOG_DIR)
    state = load_state(args.state)

    py_files = []
    for root, _, files in os.walk(args.root):
        for f in files:
            if f.endswith(".py") and not f.startswith("."):
                py_files.append(os.path.join(root, f))

    logger.info(f"Found {len(py_files)} Python files.")
    total_changed = 0
    total_skipped = 0

    for path in tqdm(py_files, desc="Processing files", unit="file"):
        try:
            ch, sk = process_file(
                path, state,
                args.llm_url, args.model,
                args.retries, args.backoff,
                logger, dry_run=args.dry_run
            )
            total_changed += ch
            total_skipped += sk
            save_state(args.state, state)
        except Exception as e:
            logger.error(f"Error in {path}: {e}")

    logger.info(f"Done. Changed={total_changed}, Skipped={total_skipped}")
    print(f"\n✅ Changed={total_changed}, Skipped={total_skipped}")

if __name__ == "__main__":
    main()
