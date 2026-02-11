#!/usr/bin/env python3
"""
Translate exercises/exercises.json to a target language using Gemini
(GEMINI_API_KEY), preserving exercise `id` and `images` exactly.

This script is intentionally dependency-free (stdlib only).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Iterable, List, Tuple


IMAGE_EXT_RE = re.compile(r"\.(?:png|jpe?g|gif|webp|svg)$", re.IGNORECASE)
URL_RE = re.compile(r"^https?://", re.IGNORECASE)


class RateLimitError(RuntimeError):
    def __init__(self, retry_after_s: float, message: str) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s


def load_dotenv(path: str) -> None:
    # Minimal .env loader: KEY=VALUE, ignores blank lines and comments.
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'").strip('"')
                os.environ.setdefault(k, v)
    except FileNotFoundError:
        return


def should_translate(key: str | None, s: str) -> bool:
    if not isinstance(s, str):
        return False
    if s == "":
        return False
    if key == "id":
        return False
    if key is not None and "image" in key.lower():
        return False
    if URL_RE.match(s):
        return False
    if IMAGE_EXT_RE.search(s):
        return False
    # Common path-looking strings for this repo (e.g., "3_4_Sit-Up/0.jpg")
    if "/" in s and IMAGE_EXT_RE.search(s.split("?", 1)[0]):
        return False
    return True


def collect_string_paths(
    obj: Any,
    path: Tuple[Any, ...] = (),
    parent_key: str | None = None,
) -> List[Tuple[Tuple[Any, ...], str, str | None]]:
    out: List[Tuple[Tuple[Any, ...], str, str | None]] = []

    if isinstance(obj, dict):
        for k, v in obj.items():
            # Preserve id and images exactly.
            if k == "id":
                continue
            if k == "images":
                continue
            out.extend(collect_string_paths(v, path + (k,), k))
        return out

    if isinstance(obj, list):
        for i, v in enumerate(obj):
            out.extend(collect_string_paths(v, path + (i,), parent_key))
        return out

    if isinstance(obj, str):
        if should_translate(parent_key, obj):
            out.append((path, obj, parent_key))
        return out

    return out


def set_at_path(root: Any, path: Tuple[Any, ...], value: Any) -> None:
    cur = root
    for p in path[:-1]:
        cur = cur[p]
    cur[path[-1]] = value


def extract_json_array(text: str) -> List[str]:
    # Gemini can sometimes wrap JSON in prose/markdown; extract the first [...] block.
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Could not find JSON array in model output")
    snippet = text[start : end + 1]
    parsed = json.loads(snippet)
    if not isinstance(parsed, list) or not all(isinstance(x, str) for x in parsed):
        raise ValueError("Model output JSON is not a string array")
    return parsed


def gemini_generate(
    *,
    api_key: str,
    model: str,
    prompt: str,
    temperature: float,
    max_output_tokens: int,
    timeout_s: int,
) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_output_tokens,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        # Common cause: model name not available for the project/key -> 404.
        if e.code == 404:
            raise
        msg = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else str(e)
        if e.code == 429:
            m = re.search(r"Please retry in ([0-9]+(?:\\.[0-9]+)?)s", msg)
            retry_s = float(m.group(1)) if m else 15.0
            raise RateLimitError(retry_s, f"Gemini HTTP 429: {msg[:500]}") from e
        raise RuntimeError(f"Gemini HTTP {e.code}: {msg[:500]}") from e
    j = json.loads(body)
    try:
        return j["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"Unexpected Gemini response shape: {body[:500]}") from e


def translate_batch(
    *,
    api_key: str,
    model: str,
    target_language: str,
    src: List[str],
    timeout_s: int,
) -> List[str]:
    # Keep prompt tight; we rely on JSON-only output for deterministic parsing.
    # We avoid translating ids/paths by pre-filtering them in the collector.
    src_json = json.dumps(src, ensure_ascii=False)
    prompt = (
        f"Translate the following JSON array of English strings into natural, neutral {target_language}.\n"
        "Rules:\n"
        "- Return ONLY a valid JSON array of strings (same length, same order).\n"
        "- Do not add explanations, markdown, or extra keys.\n"
        "- Preserve numbers, reps, and punctuation.\n\n"
        f"Input:\n{src_json}\n"
    )

    # A small retry loop for transient issues / imperfect formatting.
    # If a model name is no longer valid, auto-fallback to a few common IDs.
    candidates: List[str] = [model]
    if not model.endswith("-latest"):
        candidates.append(f"{model}-latest")
    # Reasonable current defaults (if available to the key).
    candidates.extend(["gemini-2.5-flash", "gemini-2.5-flash-lite"])

    last_err: Exception | None = None
    for candidate_model in candidates:
        attempts = 0
        rate_waits = 0
        while attempts < 4:
            try:
                text = gemini_generate(
                    api_key=api_key,
                    model=candidate_model,
                    prompt=prompt,
                    temperature=0.2,
                    max_output_tokens=8192,
                    timeout_s=timeout_s,
                )
                out = extract_json_array(text)
                if len(out) != len(src):
                    raise ValueError(f"Length mismatch: expected {len(src)} got {len(out)}")
                return out
            except RateLimitError as e:
                last_err = e
                # Respect server-provided backoff.
                rate_waits += 1
                print(
                    f"Rate limited on {candidate_model}; waiting {max(1.0, e.retry_after_s) + 1.0:.1f}s (attempt {rate_waits}).",
                    file=sys.stderr,
                    flush=True,
                )
                # Avoid an infinite loop if the key is out of quota for a long time.
                if rate_waits > 60:
                    raise
                time.sleep(max(1.0, e.retry_after_s) + 1.0)
                continue
            except urllib.error.HTTPError as e:
                last_err = e
                # Try the next model on 404 (model not found).
                if getattr(e, "code", None) == 404:
                    break
                attempts += 1
                time.sleep(1.0 * attempts)
                continue
            except Exception as e:  # noqa: BLE001
                last_err = e
                attempts += 1
                time.sleep(1.0 * attempts)
                continue

    assert last_err is not None
    raise last_err


def chunked(items: List[str], n: int) -> Iterable[List[str]]:
    for i in range(0, len(items), n):
        yield items[i : i + n]


def chunked_by_chars(items: List[str], *, max_items: int, max_chars: int) -> Iterable[List[str]]:
    batch: List[str] = []
    chars = 0
    for s in items:
        s_len = len(s)
        if batch and (len(batch) >= max_items or chars + s_len > max_chars):
            yield batch
            batch = []
            chars = 0
        batch.append(s)
        chars += s_len
    if batch:
        yield batch


def load_cache(path: str) -> Dict[str, str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            j = json.load(f)
        if isinstance(j, dict):
            return {str(k): str(v) for k, v in j.items()}
    except FileNotFoundError:
        return {}
    return {}


def save_cache(path: str, cache: Dict[str, str]) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="exercises/exercises.json")
    ap.add_argument("--out", dest="out_path", default="exercises/exercises.es.json")
    ap.add_argument("--target-language", default="Spanish")
    ap.add_argument("--model", default="gemini-1.5-flash-latest")
    ap.add_argument("--batch-size", type=int, default=200)
    ap.add_argument("--batch-max-chars", type=int, default=14000)
    ap.add_argument("--timeout-s", type=int, default=120)
    ap.add_argument("--dotenv", default=".env")
    ap.add_argument("--cache", default="exercises/exercises.es.cache.json")
    args = ap.parse_args()

    load_dotenv(args.dotenv)
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        print("Missing GEMINI_API_KEY (set it in env or .env).", file=sys.stderr)
        return 2

    with open(args.in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Collect translation targets.
    paths = collect_string_paths(data)
    unique: List[str] = []
    idx_by_string: Dict[str, int] = {}
    for _, s, _ in paths:
        if s not in idx_by_string:
            idx_by_string[s] = len(unique)
            unique.append(s)

    if not unique:
        print("No translatable strings found.", file=sys.stderr)
        return 0

    translations: List[str] = [""] * len(unique)
    cache = load_cache(args.cache) if args.cache else {}
    for src_s, i in idx_by_string.items():
        if src_s in cache:
            translations[i] = cache[src_s]

    remaining = [s for s in unique if not translations[idx_by_string[s]]]
    done = len(unique) - len(remaining)
    if done:
        print(
            f"Cache hit: {done}/{len(unique)} unique strings already translated.",
            file=sys.stderr,
            flush=True,
        )

    for batch in chunked_by_chars(remaining, max_items=args.batch_size, max_chars=args.batch_max_chars):
        batch_out = translate_batch(
            api_key=api_key,
            model=args.model,
            target_language=args.target_language,
            src=batch,
            timeout_s=args.timeout_s,
        )
        for src_s, es_s in zip(batch, batch_out):
            translations[idx_by_string[src_s]] = es_s
            if args.cache:
                cache[src_s] = es_s
        if args.cache:
            save_cache(args.cache, cache)
        done += len(batch)
        print(f"Translated {done}/{len(unique)} unique strings...", file=sys.stderr, flush=True)

    # Apply translations in place.
    for p, s, _k in paths:
        es = translations[idx_by_string[s]]
        set_at_path(data, p, es)

    # Safety: ensure every exercise keeps id/images unchanged.
    with open(args.in_path, "r", encoding="utf-8") as f:
        orig = json.load(f)

    if isinstance(orig, list) and isinstance(data, list):
        for o, n in zip(orig, data):
            if isinstance(o, dict) and isinstance(n, dict):
                if o.get("id") != n.get("id"):
                    raise RuntimeError("Invariant violated: exercise id changed")
                if o.get("images") != n.get("images"):
                    raise RuntimeError("Invariant violated: exercise images changed")

    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Wrote {args.out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
