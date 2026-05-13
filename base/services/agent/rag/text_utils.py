from __future__ import annotations

import ast
import json
import re
import unicodedata
from typing import Any, Dict, List


_TERM_RE = re.compile(r"[a-z0-9]+|[\u1780-\u17FF]+", re.IGNORECASE)
_KHMER_RE = re.compile(r"[\u1780-\u17FF]+")


def normalize_search_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text or "").lower()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def extract_overlap_terms(text: str) -> List[str]:
    normalized = normalize_search_text(text)
    return [match.group(0) for match in _TERM_RE.finditer(normalized)]


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def extract_significant_terms(text: str) -> List[str]:
    significant: List[str] = []
    for term in extract_overlap_terms(text):
        if _KHMER_RE.fullmatch(term):
            if len(term) >= 2:
                significant.append(term)
        elif len(term) >= 3 or term.isdigit():
            significant.append(term)
    return _dedupe_preserve_order(significant)


def compute_overlap_score(query: str, text: str) -> float:
    query_terms = set(extract_overlap_terms(query))
    text_terms = set(extract_overlap_terms(text))
    token_overlap = (
        len(query_terms & text_terms) / (len(query_terms) + 1e-9)
        if query_terms
        else 0.0
    )

    normalized_text = normalize_search_text(text)
    significant_terms = extract_significant_terms(query)
    substring_overlap = (
        sum(1 for term in significant_terms if term in normalized_text)
        / len(significant_terms)
        if significant_terms
        else token_overlap
    )

    overlap = 0.40 * token_overlap + 0.60 * substring_overlap
    return round(min(max(overlap, 0.0), 1.0), 6)


def extract_json_object(raw: str) -> Dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*", "", (raw or "").strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    candidates = [cleaned]
    balanced = _extract_balanced_object(cleaned)
    if balanced and balanced != cleaned:
        candidates.append(balanced)

    for candidate in candidates:
        parsed = _parse_json_candidate(candidate)
        if isinstance(parsed, dict):
            return parsed

    raise ValueError(f"No valid JSON object found in planner response: {cleaned[:300]}")


def _extract_balanced_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    return None


def _parse_json_candidate(candidate: str) -> Dict[str, Any] | None:
    if not candidate:
        return None

    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass

    pythonish = re.sub(r"\btrue\b", "True", candidate, flags=re.IGNORECASE)
    pythonish = re.sub(r"\bfalse\b", "False", pythonish, flags=re.IGNORECASE)
    pythonish = re.sub(r"\bnull\b", "None", pythonish, flags=re.IGNORECASE)

    try:
        parsed = ast.literal_eval(pythonish)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None
