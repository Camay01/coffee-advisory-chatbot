"""
input_parser.py — Extract structured data from freeform user text.

PERFORMANCE REWRITE:
  All LLM calls for name / location / farm-size / soil extraction have been
  replaced with deterministic regex + lookup tables.
  LLM is only invoked for soil values as a last-resort fallback when regex
  produces nothing AND the message clearly contains soil data.

Covers:
  - Profile fields: name, location, farm size, crop
  - Soil values from text (regex-first, LLM fallback)
  - Utility detectors: is_question, contains_soil_data, contains_profile_info
"""

from __future__ import annotations

import ast
import re
import unicodedata

from config import SOIL_PARAMS, KNOWN_ZONES

def _llm_call(system: str, user: str, num_predict: int = 100) -> str:
    from .llm_client import llm_call
    return llm_call(system=system, user=user, num_predict=num_predict)

# Verbose patterns: "organic carbon is 0.8", "available phosphorus: 12"
_VERBOSE_SOIL_RE = re.compile(
    r'\b(pH|organic\s+carbon|available\s+(?:phosphorus|nitrogen|potassium|zinc|boron)'
    r'|nitrogen|phosphorus|potassium|zinc|boron)\b[^\.]{0,30}?\d',
    re.IGNORECASE,
)

# Terse patterns: "pH 5.5", "N: 280", "Zn 0.6"
_TERSE_SOIL_RE = re.compile(
    r'(?<!\w)(pH|OC|Zn|[NPKB])(?!\w)[^.\n]{0,25}?(?:is|:|\s)\s*\d+\.?\d*',
    re.IGNORECASE,
)


def contains_soil_data(text: str) -> bool:
    """Returns True if the message contains explicit numeric soil parameters."""
    return bool(_VERBOSE_SOIL_RE.search(text) or _TERSE_SOIL_RE.search(text))

_PARAM_ALIASES: dict[str, str] = {
    "ph": "pH", "oc": "OC",
    "n": "N", "p": "P", "k": "K", "zn": "Zn", "b": "B",
}
_ALL_PARAM_KEYS: set[str] = {p for p, _ in SOIL_PARAMS}

# Covers both terse ("pH 5.2") and verbose ("organic carbon is 0.8%")
_SOIL_PARSE_RE = re.compile(
    r'(?:'
    # terse: key then value
    r'\b(pH|OC|Zn|[NPKB])\b[^.\n,]{0,15}?(\d+\.?\d*)'
    r'|'
    # verbose aliases
    r'\b(organic\s+carbon|organic\s+matter|available\s+nitrogen|available\s+phosphorus'
    r'|available\s+potassium|available\s+zinc|available\s+boron'
    r'|nitrogen|phosphorus|potassium|zinc|boron)\b[^.\n,]{0,30}?(\d+\.?\d*)'
    r')',
    re.IGNORECASE,
)

_VERBOSE_ALIAS_MAP: dict[str, str] = {
    "organic carbon": "OC", "organic matter": "OC",
    "available nitrogen": "N", "nitrogen": "N",
    "available phosphorus": "P", "phosphorus": "P",
    "available potassium": "K", "potassium": "K",
    "available zinc": "Zn", "zinc": "Zn",
    "available boron": "B", "boron": "B",
}


def parse_soil_input(raw: str) -> dict:
    """
    Extract soil parameter values from freeform text.

    Stage 1 — regex (covers terse AND verbose natural-language input).
    Stage 2 — LLM fallback only when regex finds nothing AND the message
               clearly looks like it contains soil data.

    Returns dict of ONLY explicitly mentioned parameters.
    """
    result: dict[str, float] = {}

    for m in _SOIL_PARSE_RE.finditer(raw):
        if m.group(1):                          # terse match
            key_raw, val_str = m.group(1), m.group(2)
            key = _PARAM_ALIASES.get(key_raw.lower(), key_raw)
        else:                                   # verbose match
            key_raw, val_str = m.group(3), m.group(4)
            key = _VERBOSE_ALIAS_MAP.get(key_raw.strip().lower())

        if key and key in _ALL_PARAM_KEYS and val_str:
            try:
                result[key] = float(val_str)
            except ValueError:
                pass

    if result:
        return result

    # ── LLM fallback (only when regex misses something obvious) ──────────────
    if not contains_soil_data(raw):
        return {}

    llm_result = _llm_call(
        system=(
            "Extract soil test values from the user message.\n"
            "Return ONLY a valid Python dict with keys from: pH, OC, N, P, K, Zn, B.\n"
            "Include ONLY keys explicitly mentioned. Values must be numeric.\n"
            "Map common phrases:\n"
            "  'organic carbon', 'OC', 'OC%' → OC\n"
            "  'nitrogen', 'available N', 'N kg/ha' → N\n"
            "  'phosphorus', 'available P' → P\n"
            "  'potassium', 'available K' → K\n"
            "  'zinc', 'Zn' → Zn\n"
            "  'boron', 'B' → B\n\n"
            "Examples:\n"
            "  'My pH is 4.9'               → {'pH': 4.9}\n"
            "  'pH 5.5, N 300'              → {'pH': 5.5, 'N': 300}\n"
            "  'I grow Arabica in Kodagu'   → {}\n\n"
            "If nothing mentioned, return exactly: {}\n"
            "CRITICAL: Do NOT infer or guess values not stated."
        ),
        user=raw,
        num_predict=100,
    )
    try:
        clean = re.sub(r"```.*?```", "", llm_result, flags=re.DOTALL).strip()
        parsed = ast.literal_eval(clean)
        if not isinstance(parsed, dict):
            return {}
        return {k: float(v) for k, v in parsed.items() if k in _ALL_PARAM_KEYS}
    except Exception:
        return {}


def try_extract_soil_early(prompt: str, user_data: dict) -> None:
    """
    Extract and store soil values from any message into user_data["measured_soil"].
    Values are never lost even if captured mid-onboarding.
    """
    if contains_soil_data(prompt):
        soil_vals = parse_soil_input(prompt)
        if soil_vals:
            if "measured_soil" not in user_data:
                user_data["measured_soil"] = {}
            user_data["measured_soil"].update(soil_vals)
            if "soil_raw" not in user_data:
                user_data["soil_raw"] = prompt

_KNOWN_NON_NAMES: set[str] = {
    "idukki", "wayanad", "kodagu", "hassan", "chikmagalur", "coorg",
    "arabica", "robusta", "coffee", "chandragiri", "kerala", "karnataka",
    "skip", "yes", "no", "ok", "okay", "sure", "hello", "hi",
    "ph", "oc", "nitrogen", "phosphorus", "potassium", "zinc", "boron",
    "farm", "crop", "soil", "sand", "clay", "well", "fine", "good",
    "help", "need", "want", "have", "grow", "plant", "field", "land",
    "rain", "water", "tree", "leaf", "root", "stem", "seed", "fruit",
}

# "my name is X", "I am X", "call me X", "this is X"
_NAME_INTRO_RE = re.compile(
    r'(?:my\s+name\s+is|i\s+am|i\'m|call\s+me|this\s+is)\s+([A-Za-z]{2,30})',
    re.IGNORECASE,
)


def extract_name(raw: str) -> str | None:
    """
    Return person's first name, or None if none detected.
    Pure regex — no LLM.
    """
    stripped = raw.strip()

    # Single-word reply that looks like a name
    if (
        len(stripped.split()) == 1
        and stripped.isalpha()
        and 2 <= len(stripped) <= 30
        and stripped.lower() not in _KNOWN_NON_NAMES
        and not contains_soil_data(stripped)
    ):
        return stripped.title()

    # "My name is Rajan" / "I am Anu" patterns
    m = _NAME_INTRO_RE.search(raw)
    if m:
        candidate = m.group(1).strip()
        if candidate.lower() not in _KNOWN_NON_NAMES:
            return candidate.title()

    return None

# Broader Indian district/state terms that might appear in messages
_LOCATION_RE = re.compile(
    r'\b(?:in|from|at|near|around|located\s+in|my\s+farm\s+(?:is\s+)?in)\s+'
    r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})',
    re.IGNORECASE,
)

# Common suffixes that signal a place name rather than a personal name
_PLACE_SUFFIXES = re.compile(
    r'\b\w+(?:pur|abad|nagar|pura|patti|pally|ganj|pet|halli|pura|kere|'
    r'giri|ghat|mane|wadi|konam|puram)\b',
    re.IGNORECASE,
)

_FICTIONAL: set[str] = {"mars", "narnia", "mordor", "wakanda", "atlantis", "hogwarts", "pandora"}


def extract_location(raw: str) -> str | None:
    """Return clean zone/district name, or None. No LLM."""
    lower = raw.lower()

    # Fast path: known zone list (case-insensitive)
    for zone in KNOWN_ZONES:
        if zone in lower:
            return zone.title()

    # Pattern: "my farm is in Belur" / "from West Hassan"
    m = _LOCATION_RE.search(raw)
    if m:
        candidate = m.group(1).strip()
        if (
            candidate.lower() not in _KNOWN_NON_NAMES
            and candidate.lower() not in _FICTIONAL
            and len(candidate) <= 60
        ):
            return candidate.title()

    # Place-name suffix heuristic ("Mudigere", "Virajpet", "Somwarpet" …)
    m2 = _PLACE_SUFFIXES.search(raw)
    if m2:
        candidate = m2.group(0).strip()
        if candidate.lower() not in _FICTIONAL:
            return candidate.title()

    return None

_FARM_SIZE_RE = re.compile(
    r'(\d+\.?\d*)\s*'
    r'(?:ha(?:ctare)?s?|acres?(?:\s*×\s*0\.4)?)',
    re.IGNORECASE,
)

# Also handle bare numbers when the question was specifically about farm size
_BARE_NUMBER_RE = re.compile(r'\b(\d+\.?\d*)\b')


def extract_farm_size(raw: str) -> str | None:
    """Return numeric farm size string, or None. No LLM."""
    m = _FARM_SIZE_RE.search(raw)
    if m:
        val = m.group(1)
        # If unit was acres, convert to hectares (1 acre ≈ 0.4047 ha)
        if re.search(r'acre', m.group(0), re.IGNORECASE):
            try:
                val = str(round(float(val) * 0.4047, 2))
            except ValueError:
                pass
        return val

    # Bare number heuristic — only if "ha" or "hectare" appears anywhere nearby
    if re.search(r'\bhect(?:are)?s?\b|\bha\b', raw, re.IGNORECASE):
        m2 = _BARE_NUMBER_RE.search(raw)
        if m2:
            return m2.group(1)

    return None

def extract_crop(raw: str) -> str | None:
    """Return coffee crop category if mentioned, or None."""
    lower = raw.lower()
    if "arabica" in lower:
        return "Arabica"
    if "robusta" in lower:
        return "Robusta"
    if "coffee" in lower:
        return "coffee"
    return None


def detect_non_coffee_crop(raw: str) -> str | None:
    """
    Return explicitly non-coffee crop name if detected, else None.

    Uses word-boundary regex matching to avoid substring false-positives
    such as "tea" inside "Estate", "pepper" inside "peppercorn", or
    "corn" inside "acorn".  Multi-word crops (e.g. "oil palm") are
    matched as a whole phrase with boundaries on the outer edges only.
    """
    lower = raw.lower()
    non_coffee = [
        "tea", "cocoa", "cacao", "pepper", "cardamom", "vanilla",
        "rubber", "coconut", "banana", "rice", "wheat", "maize",
        "cotton", "sugarcane", "turmeric", "ginger", "clove", "nutmeg",
        "vegetable", "fruit", "arecanut", "betel", "tobacco", "oil palm", "corn", "cabbage",
    ]
    for crop in non_coffee:
        pattern = r'\b' + re.escape(crop) + r'\b'
        if re.search(pattern, lower):
            return crop.title()
    return None

def prefill_profile_from_message(prompt: str, user_data: dict) -> None:
    """
    Extract all profile signals from a single rich message and pre-fill
    user_data so onboarding never re-asks known fields.
    """
    if not user_data.get("location"):
        loc = extract_location(prompt)
        if loc:
            user_data["location"] = loc
    if not user_data.get("farm_size"):
        fs = extract_farm_size(prompt)
        if fs:
            user_data["farm_size"] = fs
    if not user_data.get("crop"):
        crop = extract_crop(prompt)
        if crop:
            user_data["crop"] = crop
    try_extract_soil_early(prompt, user_data)

def is_question(text: str) -> bool:
    """Detect if a message contains an advisory question."""
    lower = text.lower().strip()
    if "?" in text:
        return True
    starters = [
        "what", "how", "why", "when", "where", "which",
        "is ", "are ", "does ", "do ", "can ", "could ",
        "should ", "tell me", "explain", "level", "interpret",
        "what's", "advise", "advice",
    ]
    return any(lower.startswith(w) or f" {w}" in lower for w in starters)


def contains_profile_info(text: str) -> bool:
    """
    Detect if the message contains ownership signals (the user describing THEIR farm).
    Bare crop or location words in a question do NOT qualify.
    """
    lower = text.lower()
    ownership_signals = [
        "i am from", "i'm from", "i grow", "my farm", "my crop",
        "my estate", "my plantation", "my field", "my soil",
        "i cultivate", "i farm", "i run a farm", "i manage", "i own a farm",
        "we grow", "we cultivate", "we farm",
        "located in", "estate", "plantation",
        "hectare", " ha ", " ha,", " ha.",
    ]
    return any(signal in lower for signal in ownership_signals)