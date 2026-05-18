"""
input_parser.py — Extract structured data from freeform user text.

Covers:
  - Profile fields: name, location, farm size, crop
  - Soil values from text (regex-first, LLM fallback)
  - Utility detectors: is_question, contains_soil_data, contains_profile_info
"""

from __future__ import annotations

import ast
import re

from config import SOIL_PARAMS, KNOWN_ZONES
from units.llm_client import llm_call

def contains_soil_data(text: str) -> bool:
    """
    Returns True if the message contains explicit numeric soil parameters.
    This is a HARD signal — always checked before step gates.
    """
    verbose = re.compile(
        r'\b(pH|organic\s+carbon|available\s+(?:phosphorus|nitrogen|potassium|zinc|boron)'
        r'|nitrogen|phosphorus|potassium|zinc|boron)\b[^\.]{0,30}?\d',
        re.IGNORECASE,
    )
    if verbose.search(text):
        return True
    terse = re.compile(
        r'(?<!\w)(pH|OC|Zn|[NPKB])(?!\w)[^.\n]{0,25}?(?:is|:|\s)\s*\d+\.?\d*',
        re.IGNORECASE,
    )
    return bool(terse.search(text))


def parse_soil_input(raw: str) -> dict:
    """
    Extract soil parameter values from freeform text.
    Stage 1: regex for terse structured input ("P 8", "pH 5.2, N 240").
    Stage 2: LLM for natural-language fallback ("my pH is 4.9").
    Returns dict of ONLY explicitly mentioned parameters.
    """
    PARAM_ALIASES = {
        "ph": "pH", "oc": "OC",
        "n": "N", "p": "P", "k": "K", "zn": "Zn", "b": "B",
    }
    ALL_PARAM_KEYS = {p for p, _ in SOIL_PARAMS}

    terse_re = re.compile(
        r'\b(pH|OC|Zn|[NPKB])\b[^.\n,]{0,15}?(\d+\.?\d*)',
        re.IGNORECASE,
    )
    regex_vals = {}
    for m in terse_re.finditer(raw):
        key = PARAM_ALIASES.get(m.group(1).lower(), m.group(1))
        if key in ALL_PARAM_KEYS:
            try:
                regex_vals[key] = float(m.group(2))
            except ValueError:
                pass
    if regex_vals:
        return regex_vals

    # LLM fallback for natural-language input
    result = llm_call(
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
        clean = re.sub(r"```.*?```", "", result, flags=re.DOTALL).strip()
        parsed = ast.literal_eval(clean)
        if not isinstance(parsed, dict):
            return {}
        return {k: float(v) for k, v in parsed.items() if k in ALL_PARAM_KEYS}
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

_KNOWN_NON_NAMES = {
    "idukki", "wayanad", "kodagu", "hassan", "chikmagalur", "coorg",
    "arabica", "robusta", "coffee", "chandragiri", "kerala", "karnataka",
    "skip", "yes", "no", "ok", "okay", "sure", "hello", "hi",
    "ph", "oc", "nitrogen", "phosphorus", "potassium", "zinc", "boron",
    "farm", "crop", "soil", "sand", "clay", "well", "fine", "good",
    "help", "need", "want", "have", "grow", "plant", "field", "land",
    "rain", "water", "tree", "leaf", "root", "stem", "seed", "fruit",
}


def extract_name(raw: str) -> str | None:
    """Return person's first name, or None if none detected."""
    stripped = raw.strip()
    # Pre-LLM heuristic: single alphabetic word, 2-30 chars, not a known non-name
    if (
        len(stripped.split()) == 1
        and stripped.isalpha()
        and 2 <= len(stripped) <= 30
        and stripped.lower() not in _KNOWN_NON_NAMES
        and not contains_soil_data(stripped)
    ):
        return stripped.title()

    result = llm_call(
        system=(
            "You extract a PERSONAL (human given/first) name from a message.\n"
            "The user was asked for their name.\n"
            "Return ONLY the personal name — no punctuation, no explanation.\n\n"
            "CRITICAL RULES:\n"
            "- Location names, districts, villages, states are NOT personal names.\n"
            "- Crop names (Arabica, Robusta, coffee) are NOT personal names.\n"
            "- Soil values (pH, N, P, K) are NOT personal names.\n"
            "- Short personal names like 'Anu', 'Raj', 'Uma', 'Biju' ARE valid.\n"
            "- If the input contains ONLY location, crop, or soil data, return: NONE\n\n"
            "Examples:\n"
            "  'anu' → Anu\n"
            "  'I am from Idukki and grow Arabica' → NONE\n"
            "  'My name is Rajan from Kodagu' → Rajan\n"
            "  'good morning' → NONE"
        ),
        user=raw,
        num_predict=15,
    )
    result = result.strip()
    if not result or result.upper() == "NONE" or len(result.split()) > 5:
        return None
    if result.lower() in _KNOWN_NON_NAMES:
        return None
    return result.title()


def extract_location(raw: str) -> str | None:
    """Return clean zone/district name, or None."""
    lower = raw.lower()
    for zone in KNOWN_ZONES:
        if zone in lower:
            return zone.title()

    result = llm_call(
        system=(
            "Extract only the farm location name (village, zone, district, or region) "
            "from the user message. Return ONLY the location name — no extra words.\n"
            "The location must be a real geographical place where coffee can be grown.\n"
            "If no real location is mentioned, return exactly: NONE.\n"
            "Fictional locations (Mars, Narnia) → NONE\n"
            "Example: 'I have a field in West Belur in Hassan district.' → West Belur, Hassan\n"
            "Example: 'pH 5.5, N 280' → NONE"
        ),
        user=raw,
        num_predict=20,
    )
    result = result.strip()
    if not result or result.upper() == "NONE" or len(result) > 80:
        return None
    fictional = {"mars", "narnia", "mordor", "wakanda", "atlantis", "hogwarts", "pandora"}
    if result.lower() in fictional:
        return None
    return result


def extract_farm_size(raw: str) -> str | None:
    """Return numeric farm size in hectares, or None."""
    result = llm_call(
        system=(
            "Extract the farm size in hectares from the user message. "
            "Return ONLY the numeric value e.g. '2' or '3.5'. "
            "If no farm size is mentioned, return exactly: NONE"
        ),
        user=raw,
        num_predict=10,
    )
    result = result.strip()
    if not result or result.upper() == "NONE":
        return None
    try:
        float(result.replace("ha", "").strip())
        return result.replace("ha", "").strip()
    except ValueError:
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
    """Return explicitly non-coffee crop name if detected, else None."""
    lower = raw.lower()
    non_coffee = [
        "tea", "cocoa", "cacao", "pepper", "cardamom", "vanilla",
        "rubber", "coconut", "banana", "rice", "wheat", "maize",
        "cotton", "sugarcane", "turmeric", "ginger", "clove", "nutmeg",
        "vegetable", "fruit", "arecanut", "betel", "tobacco", "oil palm","corn","cabbage"
    ]
    for crop in non_coffee:
        if crop in lower:
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
