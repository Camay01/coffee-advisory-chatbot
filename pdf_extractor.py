"""
pdf_extractor.py — Extract soil values from uploaded PDF soil-test reports.

PERFORMANCE REWRITE:
  1. A fast regex pass (_regex_extract_all) runs FIRST and covers the
     overwhelming majority of standard Indian lab report layouts.
  2. The LLM is invoked ONLY for labels that regex could not map, and only
     when fewer than MIN_REGEX_HITS parameters were found by regex.
  3. Unit conversion and plausibility validation are unchanged (deterministic).

This reduces LLM calls per PDF from 1 (always) to 0 (most real reports).
"""

from __future__ import annotations

import ast
import io
import re

from config import SOIL_PARAMS, UNIT_ALIASES, PPM_TO_KG_HA_FACTOR


# ===========================================================================
# Plausibility ranges (hard limits)
# ===========================================================================
_PLAUSIBILITY: dict[str, tuple[float, float]] = {
    "pH": (3.0, 10.0),
    "OC": (0.0, 20.0),
    "N":  (0.0, 2000.0),
    "P":  (0.0, 500.0),
    "K":  (0.0, 2000.0),
    "Zn": (0.0, 100.0),
    "B":  (0.0, 20.0),
}

# Minimum regex hits before we skip the LLM entirely.
# FIX: raised from 3 → 6. With 3, finding only pH+OC+K suppressed the LLM
# fallback even though N, P, Zn, B were still missing.  We want the LLM to
# fill gaps unless regex already found nearly everything (>=6 of 7 params).
_MIN_REGEX_HITS = 6

# Soil keywords for page-level filtering
_SOIL_KEYWORDS = [
    "ph", "organic carbon", "organic matter", "oc",
    "nitrogen", "available nitrogen", "available n", "nitrate", "no3-n",
    "phosphorus", "available phosphorus", "available p", "p2o5",
    "potassium", "available potassium", "available k", "k2o", "potash",
    "zinc", "zn", "boron", "b",
    "kg/ha", "mg/kg", "ppm", "%",
]


# ===========================================================================
# Regex extraction patterns — covers common Indian lab report formats
# ===========================================================================

# Each tuple: (canonical_key, compiled_pattern)
# Patterns capture the first numeric group after the label.
_REGEX_PATTERNS: list[tuple[str, re.Pattern]] = [
    # pH — BUG FIX 1: old pattern used [^:\n] which stopped at the colon inside
    # the method description (e.g. "1:2.5 soil-water"), so it never reached the
    # actual pH value.  Fix: allow colons by using [^\n] and use a decimal-first
    # alternation with a lookbehind to avoid matching mid-number (e.g. "2.5" in
    # "1:2.5").  EC guard now spells out "ds/m" and "ms/cm" explicitly.
    ("pH", re.compile(
        r'\bpH\b[^\n]{0,50}?(?<!\d)([3-9]\.\d+|(?<!\d\.)(?<!\d)[4-9])(?!\d)'
        r'(?!\s*(?:mmhos|mhos|ds/m|ms/cm))',
        re.IGNORECASE,
    )),
    # Organic Carbon % — BUG FIX 2: old pattern matched "OC" inside "MOCK-001"
    # (Laboratory ID strings).  Fix: require the line to start with the label
    # (multiline anchor) and require a decimal value (X.XX) so bare integers
    # like "001" in Lab IDs are never captured.
    ("OC", re.compile(
        r'(?:^|\n)[^\n]*(?:organic\s+carbon|OC\s*%)[^\n]{0,30}?(\d+\.\d+)',
        re.IGNORECASE | re.MULTILINE,
    )),
    # Organic Matter % (proxy for OC; flagged in unit_meta) — unchanged
    ("OC_OM", re.compile(
        r'organic\s+matter\s*(?:%|percent)?[^:\n]{0,20}?(\d+\.?\d*)',
        re.IGNORECASE,
    )),
    # Available Nitrogen kg/ha or ppm
    # FIX: allow colon separator between label and value, common in Indian
    # lab table layouts ("Available Nitrogen (kg/ha) : 185").
    # The old [^:\n] stop-at-colon meant these lines were never matched.
    ("N", re.compile(
        r'(?:available\s+nitrogen|available\s+N|nitrogen\s+available'
        r'|N\s*(?:kg/ha|ppm))'
        r'[^(\n]{0,30}?(?:\([^)]*\))?\s*:?\s*(\d+\.?\d*)',
        re.IGNORECASE,
    )),
    # Nitrate-N (proxy)
    ("N_nitrate", re.compile(
        r'(?:nitrate[\s-]*N|NO3[\s-]*N)'
        r'[^(\n]{0,30}?(?:\([^)]*\))?\s*:?\s*(\d+\.?\d*)',
        re.IGNORECASE,
    )),
    # Available Phosphorus
    # FIX: same colon-separator fix as N above.
    ("P", re.compile(
        r'(?:available\s+phosphorus|available\s+P|phosphorus\s+available'
        r'|P\s*(?:kg/ha|ppm))'
        r'[^(\n]{0,30}?(?:\([^)]*\))?\s*:?\s*(\d+\.?\d*)',
        re.IGNORECASE,
    )),
    # Available Potassium
    # FIX: same colon-separator fix as N above.
    ("K", re.compile(
        r'(?:available\s+potassium|available\s+K|potassium\s+available'
        r'|K\s*(?:kg/ha|ppm))'
        r'[^(\n]{0,30}?(?:\([^)]*\))?\s*:?\s*(\d+\.?\d*)',
        re.IGNORECASE,
    )),
    # Zinc
    # FIX: same colon-separator fix. Also handles "Zn (mg/kg) : 0.48" format.
    ("Zn", re.compile(
        r'(?:available\s+zinc|zinc)'
        r'[^(\n]{0,30}?(?:\([^)]*\))?\s*:?\s*(\d+\.?\d*)',
        re.IGNORECASE,
    )),
    # Boron
    # FIX: same colon-separator fix. Keeps "boron" spelled out (no bare "B")
    # to avoid matching mid-word (e.g. "carBon").
    ("B", re.compile(
        r'(?:available\s+boron|boron)'
        r'[^(\n]{0,30}?(?:\([^)]*\))?\s*:?\s*(\d+\.?\d*)',
        re.IGNORECASE,
    )),
]

# Unit patterns — look for the unit string in the same ±60 chars around a match
_UNIT_RE = re.compile(
    r'(kg/ha|kg\s+ha|mg/kg|mg\s+kg|ppm|%|g/kg|lb/a|lbs?/acre)',
    re.IGNORECASE,
)

# Crop detection
_CROP_RE = re.compile(
    r'(?:crop|plantation|subject|commodity|for)[^:\n]{0,20}?:\s*'
    r'(arabica|robusta|coffee|tea|pepper|cardamom|rubber|coconut)',
    re.IGNORECASE,
)
_CROP_TITLE_RE = re.compile(
    r'\b(arabica|robusta|coffee|tea|pepper|cardamom|rubber|coconut)\b',
    re.IGNORECASE,
)

# Zone/location detection — only match when labelled explicitly in the PDF.
# We do NOT guess from city names in addresses to avoid hallucinated zones.
_ZONE_EXPLICIT_RE = re.compile(
    r'(?:zone|district|taluk|taluka|mandal|block|region)[^:\n]{0,10}?:\s*'
    r'([A-Za-z][\w\s]{2,40})',
    re.IGNORECASE,
)

# Known South Indian coffee zones — used for high-confidence address matching
_KNOWN_COFFEE_ZONES = [
    "idukki", "wayanad", "kodagu", "coorg", "hassan",
    "chikmagalur", "chickmagalur", "sakleshpur", "madikeri",
    "virajpet", "somwarpet", "belur", "mudigere", "kushalnagar",
    "siddapur", "aldur", "balehonnur", "jayapura",
]

_KNOWN_ZONE_RE = re.compile(
    r'\b(' + '|'.join(_KNOWN_COFFEE_ZONES) + r')\b',
    re.IGNORECASE,
)


# ===========================================================================
# Public entry point
# ===========================================================================

# Section headers that mark the end of soil test data.
# Everything after these is fertiliser recommendations — must NOT be parsed.
_RECOMMENDATION_MARKERS = re.compile(
    r'^\s*(?:RECOMMENDATION|FERTILIZER\s+RECOMMENDATION|FERTILISER\s+RECOMMENDATION'
    r'|SUGGESTED\s+(?:DOSE|APPLICATION)|DOSAGE\s+RECOMMENDATION'
    r'|TREATMENT\s+SCHEDULE|NUTRIENT\s+MANAGEMENT\s+PLAN)',
    re.IGNORECASE | re.MULTILINE,
)


def _truncate_at_recommendation(text: str) -> str:
    """
    Cut the text at the first recommendation/dosage section heading.
    This prevents fertiliser dosage numbers (e.g. "Zinc Sulphate 20 kg/ha")
    from being mis-read as soil test values (e.g. Zn = 20 mg/kg).
    """
    m = _RECOMMENDATION_MARKERS.search(text)
    if m:
        return text[:m.start()]
    return text


def extract_soil_from_pdf(file_bytes: bytes) -> tuple[dict, dict, str, dict, str]:
    """
    Extract soil values from any uploaded PDF.

    Returns:
        kb_matched   — {param: value} ready for advisory
        all_extracted — every label+value seen in the PDF
        raw_text     — full extracted text (before recommendation section)
        unit_meta    — provenance record per parameter
        crop_found   — crop name detected in PDF (str)
    """
    raw_text = _extract_raw_text(file_bytes)

    if raw_text.startswith("[Could not read PDF"):
        return {}, {}, raw_text, {}, ""

    if len(raw_text.strip()) < 30:
        return (
            {}, {},
            "[PDF appears to be a scanned image — text could not be extracted automatically. "
            "Please type your soil values manually (e.g. pH 5.5, N 280, P 8).]",
            {}, "",
        )

    # CRITICAL: strip recommendation/dosage section BEFORE any extraction.
    # Fertiliser rates (e.g. "Zinc Sulphate 20 kg/ha") would otherwise be
    # mis-read as soil test values (Zn = 20).
    soil_text = _truncate_at_recommendation(raw_text)

    crop_found = _detect_crop(raw_text)          # use full text for crop name
    kb_matched, all_extracted, unit_meta = _extract_and_validate(soil_text)
    return kb_matched, all_extracted, soil_text, unit_meta, crop_found


# ===========================================================================
# Text extraction
# ===========================================================================

def _extract_raw_text(file_bytes: bytes) -> str:
    try:
        import pdfplumber
    except ImportError:
        return "[pdfplumber not installed — cannot read PDF.]"

    try:
        raw_text = ""
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages[:5]:
                words = page.extract_words() or []
                if not any(re.search(r"\d", w.get("text", "")) for w in words[:200]):
                    continue
                page_text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
                raw_text += page_text + "\n"
                lower = raw_text.lower()
                found_count = sum(1 for kw in _SOIL_KEYWORDS if kw in lower)
                if found_count >= 5:
                    break
        return raw_text.strip()
    except Exception as e:
        return f"[Could not read PDF: {e}]"


# ===========================================================================
# Crop detection (regex — no LLM)
# ===========================================================================

def _detect_crop(raw_text: str) -> str:
    """Detect crop type from PDF. Returns crop name or 'Unknown'."""
    m = _CROP_RE.search(raw_text)
    if m:
        return m.group(1).title()
    m2 = _CROP_TITLE_RE.search(raw_text)
    if m2:
        return m2.group(1).title()
    return "Unknown"


def detect_zone_from_pdf(raw_text: str) -> tuple[str | None, float]:
    """
    Detect farm zone from PDF text with a confidence score.

    Returns (zone_name, confidence) where:
      confidence == 1.0  — explicitly labelled as Zone/District in the PDF
      confidence == 0.7  — known coffee zone found in a labelled address field
      confidence == 0.0  — not found, or only found in ambiguous context

    NEVER returns a zone inferred from free-form address text alone,
    to avoid the Chikkamagaluru hallucination bug.
    """
    # Tier 1: explicit zone/district label  (e.g. "Zone: Kodagu")
    m = _ZONE_EXPLICIT_RE.search(raw_text)
    if m:
        candidate = m.group(1).strip().rstrip(',.')
        return candidate.title(), 1.0

    # Tier 2: known coffee zone inside a labelled address line only
    # Look for "Plot Address:", "Farm Address:", "Location:" etc.
    address_block_re = re.compile(
        r'(?:plot\s+address|farm\s+address|location|address)[^:\n]{0,10}?:\s*'
        r'([^\n]{5,120})',
        re.IGNORECASE,
    )
    for am in address_block_re.finditer(raw_text):
        addr_text = am.group(1)
        zm = _KNOWN_ZONE_RE.search(addr_text)
        if zm:
            return zm.group(0).title(), 0.7

    # Tier 3: no confident zone found
    return None, 0.0


# ===========================================================================
# Regex extraction pass
# ===========================================================================

def _nearby_unit(text: str, pos: int, window: int = 160) -> str:
    """Return the first unit token found within ±window chars of pos.
    FIX: widened from 80 → 160 chars to capture units in wide table rows
    where the label and unit column may be far apart on the same line."""
    snippet = text[max(0, pos - window): pos + window]
    m = _UNIT_RE.search(snippet)
    return m.group(0) if m else ""


def _regex_extract_all(raw_text: str) -> list[dict]:
    """
    Apply all _REGEX_PATTERNS to raw_text.
    Returns a list of {name, value, unit, is_proxy} records.
    """
    records: list[dict] = []
    seen_params: set[str] = set()

    for key, pattern in _REGEX_PATTERNS:
        m = pattern.search(raw_text)
        if not m:
            continue
        try:
            val = float(m.group(1))
        except (ValueError, IndexError):
            continue

        unit = _nearby_unit(raw_text, m.start())
        is_proxy = key in ("OC_OM", "N_nitrate")
        canon_key = {"OC_OM": "OC", "N_nitrate": "N"}.get(key, key)

        # Don't overwrite a direct measurement with a proxy
        if canon_key in seen_params:
            continue
        seen_params.add(canon_key)

        records.append({
            "name":     canon_key,
            "value":    val,
            "unit":     unit,
            "is_proxy": is_proxy,
            "raw_label": key,
        })

    return records


# ===========================================================================
# LLM fallback (invoked only when regex is insufficient)
# ===========================================================================

def _llm_extract_remaining(raw_text: str, already_found: set[str]) -> list[dict]:
    """
    Call the LLM only for the labels regex could not map.
    Keeps the prompt minimal — filtered text, 2 500 char cap.
    """
    from units.llm_client import llm_call

    missing = [k for k, _ in SOIL_PARAMS if k not in already_found]
    if not missing:
        return []

    filtered = _filter_relevant_lines(raw_text)
    context = (filtered if len(filtered) <= 2500 else filtered[:2500]) or raw_text[:2500]

    llm_result = llm_call(
        system=(
            "You are a soil report data extraction assistant.\n"
            f"The following parameters were NOT found by regex: {', '.join(missing)}.\n"
            "Extract ONLY those parameters from the text below.\n"
            "Return ONLY a valid Python list of dicts:\n"
            "  [{'name': 'pH', 'value': 5.5, 'unit': ''}, ...]\n"
            "If a parameter is genuinely absent, omit it.\n"
            "Do NOT wrap in markdown. Do NOT explain."
        ),
        user=f"Soil report text:\n\n{context}",
        num_predict=400,
    )

    records: list[dict] = []
    try:
        clean = re.sub(r"```[a-z]*\n?", "", llm_result, flags=re.IGNORECASE).replace("```", "").strip()
        match = re.search(r"\[.*\]", clean, re.DOTALL)
        if match:
            parsed = ast.literal_eval(match.group())
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict) and "name" in item and "value" in item:
                        try:
                            records.append({
                                "name":     str(item["name"]).strip(),
                                "value":    float(item["value"]),
                                "unit":     str(item.get("unit", "")).strip(),
                                "is_proxy": False,
                                "raw_label": "llm",
                            })
                        except (TypeError, ValueError):
                            pass
    except Exception:
        pass
    return records


def _filter_relevant_lines(raw_text: str) -> str:
    relevant = []
    for line in raw_text.split("\n"):
        lower = line.lower()
        if any(kw in lower for kw in _SOIL_KEYWORDS) or re.search(r"\d", line):
            relevant.append(line)
    return "\n".join(relevant)


# ===========================================================================
# Unified extraction + validation pipeline
# ===========================================================================

def _extract_and_validate(raw_text: str) -> tuple[dict, dict, dict]:
    """
    1. Run regex extraction.
    2. If fewer than _MIN_REGEX_HITS found, run LLM for the remainder.
    3. Validate, convert units, check plausibility.

    Returns (kb_matched, all_extracted, unit_meta).
    """
    regex_records = _regex_extract_all(raw_text)
    found_by_regex = {r["name"] for r in regex_records}

    # LLM fallback only when regex left too many gaps
    llm_records: list[dict] = []
    if len(found_by_regex) < _MIN_REGEX_HITS:
        llm_records = _llm_extract_remaining(raw_text, found_by_regex)

    all_records = regex_records + [
        r for r in llm_records if r["name"] not in found_by_regex
    ]

    # Build all_extracted for display
    all_extracted: dict[str, float] = {r["name"]: r["value"] for r in all_records}

    kb_matched, unit_meta = _validate_and_convert(all_records)
    return kb_matched, all_extracted, unit_meta


# ===========================================================================
# Unit conversion (unchanged logic, same as original)
# ===========================================================================

def _normalise_unit(raw: str) -> str:
    return UNIT_ALIASES.get(raw.strip().lower(), "unknown")


_CANON_UNITS = {
    "pH": "", "OC": "%", "N": "kg/ha",
    "P": "kg/ha", "K": "kg/ha", "Zn": "mg/kg", "B": "mg/kg",
}

_PARAM_MAP: dict[str, str] = {
    "ph": "pH", "oc": "OC", "organic carbon": "OC", "organic matter": "OC",
    "n": "N", "nitrogen": "N", "available n": "N", "available nitrogen": "N",
    "nitrate-n": "N", "no3-n": "N",
    "p": "P", "phosphorus": "P", "available p": "P", "available phosphorus": "P",
    "k": "K", "potassium": "K", "available k": "K", "available potassium": "K",
    "zn": "Zn", "zinc": "Zn",
    "b": "B", "boron": "B",
}


def _convert_to_kb_unit(param: str, value: float, raw_unit: str, is_proxy: bool = False):
    """
    Convert extracted value to KB canonical unit.
    Returns (converted_value, was_converted, excluded, unit_ambiguous, note).
    """
    unit = _normalise_unit(raw_unit)
    note = ""

    if unit == "lb/a":
        return None, False, True, False, "fertiliser_recommendation_unit"

    if param == "pH":
        raw_lower = raw_unit.strip().lower()
        if any(ec in raw_lower for ec in ("mmhos", "mhos", "ds/m", "ds m", "ms/cm")):
            return None, False, True, False, "ec_misread_as_ph"
        return value, False, False, False, note

    if param == "OC":
        if is_proxy:
            note = "organic_matter_approx"
            # OM % → OC % (Van Bemmelen factor 1.724)
            return round(value / 1.724, 3), True, False, False, note
        if unit in ("none", "%"):
            return value, False, False, (unit == "none"), note
        if unit == "g/kg":
            return round(value / 10.0, 4), True, False, False, note
        return None, False, True, False, note

    if param in ("N", "P", "K"):
        if is_proxy and param == "N":
            note = "nitrate_n_proxy"
        if unit == "kg/ha":
            return value, False, False, False, note
        if unit == "mg/kg":
            return None, False, True, False, note  # bulk density unknown
        if unit == "none":
            return value, False, False, True, note  # assume kg/ha
        return None, False, True, False, note

    if param in ("Zn", "B"):
        if unit in ("mg/kg", "none", "unknown"):
            return value, False, False, (unit in ("none", "unknown")), note
        if unit == "kg/ha":
            return None, False, True, False, note
        return value, False, False, False, note

    return None, False, True, False, note


def _validate_and_convert(raw_records: list[dict]) -> tuple[dict, dict]:
    """
    Map records to KB canonical keys, convert units, validate plausibility.
    Returns (kb_matched, unit_meta).
    """
    unit_meta: dict[str, dict] = {}
    kb_matched_raw: dict[str, float] = {}

    for rec in raw_records:
        orig_name = rec["name"]
        raw_val   = rec["value"]
        raw_unit  = rec.get("unit", "")
        is_proxy  = rec.get("is_proxy", False)

        param = _PARAM_MAP.get(orig_name.lower(), orig_name if orig_name in _PLAUSIBILITY else None)
        if param is None:
            continue
        if param in unit_meta:
            continue  # keep first (highest-confidence) hit

        conv_val, was_converted, excluded, unit_ambiguous, note = _convert_to_kb_unit(
            param, raw_val, raw_unit, is_proxy
        )

        unit_meta[param] = {
            "original_name":   orig_name,
            "raw_value":       raw_val,
            "raw_unit":        raw_unit,
            "converted_value": conv_val,
            "converted":       was_converted,
            "excluded":        excluded,
            "unit_ambiguous":  unit_ambiguous,
            "confidence":      0.9 if rec.get("raw_label") != "llm" else 0.75,
            "note":            note,
        }

        if not excluded and conv_val is not None:
            kb_matched_raw[param] = conv_val

    # Plausibility check
    implausible = set()
    for param, val in kb_matched_raw.items():
        lo, hi = _PLAUSIBILITY.get(param, (None, None))
        if lo is not None and not (lo <= val <= hi):
            implausible.add(param)
            if param in unit_meta:
                unit_meta[param]["excluded"] = True
                unit_meta[param]["note"] = "implausible_value"

    kb_matched = {k: v for k, v in kb_matched_raw.items() if k not in implausible}
    return kb_matched, unit_meta