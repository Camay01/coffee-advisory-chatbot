from __future__ import annotations

import ast
import io
import re

from config import SOIL_PARAMS, UNIT_ALIASES, PPM_TO_KG_HA_FACTOR
from units.llm_client import llm_call


# ── Soil keywords used to filter raw text before sending to LLM ──────────────
_SOIL_KEYWORDS = [
    "ph",
    "organic carbon",
    "organic matter",
    "oc",

    "nitrogen",
    "available nitrogen",
    "available n",
    "nitrate",
    "no3-n",

    "phosphorus",
    "available phosphorus",
    "available p",
    "p2o5",
    "rock phosphate",

    "potassium",
    "available potassium",
    "available k",
    "k2o",
    "potash",

    "zinc",
    "zn",

    "boron",
    "b",

    "kg/ha",
    "mg/kg",
    "ppm",
    "%",
]

# ── Plausibility ranges (hard limits; values outside → excluded) ──────────────
_PLAUSIBILITY: dict[str, tuple[float, float]] = {
    "pH": (3.0, 10.0),
    "OC": (0.0, 20.0),
    "N":  (0.0, 2000.0),
    "P":  (0.0, 500.0),
    "K":  (0.0, 2000.0),
    "Zn": (0.0, 100.0),
    "B":  (0.0, 20.0),
}

def extract_soil_from_pdf(file_bytes: bytes) -> tuple[dict, dict, str, dict, str]:
    """
    Extract soil values from any uploaded PDF, regardless of format.

    Returns:
        raw_text      — full text extracted from the PDF
        unit_meta     — provenance record per parameter
        crop_found    — string name of crop found in PDF
    """
    raw_text = _extract_raw_text(file_bytes)

    if raw_text.startswith("[Could not read PDF"):
        return {}, {}, raw_text, {}, ""

    # Scanned PDF: very little text
    if len(raw_text.strip()) < 30:
        return (
            {}, {},
            "[PDF appears to be a scanned image — text could not be extracted automatically. "
            "Please type your soil values manually (e.g. pH 5.5, N 280, P 8).]",
            {}, ""
        )

    kb_matched, all_extracted, unit_meta, crop_found = _llm_extract_soil(raw_text)
    return kb_matched, all_extracted, raw_text, unit_meta, crop_found

def _extract_raw_text(file_bytes: bytes) -> str:
    """
    Extract plain text from PDF using pdfplumber.
    Tries up to 3 pages; skips pages with no numeric content.
    Fast strategy: plain text only, no table parsing.
    """
    try:
        import pdfplumber
    except ImportError:
        return "[pdfplumber not installed — cannot read PDF.]"

    try:
        raw_text = ""
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages[:5]:  # scan at most 5 pages
                words = page.extract_words() or []
                # Skip pages with no numeric content (cover pages, T&C, etc.)
                if not any(re.search(r"\d", w.get("text", "")) for w in words[:200]):
                    continue
                page_text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
                raw_text += page_text + "\n"

                # Early exit once we have enough soil-relevant text
                lower = raw_text.lower()
                found_count = sum(1 for kw in _SOIL_KEYWORDS if kw in lower)
                if found_count >= 5:
                    break
        return raw_text.strip()
    except Exception as e:
        return f"[Could not read PDF: {e}]"


def _filter_relevant_lines(raw_text: str) -> str:
    """
    Return only lines that contain soil keywords or numeric values.
    Keeps the LLM prompt short and focused.
    """
    relevant = []
    for line in raw_text.split("\n"):
        lower = line.lower()
        has_keyword = any(kw in lower for kw in _SOIL_KEYWORDS)
        has_number  = bool(re.search(r"\d", line))
        if has_keyword or has_number:
            relevant.append(line)
    return "\n".join(relevant)


def _llm_extract_soil(raw_text: str) -> tuple[dict, dict, dict]:
    """
    Use the LLM to extract all soil parameter values from the PDF text,
    regardless of the PDF's layout or label naming conventions.

    Returns (kb_matched, all_extracted, unit_meta).
    """
    # Send only the relevant lines, capped at 2 500 chars
    filtered = _filter_relevant_lines(raw_text)
    context  = filtered[:6000] if len(filtered) > 2500 else filtered

    # If filtering produces too little text, fall back to the raw text slice
    if len(context.strip()) < 50:
        context = raw_text[:2500]

    llm_result = llm_call(
        system=(
            "You are a soil test report data extraction assistant.\n"
            "Extract soil parameter values and the CROP NAME from the text below.\n\n"
            "Target parameters:\n"
            "  pH, OC (organic carbon %), N (available nitrogen kg/ha),\n"
            "  P (available phosphorus kg/ha), K (available potassium kg/ha),\n"
            "  Zn (zinc mg/kg), B (boron mg/kg)\n\n"
            "Target metadata:\n"
            "  Crop (e.g. 'Coffee', 'Arabica', 'Tea', 'Pepper')\n\n"
            "INSTRUCTIONS:\n"
            "1. Identify the primary SUBJECT CROP. Look for 'Crop', 'Plantation', or the report title.\n"
            "   If the report title or subject mentions 'Coffee', 'Arabica', or 'Robusta', identify it as 'Coffee'.\n"
            "   If the report is explicitly for another crop (Tea, Rubber, etc.), identify it exactly as found.\n"
            "   Only return 'Unknown' if there is absolutely no mention of any crop type.\n"
            "2. Map any common label variants to the correct key (pH, OC, N, P, K, Zn, B).\n"
            "3. Return ONLY a valid Python dict with two keys:\n"
            "   'crop': 'Crop Name',\n"
            "   'measurements': [{'name': 'pH', 'value': 5.5, 'unit': ''}, ...]\n"
            "4. If a parameter or the crop is not found, use None or [].\n"
            "5. Do NOT wrap in markdown. Do NOT add any explanation.\n"
        ),
        user=f"Soil report text:\n\n{context}",
        num_predict=800,
    )

    try:
        # Robust parsing of the dict structure
        clean = re.sub(r"```[a-z]*\n?", "", llm_result, flags=re.IGNORECASE).replace("```", "").strip()
        match = re.search(r"\{.*\}", clean, re.DOTALL)
        if match:
            parsed = ast.literal_eval(match.group())
            crop_found = str(parsed.get("crop") or "").strip()
            raw_records = parsed.get("measurements", [])
        else:
            crop_found = ""
            raw_records = _parse_llm_list(llm_result)
    except Exception:
        crop_found = ""
        raw_records = _parse_llm_list(llm_result)

    # ── Regex Fallback ───────────────────────────────────────────────────────
    # If the LLM missed a parameter, try the regex heuristic.
    regex_found = _regex_extract(raw_text)
    llm_found_names = {r["name"].lower() for r in raw_records}
    
    for param_name, value in regex_found.items():
        if param_name.lower() not in llm_found_names:
            raw_records.append({
                "name": param_name,
                "value": value,
                "unit": "",  # regex doesn't capture units reliably
            })

    kb_matched, all_extracted, unit_meta = _validate_and_convert(raw_records)
    return kb_matched, all_extracted, unit_meta, crop_found


def _parse_llm_list(llm_result: str) -> list[dict]:
    """Parse LLM output into a list of {name, value, unit} dicts."""
    records: list[dict] = []
    try:
        clean = re.sub(r"```[a-z]*", "", llm_result, flags=re.IGNORECASE).replace("```", "").strip()
        match = re.search(r"\[.*\]", clean, re.DOTALL)
        if match:
            parsed = ast.literal_eval(match.group())
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict) and "name" in item and "value" in item:
                        try:
                            records.append({
                                "name":  str(item["name"]).strip(),
                                "value": float(item["value"]),
                                "unit":  str(item.get("unit", "")).strip(),
                            })
                        except (TypeError, ValueError):
                            pass
    except Exception:
        pass
    return records


def _normalise_unit(raw: str) -> str:
    return UNIT_ALIASES.get(raw.strip().lower(), "unknown")


def _convert_to_kb_unit(param: str, value: float, raw_unit: str):
    """
    Convert extracted value to KB canonical unit.
    Returns (converted_value, was_converted, excluded, unit_ambiguous).
    None for converted_value means the parameter should be excluded.
    """
    unit = _normalise_unit(raw_unit)

    # Reject fertiliser recommendation units
    if unit == "lb/a":
        return None, False, True, False

    if param == "pH":
        # Guard against EC readings mislabelled as pH
        raw_lower = raw_unit.strip().lower()
        if any(ec in raw_lower for ec in ("mmhos", "mhos", "ds/m", "ds m", "ms/cm")):
            return None, False, True, False
        return value, False, False, False

    if param == "OC":
        if unit in ("none", "%"):
            return value, False, False, (unit == "none")
        if unit == "g/kg":
            return round(value / 10.0, 4), True, False, False
        return None, False, True, False

    if param in ("N", "P", "K"):
        if unit == "kg/ha":
            return value, False, False, False
        if unit == "mg/kg":
            # ppm → kg/ha requires lab-specific bulk density; flag rather than guess
            return None, False, True, False
        if unit == "none":
            return value, False, False, True  # unit absent; assume kg/ha
        return None, False, True, False

    if param in ("Zn", "B"):
        if unit in ("mg/kg", "none"):
            return value, False, False, (unit == "none")
        if unit == "kg/ha":
            return None, False, True, False
        if unit == "unknown":
            return value, False, False, False
        return None, False, True, False

    return None, False, True, False


def _validate_and_convert(
    raw_records: list[dict],
) -> tuple[dict, dict, dict]:
    """
    Map LLM-extracted records to KB canonical keys, convert units,
    and validate plausibility.

    Returns (kb_matched, all_extracted, unit_meta).
    """
    # Canonical key mapping for common synonyms
    PARAM_MAP = {
        "ph": "pH", "oc": "OC", "organic carbon": "OC", "organic matter": "OC",
        "n": "N", "nitrogen": "N", "available n": "N", "available nitrogen": "N",
        "nitrate-n": "N", "no3-n": "N",
        "p": "P", "phosphorus": "P", "available p": "P", "available phosphorus": "P",
        "k": "K", "potassium": "K", "available k": "K", "available potassium": "K",
        "zn": "Zn", "zinc": "Zn",
        "b": "B", "boron": "B",
    }

    all_extracted: dict[str, float] = {}
    unit_meta:     dict[str, dict]  = {}
    kb_matched_raw: dict[str, float] = {}

    for rec in raw_records:
        orig_name = rec["name"]
        raw_val   = rec["value"]
        raw_unit  = rec["unit"]

        all_extracted[orig_name] = raw_val

        # Map to canonical KB key
        param = PARAM_MAP.get(orig_name.lower())
        if param is None:
            continue  # not a KB parameter — noted in all_extracted only

        # Avoid overwriting a high-confidence record with a lower-confidence one
        if param in unit_meta:
            continue

        conv_val, was_converted, excluded, unit_ambiguous = _convert_to_kb_unit(
            param, raw_val, raw_unit
        )

        unit_meta[param] = {
            "original_name":  orig_name,
            "raw_value":      raw_val,
            "raw_unit":       raw_unit,
            "converted_value": conv_val,
            "converted":      was_converted,
            "excluded":       excluded,
            "unit_ambiguous": unit_ambiguous,
            "confidence":     0.85,
        }

        if not excluded and conv_val is not None:
            kb_matched_raw[param] = conv_val

    # Plausibility validation
    implausible = set()
    for param, val in kb_matched_raw.items():
        lo, hi = _PLAUSIBILITY.get(param, (None, None))
        if lo is not None and not (lo <= val <= hi):
            implausible.add(param)
            if param in unit_meta:
                unit_meta[param]["excluded"] = True
                unit_meta[param]["note"] = "implausible_value"

    kb_matched = {k: v for k, v in kb_matched_raw.items() if k not in implausible}
    return kb_matched, all_extracted, unit_meta

def _regex_extract(raw_text: str):
    patterns = {
        "pH": r"pH\s+([0-9.]+)",
        "OC": r"Organic Carbon.*?([0-9.]+)",
        "P": r"Available Phosphorus.*?([0-9.]+)",
        "K": r"Available Potassium.*?([0-9.]+)",
        "Zn": r"Zinc.*?([0-9.]+)",
        "B": r"Boron.*?([0-9.]+)",
    }

    found = {}

    for param, pattern in patterns.items():
        match = re.search(pattern, raw_text, re.IGNORECASE)
        if match:
            try:
                found[param] = float(match.group(1))
            except:
                pass

    return found