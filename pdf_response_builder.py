"""
pdf_response_builder.py — Format PDF extraction results into user-facing messages.

Handles:
  - Unit conversion transparency notes
  - Excluded parameter explanations
  - Tiered advisory-ready / found-but-excluded / unrecognised breakdown
"""

from __future__ import annotations

import re

from config import UNIT_ALIASES

def _normalise_unit(raw: str) -> str:
    return UNIT_ALIASES.get(raw.strip().lower(), "unknown")


_CANON_UNITS = {
    "pH": "", "OC": "%", "N": "kg/ha",
    "P": "kg/ha", "K": "kg/ha", "Zn": "mg/kg", "B": "mg/kg",
}

_FACTOR_NOTE = {
    "N": "× 1.68 (15 cm depth, 1.12 g/cm³ — ICAR standard)",
    "P": "× 1.68 (15 cm depth, 1.12 g/cm³ — ICAR standard)",
    "K": "× 1.68 (15 cm depth, 1.12 g/cm³ — ICAR standard)",
}

_DISPLAY_EXCLUDE = re.compile(
    r"^(medium|coarse|fine|c|m|f|page|report|lab|date|sample|field|county|client|"
    r"farm|broadcast|row|drill|comments?)$|tons?/acre|bu/a|bu/acre|cwt|"
    r"^(n lb|p2o5|k2o|s lb|enp|lime)$",
    re.IGNORECASE,
)

def build_unit_conversion_note(unit_meta: dict) -> str:
    """
    Build a transparent human-readable note about unit conversions, exclusions
    """
    if not unit_meta:
        return ""

    conversion_lines, caveat_lines = [], []

    for param, meta in unit_meta.items():
        raw_val       = meta.get("raw_value")
        raw_unit      = meta.get("raw_unit", "")
        conv_val      = meta.get("converted_value")
        converted     = meta.get("converted", False)
        excluded      = meta.get("excluded", False)
        note          = meta.get("note", "")
        confidence    = meta.get("confidence", 1.0)
        orig_name     = meta.get("original_name", param)
        canon         = _CANON_UNITS.get(param, "")

        if excluded:
            raw_unit_norm = _normalise_unit(str(raw_unit))
            if param in ("N", "P", "K") and raw_unit_norm == "mg/kg":
                excl_why    = "converting ppm to kg/ha requires knowing your lab's bulk density and sample depth"
                excl_action = f"Ask your lab for {param} in **kg/ha**, then enter it manually (e.g. _{param} 240 kg/ha_)"
            elif raw_unit_norm == "lb/a":
                excl_why    = "this value is from the fertiliser recommendations section, not the soil test"
                excl_action = "use the soil test results row"
        
        elif converted:
            factor_str = f" _{_FACTOR_NOTE[param]}_" if param in _FACTOR_NOTE else ""
            conversion_lines.append(
                f"**{param}** (_{orig_name}_): {raw_val} {raw_unit} → **{conv_val} {canon}**{factor_str}"
            )
        if "organic_matter_approx" in note:
            caveat_lines.append(
                "**OC**: Organic Matter (%) was used as a proxy for Organic Carbon — "
                "the interpretation is approximate. True OC ≈ 58% of OM."
            )
        if "nitrate_n_proxy" in note:
            caveat_lines.append(
                "**N**: Nitrate-N (NO₃-N) is not the same as total Available Nitrogen. "
                "The interpretation is indicative only — confirm with your lab."
            )

    if not any([conversion_lines, caveat_lines]):
        return ""

    parts = []
    if conversion_lines:
        parts.append(
            "**Unit conversions applied** (PDF values converted to advisory-standard units):\n"
            + "\n".join(conversion_lines)
        )
    
    if caveat_lines:
        parts.append("**Parameter mapping notes:**\n\n" + "\n".join(caveat_lines))
    return "\n\n".join(parts)


def build_pdf_extraction_response(
    kb_matched: dict,
    all_extracted: dict,
    unit_meta: dict,
    pdf_name: str,
) -> str:
    """
    Build a transparent, tiered user-facing response after PDF extraction.

    Tiers:
      Advisory-ready  — extracted, mapped, validated
      Found but excluded — with reason
      Unrecognised    — noted for reference
    """
    # Tier 1: advisory-ready
    kb_supported = [
        f"**{k}**: {v} {_CANON_UNITS.get(k, '')}".strip()
        for k, v in kb_matched.items()
    ]

    # Tier 3: labels not mapped to any KB parameter
    mapped_labels = {
        meta.get("original_name", "").strip().lower()
        for meta in unit_meta.values()
    }
    not_in_kb = [
        f"  - **{label}**: {val}"
        for label, val in all_extracted.items()
        if label.strip().lower() not in mapped_labels
        and not _DISPLAY_EXCLUDE.search(label.strip())
    ]

    unit_note  = build_unit_conversion_note(unit_meta)
    unit_block = f"\n\n{unit_note}" if unit_note else ""

    n_excluded = sum(
        1 for param, meta in unit_meta.items()
        if param not in kb_matched and meta.get("excluded")
    )

    if kb_supported:
        advisory_str = ", ".join(kb_supported)
        response = (
            f"I've read **{pdf_name}** and extracted the following validated soil values:\n\n"
            f"**Ready for advisory:** {advisory_str}"
            f"{unit_block}"
        )
        if not_in_kb:
            response += (
                f"\n\n**Also found in the report:**\n" + "\n".join(not_in_kb) +
                "\n_These are noted for reference only._"
            )
        response += "\n\nWhat would you like to know about these readings?"
    else:
        all_str  = "\n".join(not_in_kb) if not_in_kb else "_(none found)_"
        excl_note = (
            f"\n\n_{n_excluded} value(s) were found but excluded — see notes above._"
            if n_excluded else ""
        )
        response = (
            f"I read **{pdf_name}** but couldn't extract any values that match my coffee "
            f"knowledge base (pH, OC, N, P, K, Zn, B) with sufficient confidence."
            f"{unit_block}"
            f"{excl_note}\n\n"
            f"**Also found in the report:**\n{all_str}\n\n"
            "The report may use different label names or units. "
            "You can type values directly — e.g. _pH 5.5, N 280, P 8_ — and I'll advise from there."
        )

    return response
