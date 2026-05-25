"""
soil_classifier.py — Deterministic soil parameter classification.

All numeric-to-band mapping lives here.  The LLM must NEVER re-classify
these numbers; it only receives the pre-computed status labels.
"""

from config import SOIL_PARAMS, SOIL_THRESHOLDS


def classify_soil_params(soil_vals: dict) -> dict:
    """
    Classify each measured soil value into a status band.

    Returns:
        {param: {"value": float, "status": str, "trigger": bool}}

    This is the ONLY place in the codebase that compares numeric soil values
    to thresholds.
    """
    classified = {}
    for param, value in soil_vals.items():
        try:
            v = float(value)
        except (TypeError, ValueError):
            continue
        bands = SOIL_THRESHOLDS.get(param)
        if bands is None:
            classified[param] = {"value": v, "status": "UNCLASSIFIED", "trigger": False}
            continue
        for upper, label, trigger in bands:
            if upper is None or v < upper:
                classified[param] = {"value": v, "status": label, "trigger": trigger}
                break
    return classified


def build_classified_soil_block(classified: dict) -> str:
    """
    Render pre-classified soil results as a deterministic prompt block.
    The LLM sees status labels, not raw numbers to re-interpret.
    """
    if not classified:
        return "No soil parameters classified."
    lines = []
    for param, info in classified.items():
        flag = "INTERVENTION WARRANTED" if info["trigger"] else "No immediate action"
        lines.append(f"  {param}: {info['value']}  →  [{info['status']}]  ({flag})")
    return "\n".join(lines)


def condition_gate(classified: dict, param: str) -> bool:
    """
    Returns True only if the given parameter was measured AND its trigger is True.
    Use this as a gate before recommending any intervention.
    """
    entry = classified.get(param)
    if entry is None:
        return False
    return entry["trigger"]


def build_soil_summary(user_data: dict) -> tuple[str, str]:
    """
    Build two human-readable strings:
      measured_str     — parameters the user actually provided
      not_provided_str — parameters NOT provided (LLM must not infer deficiency)
    """
    measured_raw = user_data.get("measured_soil", {})

    # FIX: normalise keys to canonical case so "ph" → "pH", "oc" → "OC", etc.
    # LLM fallback in parse_soil_input can return lowercase keys; a case-sensitive
    # lookup would then show those params as "not provided" in the advisory prompt,
    # telling the LLM to ignore values the user actually supplied.
    _KEY_NORM = {k.lower(): k for k, _ in SOIL_PARAMS}
    measured = {_KEY_NORM.get(k.lower(), k): v for k, v in measured_raw.items()}

    measured_parts, not_provided_parts = [], []
    for key, label in SOIL_PARAMS:
        if key in measured:
            measured_parts.append(f"{label}: {measured[key]}")
        else:
            not_provided_parts.append(label)
    measured_str     = ", ".join(measured_parts)     if measured_parts     else "None"
    not_provided_str = ", ".join(not_provided_parts) if not_provided_parts else "None"
    return measured_str, not_provided_str


def ph_severity_note(ph_value: float) -> str:
    """
    Return a calibrated, coffee-specific pH severity note.
    Target band for South Indian coffee: 5.5–6.5.
    """
    if ph_value < 5.0:
        return (
            f"HIGH-PRIORITY SOIL CORRECTION ISSUE — pH {ph_value} indicates severe soil acidity, "
            f"well below the target band of 5.5–6.5 for South Indian coffee. "
            f"At this level, root growth is inhibited, phosphorus fixation is likely occurring "
            f"(reducing fertiliser efficiency significantly), and aluminium and manganese may reach "
            f"toxic levels in the soil solution. Blossom and berry development can also be adversely affected. "
            f"Correcting soil acidity must be the first priority before any NPK application. "
            f"Apply agricultural lime or dolomite based on a buffered soil test recommendation — "
            f"dolomite is preferred where magnesium is also low. Lime application is ideally planned "
            f"around November (pre-blossom season), must be kept separate from fertiliser applications, "
            f"and should be combined with maintaining adequate mulch cover to improve buffering capacity."
        )
    elif ph_value < 5.5:
        return (
            f"MODERATE-PRIORITY SOIL CORRECTION — pH {ph_value} is below the target band of "
            f"5.5–6.5 for South Indian coffee. At this level, phosphorus availability may be reduced "
            f"due to fixation, and root growth and fertiliser response could be impaired over time. "
            f"Liming or dolomite application is recommended before the next NPK cycle — preferably "
            f"around November, kept separate from fertilisers. Maintaining mulch and organic matter "
            f"will improve soil buffering capacity."
        )
    elif ph_value <= 6.5:
        return (
            f"pH {ph_value} is within the target band of 5.5–6.5 for South Indian coffee. "
            f"No acidity correction is required at this time."
        )
    else:
        return (
            f"pH {ph_value} is above the target band of 5.5–6.5. Monitor for alkalinity effects "
            f"on micronutrient availability (particularly iron, manganese, and zinc). "
            f"Avoid liming; focus on organic matter maintenance to buffer pH drift."
        )