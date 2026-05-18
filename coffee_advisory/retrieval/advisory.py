from __future__ import annotations

from config import SOIL_PARAMS, SOIL_THRESHOLDS
from units.llm_client import llm_call
from units.soil_classifier import (
    classify_soil_params,
    build_classified_soil_block,
    build_soil_summary,
    condition_gate,
    ph_severity_note,
)
from retrieval.kb_retrieval import kb_retrieve, parse_query_context
from retrieval.retriever import check_zone_exists

def dedup_advisory(text: str) -> str:
    """Remove duplicate lines/paragraphs from advisory output."""
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    seen, unique = set(), []
    for block in blocks:
        key = " ".join(block.lower().split())
        if key not in seen:
            seen.add(key)
            unique.append(block)
    return "\n\n".join(unique)


def build_response(side_answer: str | None, next_question: str) -> str:
    if side_answer and "ERROR:" not in side_answer:
        return f"{side_answer}\n\n---\n\n{next_question}"
    return next_question

def soil_advisory(soil_vals: dict, user_data: dict) -> str:
    """
    Full advisory for a set of user-measured soil values.
    Numeric classification is done deterministically before the LLM sees the data.
    """
    zone    = user_data.get("location")
    crop    = user_data.get("crop")
    variety = user_data.get("variety")
    measured_str, not_provided_str = build_soil_summary(user_data)

    # Deterministic classification
    classified = classify_soil_params(soil_vals)
    classified_block = build_classified_soil_block(classified)

    triggered_params = [p for p, i in classified.items() if i["trigger"]]
    triggered_block = (
        "Parameters requiring advisory attention (intervention warranted):\n"
        + "\n".join(f"  {p}: {classified[p]['status']}" for p in triggered_params)
        if triggered_params
        else "No measured parameters require immediate intervention."
    )

    ph_entry = classified.get("pH")
    # Only inject the severity note when pH actually triggers intervention.
    # For adequate pH (trigger=False) the note must be suppressed — otherwise
    # the LLM treats the corrective language as a call to action even though
    # the triggered_block clearly says "No immediate action".
    ph_override = (
        f"\nPH SEVERITY NOTE: {ph_severity_note(ph_entry['value'])}\n"
        if ph_entry and ph_entry["trigger"]
        else (
            f"\nPH NOTE: pH {ph_entry['value']} is within the target band of 5.5–6.5. "
            f"No acidity correction is needed. Do NOT recommend liming.\n"
            if ph_entry else ""
        )
    )

    # Interaction blocks
    oc_trigger = condition_gate(classified, "OC")
    ph_trigger = condition_gate(classified, "pH")
    p_trigger  = condition_gate(classified, "P")
    interaction_block = ""
    if ph_trigger and oc_trigger:
        interaction_block += (
            "\nOC–ACIDITY INTERACTION: Both pH and OC are below adequate. "
            "Low OC reduces soil buffering, accelerates leaching, and worsens acidity. "
            "Correct pH first, then rebuild organic matter.\n"
        )
    if ph_trigger and p_trigger:
        interaction_block += (
            "\nP–ACIDITY INTERACTION: Phosphorus fixation is occurring under current acidity. "
            "Correct soil pH first — this is the most effective step to improve P availability.\n"
        )

    param_list = ", ".join(
        f"{p}: {classified[p]['value']}" for p in classified
    ) or measured_str

    # KB retrieval
    query_ctx      = parse_query_context(param_list)
    retrieval_zone = query_ctx.get("location") or zone
    retrieval_crop = query_ctx.get("crop") or crop
    chunks = kb_retrieve(param_list, zone=retrieval_zone, crop=retrieval_crop,
                         variety=variety, user_data=user_data)
    kb_context = "\n\n---\n\n".join(chunks) if chunks else "NO KB DATA FOUND."
    n_chunks = len(chunks)

    zone_in_kb = check_zone_exists(retrieval_zone) if retrieval_zone else False
    zone_note  = (
        f"\nCRITICAL: Zone '{retrieval_zone}' has no KB records. Do not give zone-specific advice.\n"
    ) if retrieval_zone and not zone_in_kb else ""

    no_data_rule = "4. If KB has no relevant data, say so directly."
    if n_chunks >= 1:
        no_data_rule = (
            f"4. You have {n_chunks} relevant KB records. "
            "Answer directly from them — do NOT say 'I don't have data'."
        )

    PARAMS_WITH_KB = set(SOIL_THRESHOLDS.keys())
    adequate_params     = [p for p, i in classified.items() if not i["trigger"]]
    intervention_params = [p for p, i in classified.items() if i["trigger"]]
    data_state = (
        "PARAMETER DATA STATE:\n"
        + (f"  Adequate (no action): {', '.join(adequate_params)}\n" if adequate_params else "")
        + (f"  Requires attention:   {', '.join(intervention_params)}\n" if intervention_params else "")
    )
    combined_rule = (
        "6. Add a Combined Assessment only if 2+ parameters are triggered. "
        "If only 1, do NOT speculate about others."
        if len(triggered_params) >= 2
        else "6. Only 1 parameter triggered — do NOT add a combined assessment."
    )

    TEMPLATES = """
RESPONSE TEMPLATES:
  LOW/DEFICIENT: "Your [param] of [value] [unit] is below the adequate level. [KB action]"
  HIGH:          "Your [param] of [value] [unit] is above the target range."
  ADEQUATE:      SKIP — do not mention adequate parameters at all.
  NO_KB_DATA:    "No guidance for [param] in my KB. Consult your local agronomist."
"""

    system_prompt = f"""You are a Coffee Soil Advisory Expert. Answer STRICTLY from KB Context.

======= PRE-CLASSIFIED SOIL STATUS (do NOT re-classify) =======
{classified_block}

{triggered_block}
{ph_override}{interaction_block}
{data_state}
CRITICAL: These labels were computed by Python thresholds. Never re-interpret raw numbers.
Recommend intervention ONLY for parameters where trigger=True.
===============================================================

RISK LABELLING: Open with "This is a high-priority soil correction issue." for pH < 5.0 or
any most-severe band. Use "moderate-priority concern" for pH 5.0–5.5 or similar.

COFFEE IMPACTS (include when triggered):
  pH < 5.5: root growth inhibition, phosphorus fixation, reduced fertiliser efficiency.
  pH < 5.0: aluminium/manganese toxicity risk, poor blossom/berry development.
  Low OC + low pH: reduced buffering, nutrient leaching.
  Low P under acidic pH: correct pH before adding P fertiliser.

SPECIFIC RECOMMENDATIONS:
  Acidity: agricultural lime or dolomite (prefer dolomite if Mg low).
  Timing: apply around November, separate from fertilisers.
  Always mention mulch for pH buffering. Recommend retesting after correction.
  Do NOT give specific kg/ha dosage unless KB provides it.

PARAMETERS MEASURED: {param_list}
NOT PROVIDED — do NOT infer deficiency: {not_provided_str}
Crop: {crop or "Unknown"} | Variety: {variety or "Unknown"} | Zone: {zone or "Unknown"}
{zone_note}
{TEMPLATES}
RULES:
1. Comment ONLY on parameters provided: {param_list}
2. Never mention parameters not in that list.
3. Use ONLY KB Context.
{no_data_rule}
5. Severity order: pH → OC → N → P → K → Zn → B
{combined_rule}
7. Do NOT say "Based on the KB Context" — speak as an advisor.
8. For pH: use PH SEVERITY NOTE once. Do not repeat.
9. Write clearly for a farmer — direct, practical, no jargon.
10. Keep concise — no repeated threshold explanations.

ANTI-HALLUCINATION:
H1. Never name a disease unless KB Context explicitly links it.
H2. Never state a fertiliser dosage unless KB provides one.
H3. Never reference parameters not provided.
H4. CRITICAL — VALUE ANCHORING: The ONLY numeric values you may state are those
    listed in PRE-CLASSIFIED SOIL STATUS above. Never substitute, recall, or
    invent a different number. If you write "your pH of X", X must exactly match
    the value in the classified block — not any example from training data.

OUTPUT: Natural prose or short bullets, no section headings, no ALL-CAPS labels outside risk opener.
If ALL parameters adequate: say so in one sentence and stop.

Anti-contradiction: Never say "I don't have guidance" for parameters in {sorted(PARAMS_WITH_KB)}.

======= KB CONTEXT =======
{kb_context}
=========================="""

    return llm_call(
        system=system_prompt,
        user=f"Interpret my soil values: {', '.join(f'{k}: {v}' for k, v in soil_vals.items())}",
        num_predict=1200,
    )

def answer_side_question(user_message: str, user_data: dict) -> str:
    """Answer a KB question asked during onboarding, without blocking the flow."""
    zone    = user_data.get("location")
    crop    = user_data.get("crop")
    variety = user_data.get("variety")
    measured_str, not_provided_str = build_soil_summary(user_data)

    measured_soil = user_data.get("measured_soil", {})
    classified_side = classify_soil_params(measured_soil) if measured_soil else {}
    classified_block = build_classified_soil_block(classified_side) if classified_side else measured_str

    query_ctx      = parse_query_context(user_message)
    retrieval_zone = query_ctx.get("location") or zone
    retrieval_crop = query_ctx.get("crop") or crop
    chunks = kb_retrieve(user_message, zone=retrieval_zone, crop=retrieval_crop,
                         variety=variety, user_data=user_data)
    n_chunks = len(chunks)
    kb_context = "\n\n---\n\n".join(chunks) if chunks else "NO KB DATA FOUND."

    zone_in_kb = check_zone_exists(retrieval_zone) if retrieval_zone else False
    zone_note  = (
        f"\nNOTE: Zone '{retrieval_zone}' has NO records. Do NOT give zone-specific advice.\n"
    ) if retrieval_zone and not zone_in_kb else ""

    no_data_rule = "4. If KB has no relevant data, say so."
    if n_chunks >= 1:
        no_data_rule = (
            f"4. You have {n_chunks} relevant KB records. "
            "You MUST answer from them. Do NOT say 'I don't have data'."
        )

    r1_rule = (
        f"Include ONLY information relevant to: {measured_str}."
        if measured_str != "None"
        else (
            "Answer the question asked. No soil data provided yet — "
            "give guidance ranges only (e.g. 'target pH is 5.5–6.5')."
        )
    )

    PARAMS_WITH_KB = set(SOIL_THRESHOLDS.keys())

    system_prompt = f"""You are a Coffee Soil Advisory Expert. Answer STRICTLY from KB Context.

SOIL CLASSIFICATION (deterministic — do NOT re-classify):
{classified_block}

SOIL PARAMETERS NOT PROVIDED: {not_provided_str}
{zone_note}

RULES:
1. Answer ONLY from KB Context.
2. NEVER advise on parameters not in USER-MEASURED list.
3. NEVER assume crop-specific data for crop NOT YET COLLECTED.
{no_data_rule}
5. pH target band: always 5.5–6.5.
6. Answer in 3–6 sentences. Give ranges where available.
7. Write clearly for a farmer — direct, no jargon.
8. Do NOT say "Based on the KB Context".

RELEVANCE: {r1_rule}

Anti-contradiction: Never say "I don't have guidance" for {sorted(PARAMS_WITH_KB)}.

======= KB CONTEXT =======
{kb_context}
=========================="""

    return llm_call(system=system_prompt, user=user_message, num_predict=600)

def rag_advisory(query: str, user_data: dict) -> str:
    """Full RAG advisory for post-onboarding queries."""
    zone    = user_data.get("location")
    crop    = user_data.get("crop")
    variety = user_data.get("variety")
    measured_str, not_provided_str = build_soil_summary(user_data)

    measured_soil = user_data.get("measured_soil", {})
    classified    = classify_soil_params(measured_soil) if measured_soil else {}
    classified_block = build_classified_soil_block(classified) if classified else measured_str

    triggered_params = [p for p, i in classified.items() if i["trigger"]]
    triggered_block  = (
        "Parameters requiring attention:\n"
        + "\n".join(f"  {p}: {classified[p]['status']}" for p in triggered_params)
        if triggered_params else "No measured parameters require immediate intervention."
    )

    ph_entry    = classified.get("pH")
    oc_trigger  = condition_gate(classified, "OC")
    ph_trigger  = condition_gate(classified, "pH")
    ph_override = (
        f"\nPH SEVERITY NOTE: {ph_severity_note(ph_entry['value'])}\n"
        if ph_entry and ph_entry["trigger"]
        else (
            f"\nPH NOTE: pH {ph_entry['value']} is within the target band of 5.5–6.5. "
            f"No acidity correction is needed. Do NOT recommend liming.\n"
            if ph_entry else ""
        )
    )
    p_trigger  = condition_gate(classified, "P")
    interaction_block = ""
    if ph_trigger and oc_trigger:
        interaction_block += (
            "\nOC–ACIDITY INTERACTION: Both pH and OC are below adequate. "
            "Correct pH first, then rebuild organic matter.\n"
        )
    if ph_trigger and p_trigger:
        interaction_block += (
            "\nP–ACIDITY INTERACTION: Correct soil pH before adding P fertiliser.\n"
        )

    query_ctx      = parse_query_context(query)
    retrieval_zone = query_ctx.get("location") or zone
    retrieval_crop = query_ctx.get("crop") or crop
    chunks = kb_retrieve(query, zone=retrieval_zone, crop=retrieval_crop,
                         variety=variety, user_data=user_data)
    n_chunks   = len(chunks)
    kb_context = "\n\n---\n\n".join(chunks) if chunks else "NO KB DATA FOUND."

    zone_in_kb = check_zone_exists(retrieval_zone) if retrieval_zone else False
    zone_note  = (
        f"\nCRITICAL: Zone '{retrieval_zone}' has no KB records. No zone-specific advice.\n"
    ) if retrieval_zone and not zone_in_kb else ""

    query_ctx_note = ""
    if query_ctx:
        parts = []
        if "crop" in query_ctx:
            parts.append(f"Crop in query: {query_ctx['crop']}")
        if "location" in query_ctx:
            parts.append(f"Location in query: {query_ctx['location']}")
        if parts:
            query_ctx_note = "\nQUERY CONTEXT OVERRIDE:\n" + "\n".join(f"  - {p}" for p in parts) + "\n"

    PARAMS_WITH_KB = set(SOIL_THRESHOLDS.keys())
    adequate_rag     = [p for p, i in classified.items() if not i["trigger"]]
    intervention_rag = [p for p, i in classified.items() if i["trigger"]]
    data_state = (
        "PARAMETER DATA STATE:\n"
        + (f"  Adequate: {', '.join(adequate_rag)}\n" if adequate_rag else "")
        + (f"  Requires attention: {', '.join(intervention_rag)}\n" if intervention_rag else "")
        + (f"  Not measured: {not_provided_str}\n" if not_provided_str != "None" else "")
    )

    rule1 = "1. Use ONLY KB Context. If no relevant data, say so."
    if n_chunks >= 1:
        rule1 = (
            f"1. Use ONLY KB Context. You have {n_chunks} relevant KB records. "
            "Answer directly from them — do NOT say 'I don't have data'."
        )

    TEMPLATES = """
RESPONSE TEMPLATES:
  LOW:         "Your [param] of [value] [unit] is below the adequate level."
  DEFICIENT:   "Your [param] of [value] [unit] is deficient."
  HIGH:        "Your [param] of [value] [unit] is above the target range."
  ADEQUATE:    SKIP entirely.
  NO_KB_DATA:  "No data in my KB for [param]."
"""

    system_prompt = f"""You are a Coffee Soil Advisory Expert. Answer STRICTLY from KB Context.

PROFILE:
  Name: {user_data.get('name', 'Grower')} | Location: {zone} | Crop: {crop} | Variety: {variety}
{zone_note}{query_ctx_note}

PRE-CLASSIFIED SOIL STATUS (do NOT re-classify):
{classified_block}

{triggered_block}
{ph_override}{interaction_block}
{data_state}

SOIL NOT PROVIDED — do NOT infer deficiency: {not_provided_str}

{TEMPLATES}

AGRONOMIST REFERRAL: Suggest ONLY when value is borderline, KB lacks guidance, or rates are requested.
Do NOT repeat more than once. Do NOT add as a generic disclaimer.

ANTI-CONTRADICTION: Never say "I don't have guidance" for {sorted(PARAMS_WITH_KB)}.
CONFIDENT LANGUAGE: State interpretations directly. Do not hedge.
ANTI-HALLUCINATION:
  2. Never invent values, dosages, or zone data not in KB Context.
  3. Only comment on measured parameters.
  4. If crop is NOT coffee, say so and stop.
  10. Do NOT say "Based on the KB Context".

RULES:
{rule1}

======= KB CONTEXT =======
{kb_context}
=========================="""

    return llm_call(system=system_prompt, user=query, num_predict=1200)