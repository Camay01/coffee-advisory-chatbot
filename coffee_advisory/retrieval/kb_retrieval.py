"""
kb_retrieval.py — Knowledge base retrieval and relevance ranking.

Wraps the ChromaDB retriever with query expansion, chunk scoring,
and post-retrieval filtering to keep the LLM prompt focused.
"""

from __future__ import annotations

import re

from config import SOIL_PARAMS
from retrieval.retriever import retrieve, check_zone_exists

def expand_query(user_query: str) -> list[str]:
    """Map natural language questions to KB-friendly search terms."""
    q = user_query.lower()
    variants = [user_query]

    if any(w in q for w in ["phosphorus", " p ", "phospho", "low p", "p level", "p kg"]):
        variants += [
            "Available_P_kg_ha phosphorus interpretation bands low medium high",
            "P kg/ha coffee soil advisory low deficiency fixation",
        ]
    if any(w in q for w in ["nitrogen", " n ", "low n", "n level", "n kg"]):
        variants += [
            "Available_N_kg_ha nitrogen interpretation bands low medium high",
            "nitrogen interpretation coffee soil fertilizer",
        ]
    if any(w in q for w in ["potassium", " k ", "low k", "k level", "k kg"]):
        variants += [
            "Available_K_kg_ha potassium interpretation bands",
            "potassium interpretation coffee soil",
        ]
    if any(w in q for w in ["ph", "acid", "lime", "alkalin", "liming"]):
        variants += [
            "pH soil coffee interpretation bands acidic low high severe",
            "pH lime application coffee Arabica Robusta target band 5.5 6.5",
            "soil acidity coffee advisory intervention correction",
        ]
    if any(w in q for w in ["organic", "oc", "carbon", "organic carbon", "oc%"]):
        variants += [
            "Organic_C_percent organic carbon interpretation low medium high critical",
            "OC% coffee soil advisory organic matter deficiency below 0.75",
        ]
    if any(w in q for w in ["zinc", " zn ", "micronutrient"]):
        variants += ["Zn_mg_kg zinc micronutrient interpretation bands deficiency coffee"]
    if any(w in q for w in ["boron", " b ", "micronutrient"]):
        variants += ["B_mg_kg boron micronutrient interpretation bands deficiency coffee"]
    if any(w in q for w in ["risk", "intervention", "action", "recommend", "fertiliz", "suggest"]):
        variants += ["risk level coffee soil advisory intervention suggested action priority"]

    variants += ["coffee soil interpretation bands advisory rules nutrients"]
    return variants

def _param_keywords(param: str) -> list[str]:
    MAP = {
        "pH":  ["ph", "acidity", "acidic", "alkalin", "lime", "liming", "dolomite", "5.5", "6.5"],
        "OC":  ["organic carbon", "organic_c", "oc%", "oc ", "organic matter", "carbon percent"],
        "N":   ["nitrogen", "available_n", "n kg", " n ", "urea", "ammonium"],
        "P":   ["phosphorus", "available_p", "p kg", " p ", "phospho", "fixation"],
        "K":   ["potassium", "available_k", "k kg", " k ", "potash"],
        "Zn":  ["zinc", "zn_mg", "zn mg", "micronutrient"],
        "B":   ["boron", "b_mg", "b mg", "micronutrient"],
    }
    return MAP.get(param, [param.lower()])


def _score_chunk(chunk: str, measured_params: list[str], query: str, params_are_real: bool) -> int:
    lower = chunk.lower()
    score = 0
    all_params = {p for p, _ in SOIL_PARAMS}

    for param in measured_params:
        if any(kw in lower for kw in _param_keywords(param)):
            score += 3
    for token in re.findall(r'\w+', query.lower()):
        if len(token) > 3 and token in lower:
            score += 1

    if params_are_real:
        unmeasured = all_params - set(measured_params)
        for param in unmeasured:
            hits = sum(1 for kw in _param_keywords(param) if kw in lower)
            if hits >= 2:
                score -= 3
        MICRO_SIGNALS = {
            "Zn": ["zinc deficiency causes", "zinc interaction", "zn and", "zn-p interaction"],
            "B":  ["boron deficiency causes", "boron interaction", "b and", "b toxicity"],
        }
        for micro, signals in MICRO_SIGNALS.items():
            if micro not in measured_params and any(sig in lower for sig in signals):
                score -= 10
        if measured_params:
            if not any(any(kw in lower for kw in _param_keywords(p)) for p in measured_params):
                score -= 5

    REGIONAL = [
        "rainfall", "monsoon", "seasonal pattern", "elevation effect",
        "regional trend", "district average", "zone average", "climatic zone",
    ]
    if any(sig in lower for sig in REGIONAL):
        score -= 4
    return score


def _extract_measured_params_from_query(query: str, user_data: dict | None) -> tuple[list[str], bool]:
    if user_data:
        measured = user_data.get("measured_soil", {})
        if measured:
            return list(measured.keys()), True
    found = []
    q = query.lower()
    if "ph" in q or "acid" in q or "lime" in q:    found.append("pH")
    if "organic" in q or " oc" in q or "carbon" in q: found.append("OC")
    if "nitrogen" in q or " n " in q:              found.append("N")
    if "phospho" in q or " p " in q:               found.append("P")
    if "potassium" in q or " k " in q:             found.append("K")
    if "zinc" in q or " zn" in q:                  found.append("Zn")
    if "boron" in q or " b " in q:                 found.append("B")
    return found, False

def kb_retrieve(
    query: str,
    zone: str = None,
    crop: str = None,
    variety: str = None,
    user_data: dict | None = None,
    max_chunks: int = 6,
) -> list[str]:
    """
    Retrieve, deduplicate, filter, score, and return the most relevant KB chunks.
    """
    seen: set[str] = set()
    all_docs: list[str] = []

    for variant in expand_query(query):
        docs = retrieve(variant, zone=zone, crop=crop)
        for doc in docs:
            if isinstance(doc, dict):
                doc = doc.get("text") or doc.get("content") or str(doc)
            elif not isinstance(doc, str):
                doc = str(doc)
            key = doc[:120]
            if key not in seen:
                seen.add(key)
                all_docs.append(doc)
        if len(all_docs) >= 20:
            break

    if not all_docs:
        docs = retrieve("coffee soil nutrients interpretation bands advisory")
        all_docs = [str(d) if not isinstance(d, str) else d for d in docs]

    # ── Filtering ────────────────────────────────────────────────────────────
    technical_request = any(
        kw in query.lower()
        for kw in ["sampling", "extraction method", "how is it measured", "laboratory", "test method"]
    )
    if not technical_request:
        METHOD_SIGNALS = [
            "sampling method", "extraction method", "sample collection",
            "laboratory procedure", "digestion method", "walkley", "kjeldahl",
        ]
        all_docs = [d for d in all_docs if not any(s in d.lower() for s in METHOD_SIGNALS)]

    zone_history_request = any(kw in query.lower() for kw in ["zone pattern", "historical", "zone trend"])
    if not zone_history_request:
        all_docs = [d for d in all_docs if "historical zone pattern" not in d.lower()]

    regional_request = any(kw in query.lower() for kw in ["rainfall", "monsoon", "climate", "region", "elevation"])
    if not regional_request:
        REGIONAL = ["rainfall", "monsoon", "seasonal pattern", "elevation effect", "agroclimatic"]
        all_docs = [d for d in all_docs if not any(s in d.lower() for s in REGIONAL)]

    measured_params, params_are_real = _extract_measured_params_from_query(query, user_data)

    if params_are_real:
        measured_set = set(measured_params)
        MICRO_SIGNALS = {
            "Zn": ["zinc interaction", "zn-p interaction", "zinc deficiency causes"],
            "B":  ["boron interaction", "b toxicity", "boron deficiency causes"],
        }
        for micro, signals in MICRO_SIGNALS.items():
            if micro not in measured_set:
                all_docs = [d for d in all_docs if not any(s in d.lower() for s in signals)]

    # ── Scoring & ranking ────────────────────────────────────────────────────
    scored = [(d, _score_chunk(d, measured_params, query, params_are_real)) for d in all_docs]
    scored.sort(key=lambda x: x[1], reverse=True)
    if measured_params and params_are_real:
        scored = [(d, s) for d, s in scored if s > 0]
    top_docs = [d for d, _ in scored[:max_chunks]] or all_docs[:max_chunks]

    # Strip zone-level soil data rows from chunks
    ZONE_ROW_RE = re.compile(
        r'^.*\b(zone|district|farm|sample|record|average|mean|plot)\b.*'
        r'(?:pH|OC|N|P|K|Zn|B|EC)[:\s]+[\d]+\.?[\d]*.*$',
        re.IGNORECASE | re.MULTILINE,
    )
    return [ZONE_ROW_RE.sub("", d).strip() for d in top_docs]


def parse_query_context(query: str) -> dict:
    """Extract crop/location context overrides from the query text."""
    ctx = {}
    lower = query.lower()
    if "arabica" in lower:
        ctx["crop"] = "Arabica"
    if "robusta" in lower:
        ctx["crop"] = "Robusta"
    for loc in ["idukki", "wayanad", "kodagu", "hassan", "chikmagalur", "coorg"]:
        if loc in lower:
            ctx["location"] = loc.title()
    return ctx
