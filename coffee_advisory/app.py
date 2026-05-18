"""
main.py — FastAPI backend for the Coffee Advisory chatbot.

Replaces the Streamlit app.py with a REST API.
All business logic modules remain unchanged:
  config.py              — constants, thresholds, crop catalogue
  llm_client.py          — Ollama wrapper
  soil_classifier.py     — deterministic soil classification
  input_parser.py        — text/profile extraction, soil data detection
  pdf_extractor.py       — universal LLM-based PDF soil extraction
  pdf_response_builder.py — PDF extraction response formatting
  kb_retrieval.py        — ChromaDB retrieval + ranking
  advisory.py            — LLM advisory generation (soil, RAG, side-Q)
  retriever.py           — base ChromaDB client
  build_index.py         — one-time index builder

Run with:
    uvicorn main:app --reload
"""

import hashlib
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import COFFEE_CROPS, CROP_VARIETIES, SOIL_PARAMS
from units.input_parser import (
    contains_profile_info,
    contains_soil_data,
    detect_non_coffee_crop,
    extract_crop,
    extract_farm_size,
    extract_location,
    extract_name,
    is_question,
    parse_soil_input,
    prefill_profile_from_message,
    try_extract_soil_early,
)
from units.pdf_extractor import extract_soil_from_pdf
from retrieval.pdf_response_builder import build_pdf_extraction_response
from retrieval.advisory import (
    answer_side_question,
    build_response,
    dedup_advisory,
    rag_advisory,
    soil_advisory,
)
from retrieval.retriever import check_zone_exists, retrieve

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Coffee Advisory API",
    description="REST backend for the Coffee Soil Advisory chatbot.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# In-memory session store  (replace with Redis/DB for production)
# ---------------------------------------------------------------------------

sessions: dict[str, dict[str, Any]] = {}


def _get_session(session_id: str) -> dict[str, Any]:
    if session_id not in sessions:
        sessions[session_id] = {
            "step": "choose_input",
            "messages": [
                {
                    "role": "assistant",
                    "content": (
                        "Hello! ☕ I'm your **Coffee Advisory Assistant** — here to help you "
                        "make sense of your soil and get the best from your farm.\n\n"
                        "How would you like to get started?"
                    ),
                }
            ],
            "user_data": {},
            "processed_pdfs": set(),
        }
    return sessions[session_id]


# ---------------------------------------------------------------------------
# Shared helpers (ported from app.py)
# ---------------------------------------------------------------------------

def build_completion_message(user_data: dict) -> str:
    name = user_data.get("name", "Grower")
    loc  = user_data.get("location", "Unknown")
    size = user_data.get("farm_size", "Not provided")
    crop = user_data.get("crop", "Unknown")
    var  = user_data.get("variety", "Unknown")

    measured   = user_data.get("measured_soil", {})
    soil_parts = [
        f"{label} {measured[key]}"
        for key, label in SOIL_PARAMS if key in measured
    ]
    soil_str  = ", ".join(soil_parts) if soil_parts else "Not provided"
    soil_note = f"\n**Soil values on file:** {soil_str}" if soil_str != "Not provided" else ""
    size_disp = f"{size} ha" if size not in ("Not provided", None, "") else "Not provided"

    return (
        f"You're all set, **{name}**! Here's a quick summary:\n\n"
        f"📍 **{loc}** &nbsp;|&nbsp; 🌱 **{crop}** — {var} &nbsp;|&nbsp; 🏡 **{size_disp}**"
        f"{soil_note}\n\n"
        "Feel free to ask me anything — soil health, fertiliser timing, "
        "nutrient deficiencies, pest risk."
    )


def _handle_pdf_bytes(file_bytes: bytes, filename: str, session: dict) -> tuple[dict, str, str]:
    """
    Extract soil data from raw PDF bytes.
    Returns (kb_matched, response_text, crop_found).
    Skips re-processing if the same file was already handled this session.
    """
    pdf_hash = hashlib.md5(file_bytes).hexdigest()
    if pdf_hash in session["processed_pdfs"]:
        return {}, "", ""

    session["processed_pdfs"].add(pdf_hash)

    kb_matched, all_extracted, raw_text, unit_meta, crop_found = extract_soil_from_pdf(file_bytes)

    # ── Crop guard ────────────────────────────────────────────────────────────
    non_coffee = detect_non_coffee_crop(crop_found or "")

    if not non_coffee:
        text_sample = raw_text.lower()
        if "tea board" not in text_sample and "cabbage board" not in text_sample:
            non_coffee = detect_non_coffee_crop(raw_text)

    if not non_coffee and (not crop_found or crop_found.lower() == "unknown"):
        if "coffee" in raw_text.lower() or "arabica" in raw_text.lower() or "robusta" in raw_text.lower():
            crop_found = "Coffee"

    if non_coffee:
        response = (
            f"**Crop Mismatch Detected**\n\n"
            f"I detected that this report is for **{non_coffee}**. "
            "I am specialized specifically for coffee soil analysis and don't have the data "
            "to provide accurate advice for other crops. Please upload a coffee soil report!"
        )
        return {}, response, crop_found or non_coffee

    # ── Success path ──────────────────────────────────────────────────────────
    crop_display = f"**Identified Crop:** {(crop_found or 'Coffee').title()}\n\n"
    response = crop_display + build_pdf_extraction_response(kb_matched, all_extracted, unit_meta, filename)
    return kb_matched, response, crop_found or ""


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class SessionRequest(BaseModel):
    session_id: str


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChooseInputRequest(BaseModel):
    session_id: str
    choice: str   # "upload" | "manual"


class SelectVarietyRequest(BaseModel):
    session_id: str
    variety: str


class ChatResponse(BaseModel):
    session_id: str
    step: str
    response: str
    user_data: dict
    messages: list[dict]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/session/new", summary="Create or retrieve a session")
def new_session(req: SessionRequest) -> dict:
    session = _get_session(req.session_id)
    return {
        "session_id": req.session_id,
        "step": session["step"],
        "messages": session["messages"],
        "user_data": session["user_data"],
    }


@app.post("/session/clear", summary="Reset session to initial state")
def clear_session(req: SessionRequest) -> dict:
    if req.session_id in sessions:
        first_msg = sessions[req.session_id]["messages"][0]
        sessions[req.session_id] = {
            "step": "choose_input",
            "messages": [first_msg],
            "user_data": {},
            "processed_pdfs": set(),
        }
    return {"session_id": req.session_id, "status": "cleared"}


@app.post("/chat/choose_input", response_model=ChatResponse, summary="Step 0: choose upload or manual")
def choose_input(req: ChooseInputRequest) -> ChatResponse:
    session = _get_session(req.session_id)

    if req.choice == "upload":
        session["step"] = "upload_pdf"
        response = "Great! Please upload your soil test report. I'll read it and give you personalised recommendations."
        session["messages"].append({"role": "user", "content": "Upload the soil test report"})
        session["messages"].append({"role": "assistant", "content": response})
    elif req.choice == "manual":
        session["step"] = "ask_name"
        response = "No problem! Let's start with the basics — what should I call you?"
        session["messages"].append({"role": "user", "content": "Enter the data manually"})
        session["messages"].append({"role": "assistant", "content": response})
    else:
        raise HTTPException(status_code=400, detail="choice must be 'upload' or 'manual'")

    return ChatResponse(
        session_id=req.session_id,
        step=session["step"],
        response=response,
        user_data=session["user_data"],
        messages=session["messages"],
    )


@app.post("/chat/upload_pdf", response_model=ChatResponse, summary="Step 0b: upload soil PDF")
async def upload_pdf(
    session_id: str = Form(...),
    file: UploadFile = File(...),
) -> ChatResponse:
    session = _get_session(session_id)
    file_bytes = await file.read()

    kb_matched, response, crop_found = _handle_pdf_bytes(file_bytes, file.filename or "upload.pdf", session)

    session["messages"].append({"role": "user", "content": f"[Uploaded PDF: {file.filename}]"})

    if response:
        ud = session["user_data"]
        if kb_matched:
            existing = ud.get("measured_soil", {})
            existing.update(kb_matched)
            ud["measured_soil"] = existing
            ud["soil_raw"] = f"PDF: {file.filename}"

            c_found = (crop_found or "").lower()
            if "arabica" in c_found:
                ud["crop"] = "Arabica"
            elif "robusta" in c_found:
                ud["crop"] = "Robusta"
            elif any(c in c_found for c in COFFEE_CROPS):
                ud["crop"] = "Coffee"

            bridge = (
                f"{response}\n\n"
                "I've got your soil values! To finish your profile and give you a proper advisory, "
                "what should I call you?"
            )
            session["messages"].append({"role": "assistant", "content": bridge})
            session["step"] = "ask_name"
            response = bridge
        else:
            session["messages"].append({"role": "assistant", "content": response})
            session["step"] = "ask_name"
    else:
        response = "File already processed."

    return ChatResponse(
        session_id=session_id,
        step=session["step"],
        response=response,
        user_data=session["user_data"],
        messages=session["messages"],
    )


@app.post("/chat/select_variety", response_model=ChatResponse, summary="Select crop variety from dropdown")
def select_variety(req: SelectVarietyRequest) -> ChatResponse:
    session = _get_session(req.session_id)
    ud = session["user_data"]

    ud["variety"] = req.variety
    session["messages"].append({"role": "user", "content": req.variety})

    response = (
        "Do you have any **soil test values** handy? Type what you know "
        "(e.g. *'pH 5.5, N 280, Zn 0.6'*) — or just type **skip** to move on."
    )
    session["messages"].append({"role": "assistant", "content": response})
    session["step"] = "ask_soil"

    return ChatResponse(
        session_id=req.session_id,
        step=session["step"],
        response=response,
        user_data=ud,
        messages=session["messages"],
    )


@app.post("/chat/message", response_model=ChatResponse, summary="Send a chat message (all onboarding + complete steps)")
def chat_message(req: ChatRequest) -> ChatResponse:
    session = _get_session(req.session_id)
    ud      = session["user_data"]
    prompt  = req.message.strip()

    session["messages"].append({"role": "user", "content": prompt})
    response = ""

    # ── ask_soil ──────────────────────────────────────────────────────────────
    if session["step"] == "ask_soil":
        skip_words = {"skip", "don't know", "dont know", "no", "n/a", "-", "none"}
        if prompt.lower() in skip_words:
            ud["soil_raw"] = "skipped"
            response = build_completion_message(ud)
            session["step"] = "complete"
        else:
            try_extract_soil_early(prompt, ud)
            soil_vals = ud.get("measured_soil", {})
            if soil_vals:
                advisory  = dedup_advisory(soil_advisory(soil_vals, ud))
                completion = build_completion_message(ud)
                response  = f"{completion}\n\n---\n\n{advisory}"
                session["step"] = "complete"
            else:
                response = (
                    "I didn't catch any soil values there. "
                    "Try something like *'pH 5.5, N 280, P 8'*, or type **skip** to carry on."
                )

    # ── ask_name ──────────────────────────────────────────────────────────────
    elif session["step"] == "ask_name":
        non_coffee = detect_non_coffee_crop(prompt)
        if non_coffee:
            response = (
                f"I specialise in coffee (Arabica and Robusta) — I don't have data for "
                f"**{non_coffee}**. If you also grow coffee on your farm, I'm happy to help with that!"
            )
        elif contains_soil_data(prompt):
            prefill_profile_from_message(prompt, ud)
            soil_vals = ud.get("measured_soil", {})
            if soil_vals:
                advisory = dedup_advisory(soil_advisory(soil_vals, ud))
                response = advisory
                session["step"] = "complete"
            else:
                response = "No problem! Let's start with the basics — what should I call you?"
        else:
            extracted_name = extract_name(prompt)
            if extracted_name:
                ud["name"] = extracted_name
                if is_question(prompt) and not contains_profile_info(prompt):
                    side = answer_side_question(prompt, ud)
                    next_q = f"Thanks, **{extracted_name}**! Now — which district or zone is your farm in?"
                    response = build_response(side, next_q)
                else:
                    response = f"Thanks, **{extracted_name}**! Which district or zone is your farm in?"
                session["step"] = "ask_location"
            else:
                if is_question(prompt) and not contains_profile_info(prompt):
                    response = build_response(
                        answer_side_question(prompt, ud),
                        "What should I call you? (or type **skip** to continue)",
                    )
                else:
                    response = (
                        "I didn't quite catch a name there. "
                        "What should I call you? (or type **skip** to continue)"
                    )

    # ── generic soil / question handler ──────────────────────────────────────
    elif contains_soil_data(prompt) or (is_question(prompt) and not contains_profile_info(prompt)):
        prefill_profile_from_message(prompt, ud)
        soil_vals = ud.get("measured_soil", {})
        if soil_vals:
            advisory = dedup_advisory(soil_advisory(soil_vals, ud))
            response = advisory
        else:
            response = answer_side_question(prompt, ud)

    # ── ask_location ──────────────────────────────────────────────────────────
    elif session["step"] == "ask_location":
        try_extract_soil_early(prompt, ud)
        clean_location = ud.get("location") or extract_location(prompt)
        if clean_location is None:
            response = (
                "I didn't quite catch that — could you share your farm's zone or district? "
                "Something like _Kodagu_, _Hassan_, or _Chikmagalur_ would work perfectly."
            )
        else:
            ud["location"] = clean_location
            zone_warning = (
                f"\n\n> Heads up: I don't have specific records for "
                f"**{clean_location}** in my knowledge base, so I'll apply general coffee guidelines."
                if not check_zone_exists(clean_location) else ""
            )
            farm_size = ud.get("farm_size") or extract_farm_size(prompt)
            if farm_size:
                ud["farm_size"] = farm_size
                if ud.get("crop"):
                    next_q = (
                        f"Got it — **{clean_location}**, **{farm_size} ha**, **{ud['crop']}**."
                        f"{zone_warning}\n\nPick your variety and we're good to go!"
                    )
                    session["step"] = "ask_variety"
                else:
                    next_q = (
                        f"Noted — **{clean_location}**, **{farm_size} ha**."
                        f"{zone_warning}\n\nWhat are you growing — Arabica, Robusta, or something else?"
                    )
                    session["step"] = "ask_crop"
            else:
                next_q = (
                    f"Thanks — **{clean_location}** noted.{zone_warning}\n\n"
                    "How large is your farm? _(Enter hectares, or type **skip** if unsure)_"
                )
                session["step"] = "ask_farm_size"

            if is_question(prompt) and not contains_profile_info(prompt):
                response = build_response(answer_side_question(prompt, ud), next_q)
            else:
                response = next_q

    # ── ask_farm_size ─────────────────────────────────────────────────────────
    elif session["step"] == "ask_farm_size":
        try_extract_soil_early(prompt, ud)
        if not ud.get("farm_size"):
            skip_words = {"skip", "don't know", "dont know", "no", "n/a", "-"}
            if prompt.lower() in skip_words:
                ud["farm_size"] = "Not provided"
            else:
                farm_size = extract_farm_size(prompt)
                ud["farm_size"] = farm_size or prompt or "Not provided"

        if ud.get("crop"):
            next_q = f"You're growing **{ud['crop']}** — which variety? Pick one."
            session["step"] = "ask_variety"
        else:
            next_q = "And what are you growing — Arabica, Robusta, or something else?"
            session["step"] = "ask_crop"

        if is_question(prompt) and not contains_profile_info(prompt):
            response = build_response(answer_side_question(prompt, ud), next_q)
        else:
            response = next_q

    # ── ask_crop ──────────────────────────────────────────────────────────────
    elif session["step"] == "ask_crop":
        try_extract_soil_early(prompt, ud)
        ud["crop"] = prompt
        if prompt.lower() in COFFEE_CROPS:
            next_q = f"Which variety of **{prompt}** are you growing?"
            if is_question(prompt) and not contains_profile_info(prompt):
                response = build_response(answer_side_question(prompt, ud), next_q)
            else:
                response = next_q
            session["step"] = "ask_variety"
        else:
            response = (
                f"I specialise in coffee — Arabica and Robusta — so I don't have advisory data "
                f"for **{prompt}**. If you're also growing coffee, I'd be happy to help!"
            )

    # ── complete (RAG chat) ───────────────────────────────────────────────────
    elif session["step"] == "complete":
        response = rag_advisory(prompt, ud)

    else:
        response = "Apologies, I lost my place! Share your name, location, or crop and I'll pick up from there."

    session["messages"].append({"role": "assistant", "content": response})

    return ChatResponse(
        session_id=req.session_id,
        step=session["step"],
        response=response,
        user_data=ud,
        messages=session["messages"],
    )


@app.post("/chat/upload_pdf_complete", response_model=ChatResponse, summary="Upload PDF during complete/chat phase")
async def upload_pdf_complete(
    session_id: str = Form(...),
    file: UploadFile = File(...),
) -> ChatResponse:
    session = _get_session(session_id)
    ud = session["user_data"]
    file_bytes = await file.read()

    kb_matched, response, crop_found = _handle_pdf_bytes(file_bytes, file.filename or "upload.pdf", session)
    session["messages"].append({"role": "user", "content": f"[Uploaded PDF: {file.filename}]"})

    if response:
        if kb_matched:
            if not ud.get("crop"):
                c_found = (crop_found or "").lower()
                if "arabica" in c_found:
                    ud["crop"] = "Arabica"
                elif "robusta" in c_found:
                    ud["crop"] = "Robusta"
                elif any(c in c_found for c in COFFEE_CROPS):
                    ud["crop"] = "Coffee"

            existing = ud.get("measured_soil", {})
            existing.update(kb_matched)
            ud["measured_soil"] = existing
            ud["soil_raw"] = f"PDF: {file.filename}"

        session["messages"].append({"role": "assistant", "content": response})
    else:
        response = "File already processed."

    return ChatResponse(
        session_id=session_id,
        step=session["step"],
        response=response,
        user_data=ud,
        messages=session["messages"],
    )


@app.get("/varieties", summary="Get available crop varieties")
def get_varieties(crop: str | None = None) -> dict:
    """
    Returns variety lists. If crop is provided (Arabica / Robusta / Coffee),
    returns only that crop's varieties. Otherwise returns all.
    """
    if crop:
        key = crop.capitalize()
        if key == "Coffee":
            return {"varieties": CROP_VARIETIES["Arabica"] + CROP_VARIETIES["Robusta"]}
        return {"varieties": CROP_VARIETIES.get(key, [])}
    return {"varieties": CROP_VARIETIES}


@app.get("/health", summary="Health check")
def health() -> dict:
    return {"status": "ok"}