"""
app.py — Streamlit entry point for the Coffee Advisory chatbot.

This file handles ONLY:
  - UI layout and styling
  - Session state and step routing
  - Wiring user actions to the right module functions

All logic lives in dedicated modules:
  config.py              — constants, thresholds, crop catalogue
  llm_client.py          — Ollama wrapper
  soil_classifier.py     — deterministic soil classification
  input_parser.py        — text/profile extraction, soil data detection
  pdf_extractor.py       — universal LLM-based PDF soil extraction
  pdf_response_builder.py — PDF extraction response formatting
  kb_retrieval.py        — ChromaDB retrieval + ranking
  advisory.py            — LLM advisory generation (soil, RAG, side-Q)
  retriever.py           — base ChromaDB client (unchanged)
  build_index.py         — one-time index builder (unchanged)
"""

import base64
import hashlib

import streamlit as st

from config import CROP_VARIETIES, COFFEE_CROPS, SOIL_PARAMS
from units.input_parser import (
    contains_soil_data,
    parse_soil_input,
    try_extract_soil_early,
    prefill_profile_from_message,
    extract_name,
    extract_location,
    extract_farm_size,
    extract_crop,
    detect_non_coffee_crop,
    is_question,
    contains_profile_info,
)
from units.pdf_extractor import extract_soil_from_pdf
from retrieval.pdf_response_builder import build_pdf_extraction_response
from retrieval.advisory import (
    soil_advisory,
    answer_side_question,
    rag_advisory,
    dedup_advisory,
    build_response,
)
from retrieval.retriever import retrieve, check_zone_exists

st.set_page_config(
    page_title="COFFEE ADVISORY",
    layout="centered",
    initial_sidebar_state="collapsed",
)

def get_base64_image(path: str) -> str:
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return ""


image = get_base64_image("coffee.png")

def build_completion_message(user_data: dict) -> str:
    name = user_data.get("name", "Grower")
    loc  = user_data.get("location", "Unknown")
    size = user_data.get("farm_size", "Not provided")
    crop = user_data.get("crop", "Unknown")
    var  = user_data.get("variety", "Unknown")

    measured = user_data.get("measured_soil", {})
    soil_parts = [
        f"{label} {measured[key]}"
        for key, label in SOIL_PARAMS if key in measured
    ]
    soil_str   = ", ".join(soil_parts) if soil_parts else "Not provided"
    soil_note  = f"\n**Soil values on file:** {soil_str}" if soil_str != "Not provided" else ""
    size_disp  = f"{size} ha" if size not in ("Not provided", None, "") else "Not provided"

    return (
        f"You're all set, **{name}**! Here's a quick summary:\n\n"
        f"📍 **{loc}** &nbsp;|&nbsp; 🌱 **{crop}** — {var} &nbsp;|&nbsp; 🏡 **{size_disp}**"
        f"{soil_note}\n\n"
        "Feel free to ask me anything — soil health, fertiliser timing, nutrient deficiencies, pest risk."
    )


def _handle_pdf_upload(pdf_file, key_prefix: str = "") -> tuple[dict, str, str]:
    """
    Read a PDF upload, extract soil values, and return (kb_matched, response_text).
    Caches per unique file hash so Streamlit re-runs don't re-process.
    """
    file_bytes = pdf_file.read()
    pdf_key = f"{key_prefix}_pdf_{hashlib.md5(file_bytes).hexdigest()}"

    if st.session_state.get(pdf_key):
        return {}, ""   # already processed this file

    st.session_state[pdf_key] = True
    with st.spinner("Reading your document…"):
        kb_matched, all_extracted, raw_text, unit_meta, crop_found = extract_soil_from_pdf(file_bytes)

    # ── Crop Guard ───────────────────────────────────────────────────────────
    from units.input_parser import detect_non_coffee_crop
    
    # 1. Check AI-detected crop
    non_coffee = detect_non_coffee_crop(crop_found or "")
    
    # 2. Heuristic Backup: If AI said Coffee or Unknown, but raw text contains "Tea", "Cabbage",etc.
    if not non_coffee:
        # We scan raw text but ignore common lab header noise
        text_sample = raw_text.lower()
        if "tea board" not in text_sample and "cabbage board" not in text_sample:
            non_coffee = detect_non_coffee_crop(raw_text)

    # 3. Default to Coffee if AI is unsure but text contains "Coffee"
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
        return {}, response, crop_found

    # ── Success Path ────────────────────────────────────────────────────────
    crop_display = f"**Identified Crop:** {crop_found.title() or 'Coffee'}\n\n"
    response = crop_display + build_pdf_extraction_response(kb_matched, all_extracted, unit_meta, pdf_file.name)
    return kb_matched, response, crop_found

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap');
    .stApp { background: #0D0D0D; font-family: 'Outfit', sans-serif; }
    .main  { background: radial-gradient(circle at top right, #1a140f, #0d0d0d); }
    .header-container { text-align: center; padding: 0.5rem 0; margin-bottom: 1rem; }
    .main-title {
        font-weight: 600; font-size: 2.5rem; letter-spacing: -1px;
        background: linear-gradient(135deg, #EAD7BB, #D4AF37);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .stChatMessage [data-testid="stChatMessageContent"] {
        background: rgba(255,255,255,0.04); backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px; padding: 1.2rem; color: #E0E0E0;
    }
    [data-testid="stChatMessageUser"] [data-testid="stChatMessageContent"] {
        background: linear-gradient(135deg, #3E2723, #1B1210) !important;
        border: 1px solid rgba(212,175,55,0.2) !important;
    }
    .coffee-container { display: flex; justify-content: center; margin-bottom: 1rem; }
    .coffee-banner {
        width: 100px; height: 100px; object-fit: cover;
        border-radius: 50%; border: 2px solid #D4AF37;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }
    div[data-testid="stButton"].clear-btn-wrapper button {
        position: fixed; top: 0.6rem; right: 1rem; z-index: 9999;
        background: rgba(255,255,255,0.06); color: #aaa;
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 20px; padding: 0.25rem 0.8rem;
        font-size: 0.75rem; transition: all 0.2s ease;
    }
    div[data-testid="stButton"].clear-btn-wrapper button:hover {
        background: rgba(212,175,55,0.15); color: #D4AF37; border-color: #D4AF37;
    }
    .stSelectbox div[data-baseweb="select"] {
        background-color: rgba(255,255,255,0.05) !important; border-radius: 12px !important;
    }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
    </style>
""", unsafe_allow_html=True)

if "step" not in st.session_state:
    st.session_state.step = "choose_input"
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": (
            "Hello! ☕ I'm your **Coffee Advisory Assistant** — here to help you make sense "
            "of your soil and get the best from your farm.\n\nHow would you like to get started?"
        ),
    }]
if "user_data" not in st.session_state:
    st.session_state.user_data = {}

# Clear button
if st.button("✕ Clear", key="clear_btn"):
    st.session_state.messages   = [st.session_state.messages[0]]
    st.session_state.step       = "choose_input"
    st.session_state.user_data  = {}
    st.rerun()

# Header
st.markdown("<div class='header-container'>", unsafe_allow_html=True)
if image:
    st.markdown(
        f'<div class="coffee-container">'
        f'<img src="data:image/png;base64,{image}" class="coffee-banner">'
        f'</div>',
        unsafe_allow_html=True,
    )
st.markdown("<h1 class='main-title'>COFFEE ADVISORY</h1>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if st.session_state.step != "complete":

    # ── Step 0: choose input method ──────────────────────────────────────────
    if st.session_state.step == "choose_input":
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Upload the soil test report", use_container_width=True):
                st.session_state.step = "upload_pdf"
                st.session_state.messages.append({"role": "user", "content": "Upload the soil test report"})
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "Great! Please upload your soil test report below. I'll read it and give you personalised recommendations.",
                })
                st.rerun()
        with col2:
            if st.button("Enter the data manually", use_container_width=True):
                st.session_state.step = "ask_name"
                st.session_state.messages.append({"role": "user", "content": "Enter the data manually"})
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "No problem! Let's start with the basics — what should I call you?",
                })
                st.rerun()

    # ── Step 0b: PDF upload flow ─────────────────────────────────────────────
    elif st.session_state.step == "upload_pdf":
        st.markdown("**Upload your soil test report (PDF)**")
        uploaded_pdf = st.file_uploader(
            "Upload soil test PDF", type=["pdf"],
            label_visibility="collapsed", key="welcome_pdf_uploader",
        )
        if uploaded_pdf is not None:
            kb_matched, response, crop_found = _handle_pdf_upload(uploaded_pdf, key_prefix="welcome")
            if response:
                st.session_state.messages.append({"role": "user", "content": f"[Uploaded PDF: {uploaded_pdf.name}]"})
                if kb_matched:
                    # Save soil data
                    existing = st.session_state.user_data.get("measured_soil", {})
                    existing.update(kb_matched)
                    st.session_state.user_data["measured_soil"] = existing
                    st.session_state.user_data["soil_raw"] = f"PDF: {uploaded_pdf.name}"
                    
                    # Auto-update crop if identified
                    c_found = (crop_found or "").lower()
                    if "arabica" in c_found:
                        st.session_state.user_data["crop"] = "Arabica"
                    elif "robusta" in c_found:
                        st.session_state.user_data["crop"] = "Robusta"
                    elif any(c in c_found for c in COFFEE_CROPS):
                        st.session_state.user_data["crop"] = "Coffee"
                    
                    # Bridge into onboarding
                    bridge = (
                        f"{response}\n\n"
                        "I've got your soil values! To finish your profile and give you a proper advisory, "
                        "what should I call you?"
                    )
                    st.session_state.messages.append({"role": "assistant", "content": bridge})
                    st.session_state.step = "ask_name"
                else:
                    # PDF read failed or was wrong crop
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.session_state.step = "ask_name"
                st.rerun()

    # ── Variety selectbox ────────────────────────────────────────────────────
    if st.session_state.step == "ask_variety":
        crop = st.session_state.user_data.get("crop", "").strip()

        if crop.lower() not in COFFEE_CROPS:
            st.session_state.user_data["variety"] = "N/A"
            st.session_state.messages.append({
                "role": "assistant",
                "content": (
                    f"Since you're growing **{crop}**, I don't have variety-specific records for that crop. "
                    "Share your **soil values** if you have them (e.g. *'pH 5.5, N 280'*), or type **skip**."
                ),
            })
            st.session_state.step = "ask_soil"
            st.rerun()

        varieties = (
            CROP_VARIETIES["Arabica"] + CROP_VARIETIES["Robusta"]
            if crop.lower() == "coffee"
            else CROP_VARIETIES.get(crop.capitalize(), CROP_VARIETIES["Arabica"])
        )
        st.markdown(f"**Which variety of {crop} are you growing?**")
        var_key = f"variety_select_{st.session_state.get('variety_pick_count', 0)}"
        selected = st.selectbox("Variety", ["-- Select Variety --"] + varieties,
                                label_visibility="collapsed", key=var_key)
        if selected != "-- Select Variety --":
            st.session_state["variety_pick_count"] = st.session_state.get("variety_pick_count", 0) + 1
            st.session_state.messages.append({"role": "user", "content": selected})
            st.session_state.user_data["variety"] = selected
            st.session_state.messages.append({
                "role": "assistant",
                "content": (
                    "Do you have any **soil test values** handy? Type what you know "
                    "(e.g. *'pH 5.5, N 280, Zn 0.6'*) — or just type **skip** to move on."
                ),
            })
            st.session_state.step = "ask_soil"
            st.rerun()

    # ── Text chat (onboarding steps) ─────────────────────────────────────────
    if st.session_state.step not in ("choose_input", "upload_pdf", "ask_variety"):
        if prompt := st.chat_input("Type your message…"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            response = ""
            ud = st.session_state.user_data

            # ── ask_soil step ────────────────────────────────────────────────
            if st.session_state.step == "ask_soil":
                skip_words = {"skip", "don't know", "dont know", "no", "n/a", "-", "none"}
                if prompt.strip().lower() in skip_words:
                    ud["soil_raw"] = "skipped"
                    response = build_completion_message(ud)
                    st.session_state.step = "complete"
                else:
                    try_extract_soil_early(prompt, ud)
                    soil_vals = ud.get("measured_soil", {})
                    if soil_vals:
                        with st.spinner("Analysing soil values…"):
                            advisory = dedup_advisory(soil_advisory(soil_vals, ud))
                        completion = build_completion_message(ud)
                        response = f"{completion}\n\n---\n\n{advisory}"
                        st.session_state.step = "complete"
                    else:
                        response = (
                            "I didn't catch any soil values there. "
                            "Try something like *'pH 5.5, N 280, P 8'*, or type **skip** to carry on."
                        )

            # ── ask_name step ────────────────────────────────────────────────
            elif st.session_state.step == "ask_name":
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
                        with st.spinner("Analysing soil values…"):
                            advisory = dedup_advisory(soil_advisory(soil_vals, ud))
                        response = advisory
                        st.session_state.step = "complete"
                    else:
                        response = "No problem! Let's start with the basics — what should I call you?"
                else:
                    extracted_name = extract_name(prompt)
                    if extracted_name:
                        ud["name"] = extracted_name
                        if is_question(prompt) and not contains_profile_info(prompt):
                            with st.spinner("Checking knowledge base…"):
                                side = answer_side_question(prompt, ud)
                            next_q = f"Thanks, **{extracted_name}**! Now — which district or zone is your farm in?"
                            response = build_response(side, next_q)
                        else:
                            response = f"Thanks, **{extracted_name}**! Which district or zone is your farm in?"
                        st.session_state.step = "ask_location"
                    else:
                        if is_question(prompt) and not contains_profile_info(prompt):
                            with st.spinner("Checking knowledge base…"):
                                response = answer_side_question(prompt, ud)
                            next_q = "What should I call you? (or type **skip** to continue)"
                            response = build_response(response, next_q)
                        else:
                            response = (
                                "I didn't quite catch a name there. "
                                "What should I call you? (or type **skip** to continue)"
                            )

            # ── generic soil/question handler ────────────────────────────────
            elif contains_soil_data(prompt) or (is_question(prompt) and not contains_profile_info(prompt)):
                prefill_profile_from_message(prompt, ud)
                soil_vals = ud.get("measured_soil", {})
                if soil_vals:
                    with st.spinner("Analysing soil values…"):
                        advisory = dedup_advisory(soil_advisory(soil_vals, ud))
                    response = advisory
                else:
                    with st.spinner("Checking knowledge base…"):
                        advisory = answer_side_question(prompt, ud)
                    response = advisory
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()

            # ── ask_location step ────────────────────────────────────────────
            elif st.session_state.step == "ask_location":
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
                                f"{zone_warning}\n\nPick your variety below and we're good to go!"
                            )
                            st.session_state.step = "ask_variety"
                        else:
                            next_q = (
                                f"Noted — **{clean_location}**, **{farm_size} ha**."
                                f"{zone_warning}\n\nWhat are you growing — Arabica, Robusta, or something else?"
                            )
                            st.session_state.step = "ask_crop"
                    else:
                        next_q = (
                            f"Thanks — **{clean_location}** noted.{zone_warning}\n\n"
                            "How large is your farm? _(Enter hectares, or type **skip** if unsure)_"
                        )
                        st.session_state.step = "ask_farm_size"

                    if is_question(prompt) and not contains_profile_info(prompt):
                        with st.spinner("Checking knowledge base…"):
                            side = answer_side_question(prompt, ud)
                        response = build_response(side, next_q)
                    else:
                        response = next_q

            # ── ask_farm_size step ───────────────────────────────────────────
            elif st.session_state.step == "ask_farm_size":
                try_extract_soil_early(prompt, ud)
                if not ud.get("farm_size"):
                    skip_words = {"skip", "don't know", "dont know", "no", "n/a", "-"}
                    if prompt.strip().lower() in skip_words:
                        ud["farm_size"] = "Not provided"
                    else:
                        farm_size = extract_farm_size(prompt)
                        ud["farm_size"] = farm_size or prompt.strip() or "Not provided"

                if ud.get("crop"):
                    next_q = f"You're growing **{ud['crop']}** — which variety? Pick one below."
                    st.session_state.step = "ask_variety"
                else:
                    next_q = "And what are you growing — Arabica, Robusta, or something else?"
                    st.session_state.step = "ask_crop"

                if is_question(prompt) and not contains_profile_info(prompt):
                    with st.spinner("Checking knowledge base…"):
                        side = answer_side_question(prompt, ud)
                    response = build_response(side, next_q)
                else:
                    response = next_q

            # ── ask_crop step ────────────────────────────────────────────────
            elif st.session_state.step == "ask_crop":
                try_extract_soil_early(prompt, ud)
                crop_input = prompt.strip()
                ud["crop"] = crop_input
                if crop_input.lower() in COFFEE_CROPS:
                    next_q = f"Which variety of **{crop_input}** are you growing? Select one below."
                    if is_question(prompt) and not contains_profile_info(prompt):
                        with st.spinner("Checking knowledge base…"):
                            side = answer_side_question(prompt, ud)
                        response = build_response(side, next_q)
                    else:
                        response = next_q
                    st.session_state.step = "ask_variety"
                else:
                    response = (
                        f"I specialise in coffee — Arabica and Robusta — so I don't have advisory data "
                        f"for **{crop_input}**. If you're also growing coffee, I'd be happy to help!"
                    )

            else:
                response = "Apologies, I lost my place! Share your name, location, or crop and I'll pick up from there."

            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

if st.session_state.step == "complete":

    # PDF upload in main chat
    with st.expander("Upload a document (PDF)", expanded=False):
        main_pdf = st.file_uploader(
            "Upload PDF", type=["pdf"],
            label_visibility="collapsed", key="main_chat_pdf_uploader",
        )
        if main_pdf is not None:
            kb_matched, response, crop_found = _handle_pdf_upload(main_pdf, key_prefix="main")
            if response:
                st.session_state.messages.append({"role": "user", "content": f"[Uploaded PDF: {main_pdf.name}]"})
                if kb_matched:
                    # Update crop if found (and not already set)
                    if not st.session_state.user_data.get("crop"):
                        c_found = (crop_found or "").lower()
                        if "arabica" in c_found: st.session_state.user_data["crop"] = "Arabica"
                        elif "robusta" in c_found: st.session_state.user_data["crop"] = "Robusta"
                        elif any(c in c_found for c in COFFEE_CROPS): st.session_state.user_data["crop"] = "Coffee"

                    existing = st.session_state.user_data.get("measured_soil", {})
                    existing.update(kb_matched)
                    st.session_state.user_data["measured_soil"] = existing
                    st.session_state.user_data["soil_raw"] = f"PDF: {main_pdf.name}"
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.rerun()

    # Text chat
    if query := st.chat_input("What would you like to know about your soil?"):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
        with st.chat_message("assistant"):
            with st.spinner("Looking that up for you…"):
                answer = rag_advisory(query, st.session_state.user_data)
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
