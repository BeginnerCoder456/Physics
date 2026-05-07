#!/usr/bin/env python3
"""
Physics Helper GUI — powered by Groq AI.
Run with:  streamlit run app.py
"""

import base64
import streamlit as st
from groq import Groq

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Physics AI",
    page_icon="⚛",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS — Claude-style dark theme ──────────────────────────────────────

st.markdown("""
<style>
/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #1a1a1a;
    color: #e8e8e6;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

/* ── Hide default streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background-color: #111111;
    border-right: 1px solid #2a2a2a;
}
[data-testid="stSidebar"] * { color: #c8c8c6 !important; }
[data-testid="stSidebar"] .stTextInput input {
    background-color: #1e1e1e !important;
    border: 1px solid #3a3a3a !important;
    color: #e8e8e6 !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] .stSelectbox > div > div {
    background-color: #1e1e1e !important;
    border: 1px solid #3a3a3a !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] .stButton > button {
    background-color: #2a2a2a !important;
    border: 1px solid #3a3a3a !important;
    color: #c8c8c6 !important;
    border-radius: 8px !important;
    width: 100%;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #333 !important;
    border-color: #555 !important;
}
[data-testid="stSidebar"] .stExpander {
    border: 1px solid #2a2a2a !important;
    border-radius: 8px !important;
    background-color: #1a1a1a !important;
}

/* ── Main content area ── */
[data-testid="stMainBlockContainer"] {
    max-width: 780px;
    padding: 0 1.5rem 8rem 1.5rem;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background-color: transparent !important;
    border: none !important;
    padding: 0.2rem 0 !important;
}

/* User bubble */
[data-testid="stChatMessage"][data-testid*="user"],
[data-testid="stChatMessage"]:has([aria-label="user avatar"]) {
    background-color: transparent !important;
}

/* ── Chat input bar ── */
[data-testid="stChatInput"] {
    background-color: #2a2a2a !important;
    border: 1px solid #3d3d3d !important;
    border-radius: 12px !important;
}
[data-testid="stChatInput"] textarea {
    background-color: transparent !important;
    color: #e8e8e6 !important;
    font-size: 0.95rem !important;
}
[data-testid="stChatInput"] button {
    background-color: #c96442 !important;
    border-radius: 8px !important;
}
[data-testid="stChatInput"] button:hover {
    background-color: #e07050 !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background-color: #1e1e1e !important;
    border: 1px dashed #3a3a3a !important;
    border-radius: 10px !important;
    padding: 0.5rem 1rem !important;
}
[data-testid="stFileUploader"] * { color: #888 !important; }
[data-testid="stFileUploader"] button {
    background-color: #2a2a2a !important;
    border: 1px solid #444 !important;
    color: #bbb !important;
    border-radius: 6px !important;
}

/* ── Code blocks ── */
code, pre {
    background-color: #242424 !important;
    border: 1px solid #333 !important;
    border-radius: 6px !important;
    color: #d4a96a !important;
    font-size: 0.87rem !important;
}

/* ── Headings in assistant messages ── */
h1, h2, h3 { color: #e8e8e6 !important; }

/* ── Dividers ── */
hr { border-color: #2a2a2a !important; }

/* ── Caption / small text ── */
.stCaption, small { color: #666 !important; }

/* ── Welcome card ── */
.welcome-card {
    background: linear-gradient(135deg, #1e1e1e 0%, #252525 100%);
    border: 1px solid #2a2a2a;
    border-radius: 16px;
    padding: 2.5rem;
    text-align: center;
    margin: 4rem auto 2rem auto;
    max-width: 540px;
}
.welcome-card h2 {
    font-size: 1.6rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
    color: #e8e8e6 !important;
}
.welcome-card p {
    color: #888;
    font-size: 0.95rem;
    line-height: 1.6;
    margin-bottom: 1.5rem;
}
.pill-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    justify-content: center;
}
.pill {
    background-color: #2a2a2a;
    border: 1px solid #3a3a3a;
    border-radius: 20px;
    padding: 0.35rem 0.85rem;
    font-size: 0.82rem;
    color: #aaa;
    cursor: default;
}

/* ── Thinking indicator ── */
.thinking {
    color: #666;
    font-size: 0.85rem;
    font-style: italic;
    padding: 0.4rem 0;
}
</style>
""", unsafe_allow_html=True)

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a patient, encouraging physics tutor who genuinely wants students to understand physics — not just get the answer.

═══════════════════════════════════
ACCURACY — THIS IS NON-NEGOTIABLE
═══════════════════════════════════
- Before solving anything, write out EVERY piece of given information with its exact value and unit.
- If solving from an image: read the image meticulously. Transcribe every number, symbol, arrow, label, and unit you can see. Do not guess or assume values that aren't shown.
- Carry units through every single calculation step. Never drop them.
- After getting an answer, sanity-check it: Does the magnitude make physical sense? Does the unit match what was asked?
- If you notice you made an arithmetic error, correct it immediately and explain what went wrong.
- Double-check any squared terms, square roots, and sign conventions (especially for direction, potential energy, and deceleration).

FORCE AND ACCELERATION PROBLEMS — EXTRA CARE REQUIRED:
- Always draw a text free-body diagram listing every force acting on the object with its direction (+ or −).
- Write the complete ΣF = ma equation with EVERY force substituted before doing any algebra. Do not skip straight to numbers.
- For connected objects (Atwood, blocks on inclines with ropes, pulleys): write a separate ΣF = ma equation for EACH object, then solve the system.
- For tension: after finding T, verify it is consistent by substituting back into both objects' equations.
- Never combine the net force step with the arithmetic step — keep them separate so errors are visible.
- After computing acceleration, plug it back into one of the original force equations to confirm the numbers are self-consistent.

═══════════════════════════════════
TEACHING STYLE
═══════════════════════════════════
You teach physics, you don't just calculate it. Every response should help the student understand WHY, not just WHAT.

Structure every response like this:

**🧠 The Big Idea**
In 1-2 sentences, explain the core physics concept behind this problem. Connect it to something intuitive if possible.

**📋 What We Know / What We Need**
List every given value with symbol, number, and unit.
State clearly what the question is asking for.

**🔧 The Right Formula — and Why**
State the formula. Then explain in plain English why this formula applies to this situation.

**📐 Solving Step by Step**
Number every step. Show the algebra before plugging in numbers.
Carry units through every step.
When one result feeds into the next equation, say so explicitly.

**✅ Answer + Sanity Check**
State the final answer clearly with units.
Then briefly confirm it makes sense: is the number reasonable? Does the unit match?

**💡 Key Takeaway**
One sentence the student should remember from this problem.

**❓ Check Your Understanding**
Ask the student one follow-up question that tests whether they grasped the concept. Make it feel natural, not like a test.

═══════════════════════════════════
TONE
═══════════════════════════════════
- Be warm and encouraging. Physics is hard and students get frustrated.
- Never just dump the answer. Guide them through the reasoning.
- If they got something wrong in a previous attempt, gently point out where the thinking went astray.
- Use analogies and everyday examples when introducing concepts.
- If a question is vague or missing information, ask a specific clarifying question instead of guessing.

Topics: kinematics, projectile motion, Newton's laws, friction, energy, work, power, momentum, impulse, circular motion, gravitation, waves, SHM, thermodynamics, electrostatics, circuits, optics, special relativity."""

# ── Quick-reference data ──────────────────────────────────────────────────────

FORMULA_SHEETS = {
    "Kinematics": [
        "vf = vi + a·t",
        "d = vi·t + ½a·t²",
        "vf² = vi² + 2·a·d",
        "d = (vi + vf)/2 · t",
    ],
    "Forces & Newton's Laws": [
        "F = m·a",
        "Fg = m·g  (g = 9.8 m/s²)",
        "Ff = μ·N",
        "p = m·v",
        "J = F·Δt = Δp",
    ],
    "Energy & Work": [
        "W = F·d·cos θ",
        "KE = ½·m·v²",
        "PE = m·g·h",
        "PE_spring = ½·k·x²",
        "P = W/t = F·v",
    ],
    "Circular & Rotation": [
        "ac = v²/r",
        "Fc = m·v²/r",
        "v = r·ω",
        "T = 2π/ω = 1/f",
        "τ = r·F·sin θ",
    ],
    "Waves & SHM": [
        "v = f·λ",
        "T_spring = 2π√(m/k)",
        "T_pendulum = 2π√(L/g)",
        "ω = √(k/m)",
    ],
    "Thermodynamics": [
        "PV = nRT",
        "Q = m·c·ΔT",
        "ΔU = Q − W",
        "e_Carnot = 1 − Tc/Th",
    ],
    "Electricity": [
        "F = k·q₁·q₂/r²",
        "V = I·R",
        "P = I·V = I²R = V²/R",
        "C = Q/V",
    ],
    "Optics": [
        "1/f = 1/do + 1/di",
        "M = −di/do",
        "n₁·sin θ₁ = n₂·sin θ₂",
        "E = h·f",
    ],
}

CONSTANTS = {
    "g  (gravity)":       "9.80665 m/s²",
    "G  (gravitational)": "6.674 × 10⁻¹¹ N·m²/kg²",
    "c  (light speed)":   "2.998 × 10⁸ m/s",
    "h  (Planck)":        "6.626 × 10⁻³⁴ J·s",
    "k  (Coulomb)":       "8.988 × 10⁹ N·m²/C²",
    "k_B (Boltzmann)":    "1.381 × 10⁻²³ J/K",
    "R  (gas constant)":  "8.314 J/(mol·K)",
    "e  (elem. charge)":  "1.602 × 10⁻¹⁹ C",
    "N_A (Avogadro)":     "6.022 × 10²³ mol⁻¹",
}

TEXT_MODELS   = ["llama-3.3-70b-versatile", "llama3-8b-8192"]
VISION_MODEL  = "meta-llama/llama-4-scout-17b-16e-instruct"

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚛ Physics AI")
    st.caption("Powered by Groq — free")
    st.divider()

    api_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Free at console.groq.com",
    )

    model = st.selectbox(
        "Model",
        TEXT_MODELS,
        index=0,
        help="Vision model auto-selected when you attach an image.",
    )

    st.divider()
    st.markdown("**📐 Formulas**")
    for topic, formulas in FORMULA_SHEETS.items():
        with st.expander(topic, expanded=False):
            for f in formulas:
                st.code(f, language=None)

    st.divider()
    st.markdown("**🔬 Constants**")
    for name, val in CONSTANTS.items():
        st.markdown(f"<small><b>{name}</b> = {val}</small>", unsafe_allow_html=True)

    st.divider()
    if st.button("🗑  New conversation"):
        st.session_state.messages = []
        st.session_state.pending_image = None
        st.rerun()

# ── Session state ─────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_image" not in st.session_state:
    st.session_state.pending_image = None

# ── Main area ─────────────────────────────────────────────────────────────────

# Welcome screen when no messages yet
if not st.session_state.messages:
    st.markdown("""
    <div class="welcome-card">
        <h2>⚛ Physics AI</h2>
        <p>Ask me any physics question or upload a photo of your problem.<br>
        I'll walk you through the concept, the method, and the maths.</p>
        <div class="pill-row">
            <span class="pill">Kinematics</span>
            <span class="pill">Newton's Laws</span>
            <span class="pill">Energy & Work</span>
            <span class="pill">Circuits</span>
            <span class="pill">Waves</span>
            <span class="pill">Thermodynamics</span>
            <span class="pill">Optics</span>
            <span class="pill">Relativity</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Image uploader ────────────────────────────────────────────────────────────

uploaded = st.file_uploader(
    "📷  Attach a problem image",
    type=["png", "jpg", "jpeg", "webp"],
    label_visibility="collapsed",
)

if uploaded is not None:
    raw = uploaded.read()
    st.session_state.pending_image = {
        "b64": base64.standard_b64encode(raw).decode(),
        "media_type": uploaded.type,
        "display_bytes": raw,
    }
    st.image(raw, caption="Attached — type your question below", width=380)

# ── Render conversation history ───────────────────────────────────────────────

for msg in st.session_state.messages:
    avatar = "⚛" if msg["role"] == "assistant" else "👤"
    with st.chat_message(msg["role"], avatar=avatar):
        if msg.get("image_bytes"):
            st.image(msg["image_bytes"], width=280)
        st.markdown(msg["text"])

# ── Chat input ────────────────────────────────────────────────────────────────

prompt = st.chat_input("Ask a physics question…  (or attach an image above)")

if prompt:
    if not api_key:
        st.error("Enter your Groq API key in the sidebar. Free at console.groq.com")
        st.stop()

    img_info       = st.session_state.pending_image
    active_model   = VISION_MODEL if img_info else model

    # Show user message
    with st.chat_message("user", avatar="👤"):
        if img_info:
            st.image(img_info["display_bytes"], width=280)
        st.markdown(prompt)

    st.session_state.messages.append({
        "role": "user",
        "text": prompt,
        "image_bytes": img_info["display_bytes"] if img_info else None,
    })

    # Build message list for Groq
    api_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for past in st.session_state.messages[:-1]:
        api_messages.append({"role": past["role"], "content": past["text"]})

    if img_info:
        data_url     = f"data:{img_info['media_type']};base64,{img_info['b64']}"
        image_prefix = (
            "Carefully read this image and transcribe EVERY value, label, unit, "
            "and piece of information visible — including diagrams, arrows, subscripts, "
            "and superscripts. Do not invent values not shown. Then solve:\n\n"
        )
        current_content = [
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": image_prefix + prompt},
        ]
    else:
        current_content = prompt

    api_messages.append({"role": "user", "content": current_content})
    st.session_state.pending_image = None

    # Stream response
    client = Groq(api_key=api_key)

    with st.chat_message("assistant", avatar="⚛"):
        placeholder   = st.empty()
        full_response = ""

        try:
            stream = client.chat.completions.create(
                model=active_model,
                messages=api_messages,
                max_tokens=4096,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    full_response += delta
                    placeholder.markdown(full_response + "▌")
            placeholder.markdown(full_response)

        except Exception as exc:
            err = str(exc)
            if "401" in err or "invalid_api_key" in err.lower() or "authentication" in err.lower():
                placeholder.error("❌ Invalid API key — check the sidebar.")
            elif "rate" in err.lower():
                placeholder.error("❌ Rate limit hit — wait a moment and try again.")
            elif "decommissioned" in err.lower() or "model" in err.lower():
                placeholder.error(f"❌ Model error: {exc}\nTry switching the model in the sidebar.")
            else:
                placeholder.error(f"❌ {exc}")
            st.stop()

    st.session_state.messages.append({
        "role": "assistant",
        "text": full_response,
        "image_bytes": None,
    })
