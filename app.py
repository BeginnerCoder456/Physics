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
    page_title="Physics Helper",
    page_icon="⚛",
    layout="centered",
    initial_sidebar_state="expanded",
)

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
In 1-2 sentences, explain the core physics concept behind this problem. Connect it to something intuitive if possible (e.g. "When something decelerates, it's still accelerating — just in the opposite direction to motion").

**📋 What We Know / What We Need**
List every given value with symbol, number, and unit.
State clearly what the question is asking for.

**🔧 The Right Formula — and Why**
State the formula. Then explain in plain English why this formula applies to this situation. Don't just say "use F = ma" — explain what F, m, and a represent in THIS problem.

**📐 Solving Step by Step**
Number every step. Show the algebra before plugging in numbers.
Carry units through every step.
When one result feeds into the next equation, say so explicitly: "We found v = 12 m/s above, so now..."

**✅ Answer + Sanity Check**
State the final answer clearly with units.
Then briefly confirm it makes sense: is the number reasonable? Does the unit match?

**💡 Key Takeaway**
One sentence the student should remember from this problem — a rule of thumb, a common mistake to avoid, or a deeper insight.

**❓ Check Your Understanding**
Ask the student one follow-up question that tests whether they grasped the concept (not just the calculation). Make it feel natural, not like a test.

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
        "p = m·v  (momentum)",
        "J = F·Δt = Δp  (impulse)",
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
        "V = I·R  (Ohm's law)",
        "P = I·V = I²R = V²/R",
        "C = Q/V",
    ],
    "Optics": [
        "1/f = 1/do + 1/di",
        "M = −di/do",
        "n₁·sin θ₁ = n₂·sin θ₂",
        "E = h·f  (photon energy)",
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

# Text-only models (fast, high quality)
TEXT_MODELS = [
    "llama-3.3-70b-versatile",
    "llama3-8b-8192",
]

# Vision model used automatically when an image is attached
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚛ Physics Helper")
    st.caption("Powered by Groq AI (free)")

    st.divider()

    api_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Free key at console.groq.com",
    )

    model = st.selectbox(
        "Model (text)",
        TEXT_MODELS,
        index=0,
        help="Used for text questions. Vision model is selected automatically for images.",
    )

    st.divider()

    st.subheader("📐 Formula Sheets")
    for topic, formulas in FORMULA_SHEETS.items():
        with st.expander(topic):
            for f in formulas:
                st.code(f, language=None)

    st.divider()

    st.subheader("🔬 Constants")
    for name, val in CONSTANTS.items():
        st.markdown(f"**{name}** = {val}")

    st.divider()

    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.session_state.pending_image = None
        st.rerun()

# ── Session state ─────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_image" not in st.session_state:
    st.session_state.pending_image = None

# ── Main area ─────────────────────────────────────────────────────────────────

st.title("⚛ Physics Helper")
st.caption("Type a question **or** upload a photo of your problem — I'll solve it step by step.")

# ── Image uploader ────────────────────────────────────────────────────────────

uploaded = st.file_uploader(
    "📷 Attach a problem image (optional)",
    type=["png", "jpg", "jpeg", "webp"],
    label_visibility="visible",
)

if uploaded is not None:
    raw = uploaded.read()
    st.session_state.pending_image = {
        "b64": base64.standard_b64encode(raw).decode(),
        "media_type": uploaded.type,
        "display_bytes": raw,
    }
    st.image(raw, caption="Image attached — ask your question below", width=400)

# ── Conversation history ──────────────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("image_bytes"):
            st.image(msg["image_bytes"], width=300)
        st.markdown(msg["text"])

# ── Chat input & response ─────────────────────────────────────────────────────

prompt = st.chat_input("Ask a physics question…")

if prompt:
    if not api_key:
        st.error("⚠️ Enter your Groq API key in the sidebar. Get one free at console.groq.com")
        st.stop()

    img_info = st.session_state.pending_image
    active_model = VISION_MODEL if img_info else model

    # Show user message
    with st.chat_message("user"):
        if img_info:
            st.image(img_info["display_bytes"], width=300)
        st.markdown(prompt)

    st.session_state.messages.append({
        "role": "user",
        "text": prompt,
        "image_bytes": img_info["display_bytes"] if img_info else None,
    })

    # Build Groq message list
    api_messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for past in st.session_state.messages[:-1]:
        api_messages.append({"role": past["role"], "content": past["text"]})

    # Current turn — include image if present (Groq uses OpenAI image_url format)
    if img_info:
        data_url = f"data:{img_info['media_type']};base64,{img_info['b64']}"
        image_prefix = (
            "Before solving, carefully read this image and transcribe EVERY value, "
            "label, unit, and piece of information you can see — including any diagrams, "
            "arrows, subscripts, and superscripts. Do not assume or invent values that "
            "are not clearly shown. Then solve the problem below.\n\n"
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

    with st.chat_message("assistant"):
        if img_info:
            st.caption(f"Using vision model: {VISION_MODEL}")
        placeholder = st.empty()
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
                placeholder.error("❌ Invalid API key — check the key in the sidebar.")
            elif "rate" in err.lower():
                placeholder.error("❌ Rate limit hit — wait a moment and try again.")
            else:
                placeholder.error(f"❌ Error: {exc}")
            st.stop()

    st.session_state.messages.append({
        "role": "assistant",
        "text": full_response,
        "image_bytes": None,
    })
