#!/usr/bin/env python3
"""
Physics Helper GUI — powered by Claude AI.
Run with:  streamlit run app.py
"""

import base64
import streamlit as st
import anthropic

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Physics Helper",
    page_icon="⚛",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert physics tutor. Your job is to help students solve
physics problems clearly and step-by-step.

When given a problem (typed or from an image):
1. List what is **given** and what needs to be **found**
2. State the relevant formula(s)
3. Substitute values and solve step-by-step
4. Box or clearly state the **final answer with units**

If a result from one equation is needed as input for another, chain them explicitly
and label intermediate results (e.g. "Using vf = 29.4 m/s from Step 1...").

Formatting rules:
- Use **bold** for formulas and key values
- Show each step numbered
- Always include units
- If the problem is ambiguous, state your assumptions

Topics you can handle: kinematics, projectile motion, Newton's laws, friction,
energy, work, power, momentum, impulse, circular motion, gravitation, SHM/waves,
thermodynamics, electrostatics, circuits, optics, special relativity.

If you cannot solve something, explain why and what extra information is needed."""

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
    "g  (gravity)":          "9.80665 m/s²",
    "G  (gravitational)":    "6.674 × 10⁻¹¹ N·m²/kg²",
    "c  (light speed)":      "2.998 × 10⁸ m/s",
    "h  (Planck)":           "6.626 × 10⁻³⁴ J·s",
    "k  (Coulomb)":          "8.988 × 10⁹ N·m²/C²",
    "k_B (Boltzmann)":       "1.381 × 10⁻²³ J/K",
    "R  (gas constant)":     "8.314 J/(mol·K)",
    "e  (elem. charge)":     "1.602 × 10⁻¹⁹ C",
    "N_A (Avogadro)":        "6.022 × 10²³ mol⁻¹",
}

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚛ Physics Helper")
    st.caption("Powered by Claude AI")

    st.divider()

    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        placeholder="sk-ant-...",
        help="Get your free key at console.anthropic.com",
    )

    model = st.selectbox(
        "Model",
        ["claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
        index=0,
        help="Sonnet is smarter; Haiku is faster.",
    )

    st.divider()

    # Formula quick reference
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
    st.session_state.messages = []        # list of {role, text, image_b64, image_type}

if "pending_image" not in st.session_state:
    st.session_state.pending_image = None  # {b64, media_type, display_bytes}

# ── Main area ─────────────────────────────────────────────────────────────────

st.title("⚛ Physics Helper")
st.caption(
    "Type a question **or** upload a photo of your problem — I'll solve it step by step."
)

# ── Image uploader (sits above chat, clears after use) ────────────────────────

uploaded = st.file_uploader(
    "📷 Attach a problem image (optional)",
    type=["png", "jpg", "jpeg", "webp", "gif"],
    label_visibility="visible",
    key="uploader",
)

if uploaded is not None:
    raw = uploaded.read()
    st.session_state.pending_image = {
        "b64": base64.standard_b64encode(raw).decode(),
        "media_type": uploaded.type,
        "display_bytes": raw,
    }
    st.image(raw, caption="Image attached — ask your question below", width=400)

# ── Render conversation history ───────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("image_bytes"):
            st.image(msg["image_bytes"], width=300)
        st.markdown(msg["text"])

# ── Chat input & response ─────────────────────────────────────────────────────

prompt = st.chat_input("Ask a physics question…")

if prompt:
    # Validate API key
    if not api_key:
        st.error("⚠️ Please enter your Anthropic API key in the sidebar to get started.")
        st.stop()

    # ── Show user message ──────────────────────────────────────────────────────
    img_info = st.session_state.pending_image
    with st.chat_message("user"):
        if img_info:
            st.image(img_info["display_bytes"], width=300)
        st.markdown(prompt)

    # Save to history
    st.session_state.messages.append({
        "role": "user",
        "text": prompt,
        "image_bytes": img_info["display_bytes"] if img_info else None,
    })

    # ── Build API message list ─────────────────────────────────────────────────
    api_messages = []

    for past in st.session_state.messages[:-1]:   # all but current
        if past["role"] == "user":
            api_messages.append({"role": "user", "content": past["text"]})
        else:
            api_messages.append({"role": "assistant", "content": past["text"]})

    # Current user turn (may include image)
    current_content = []
    if img_info:
        current_content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": img_info["media_type"],
                "data": img_info["b64"],
            },
        })
    current_content.append({"type": "text", "text": prompt})
    api_messages.append({"role": "user", "content": current_content})

    # Clear pending image after use
    st.session_state.pending_image = None

    # ── Stream the response ────────────────────────────────────────────────────
    client = anthropic.Anthropic(api_key=api_key)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        try:
            with client.messages.stream(
                model=model,
                max_tokens=2048,
                system=SYSTEM_PROMPT,
                messages=api_messages,
            ) as stream:
                for chunk in stream.text_stream:
                    full_response += chunk
                    placeholder.markdown(full_response + "▌")

            placeholder.markdown(full_response)

        except anthropic.AuthenticationError:
            placeholder.error("❌ Invalid API key. Check the key in the sidebar.")
            st.stop()
        except anthropic.RateLimitError:
            placeholder.error("❌ Rate limit hit. Wait a moment and try again.")
            st.stop()
        except Exception as exc:
            placeholder.error(f"❌ Error: {exc}")
            st.stop()

    # Save assistant reply
    st.session_state.messages.append({
        "role": "assistant",
        "text": full_response,
        "image_bytes": None,
    })
