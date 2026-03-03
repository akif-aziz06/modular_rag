"""
app.py
------
Streamlit frontend for the Modular Salesforce RAG Chatbot.
Run with: streamlit run app.py
"""

import streamlit as st
from orchestrator import orchestrate_query

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Salesforce Expert AI",
    page_icon="☁️",
    layout="centered",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ─ White background ─ */
.stApp {
    background: #f0f4f9 !important;
    min-height: 100vh;
}

/* ─ Main content area ─ */
.main .block-container {
    background: transparent !important;
}

/* ─ Header card ─ */
.header-card {
    background: linear-gradient(135deg, #0070D2 0%, #005fb2 100%);
    border-radius: 18px;
    padding: 28px 36px;
    margin-bottom: 24px;
    text-align: center;
    box-shadow: 0 6px 24px rgba(0, 112, 210, 0.22);
}

.header-card h1 {
    color: #ffffff;
    font-size: 2rem;
    font-weight: 700;
    margin: 0;
}

.header-card p {
    color: rgba(255,255,255,0.8);
    margin-top: 6px;
    font-size: 0.95rem;
}

/* ─ Chat messages ─ */
.stChatMessage {
    background: #ffffff !important;
    border: 1px solid #dde3ea !important;
    border-radius: 14px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05) !important;
    margin-bottom: 10px !important;
}

/* ─ Force dark text inside messages ─ */
.stChatMessage p, .stChatMessage li, .stChatMessage span {
    color: #1a2332 !important;
}

/* ─ Chat input ─ */
.stChatInputContainer, div[data-testid="stChatInput"] {
    background: #ffffff !important;
    border: 1.5px solid #0070D2 !important;
    border-radius: 14px !important;
    box-shadow: 0 2px 12px rgba(0, 112, 210, 0.1) !important;
}

/* ─ Sidebar ─ */
section[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #dde3ea !important;
}

section[data-testid="stSidebar"] * {
    color: #1a2332 !important;
}

/* ─ Sidebar stat card ─ */
.sidebar-stat {
    background: #eef4ff;
    border: 1px solid #c7dcf7;
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 10px;
    color: #1a2332;
    font-size: 0.85rem;
}

.sidebar-stat strong {
    color: #00875a;
    display: block;
    font-size: 1.2rem;
    margin-bottom: 2px;
}

/* ─ Buttons ─ */
.stButton > button {
    background: #eef4ff !important;
    border: 1.5px solid #0070D2 !important;
    color: #0070D2 !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    background: #0070D2 !important;
    color: #ffffff !important;
}

/* ─ Scrollbar ─ */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #f0f4f9; }
::-webkit-scrollbar-thumb { background: #c7dcf7; border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: #0070D2; }
</style>
""", unsafe_allow_html=True)

# ── Session State ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []          # Displayed messages  [{role, content}]

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []      # LLM memory buffer   [{role, content}]

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ☁️ Salesforce Expert AI")
    st.markdown("---")

    st.markdown("### Supported Modes")
    modes = {
        "🔧 Admin Mode": "Setup, configuration, declarative features",
        "💻 Dev Mode": "Apex, LWC, Triggers, SOQL",
        "📐 Consultant Mode": "Architecture, best practices",
        "🎓 Interview Mode": "Cert prep, recruiter answers",
        "🎮 Interactive Mode": "Quizzes, step-by-step tutorials",
    }
    for mode, desc in modes.items():
        st.markdown(f"**{mode}**  \n<span style='color:rgba(255,255,255,0.5);font-size:0.8rem'>{desc}</span>", unsafe_allow_html=True)
        st.markdown("")

    st.markdown("---")

    # Stats
    total_msgs  = len([m for m in st.session_state.messages if m["role"] == "user"])
    st.markdown(f"<div class='sidebar-stat'><strong>{total_msgs}</strong>Questions asked this session</div>", unsafe_allow_html=True)

    # Clear button
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages    = []
        st.session_state.chat_history = []
        st.rerun()

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-card">
    <h1>☁️ Salesforce Expert AI</h1>
    <p>Powered by a Modular RAG Pipeline · Groq Llama 3.1 · ChromaDB</p>
</div>
""", unsafe_allow_html=True)

# ── Chat Messages ──────────────────────────────────────────────────────────────
# Display welcome message if no history
if not st.session_state.messages:
    with st.chat_message("assistant", avatar="☁️"):
        st.markdown(
            "👋 **Hello! I'm your Salesforce Expert AI.**\n\n"
            "I can help you with:\n"
            "- 🔧 Admin setup and configuration\n"
            "- 💻 Apex, LWC, and development\n"
            "- 📐 Architecture and consulting\n"
            "- 🎓 Interview and certification prep\n\n"
            "Ask me anything Salesforce-related!"
        )

# Render all stored messages
for msg in st.session_state.messages:
    avatar = "☁️" if msg["role"] == "assistant" else "🧑"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# ── Chat Input ─────────────────────────────────────────────────────────────────
if user_input := st.chat_input("Ask a Salesforce question..."):

    # Display user message
    with st.chat_message("user", avatar="🧑"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Generate response
    with st.chat_message("assistant", avatar="☁️"):
        with st.spinner("Thinking..."):
            response = orchestrate_query(user_input, st.session_state.chat_history)
        st.markdown(response)

    # Store assistant response
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Update LLM memory buffer
    st.session_state.chat_history.append({"role": "user",      "content": user_input})
    st.session_state.chat_history.append({"role": "assistant",  "content": response})
