import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from datetime import datetime

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from src.rag.naive_rag import get_naive_rag

graph = get_naive_rag()

st.set_page_config(page_title="Ask me anything about the Dragi report", layout="centered")

st.title("RAG Agent")
st.markdown("Ask me general questions about the report, e.g., `What did report say about economics?`")

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Option to clear chat
col1, col2 = st.columns([0.7, 0.3])
with col2:
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

# --- Message area ---
st.markdown('<div class="chat-area">', unsafe_allow_html=True)
for message in st.session_state.messages:
    role = message["role"]
    timestamp = message.get("time", "")
    content = message["content"]
    bubble_class = "chat-bubble-user" if role == "user" else "chat-bubble-ai"
    icon = "🧑‍💻" if role == "user" else "🤖"
    st.markdown(
        f'<div class="{bubble_class}">{icon} {content}</div>'
        f'<div class="chat-time">{timestamp}</div>' if timestamp else "",
        unsafe_allow_html=True,
    )
st.markdown('</div>', unsafe_allow_html=True)


# --- User input ---
if prompt := st.chat_input("Type your message and press Enter..."):
    timestamp = datetime.now().strftime("%H:%M")
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "time": timestamp
    })
    # Prepare message objects for agent
    chat_history = []
    for message in st.session_state.messages:
        if message["role"] == "user":
            chat_history.append(HumanMessage(content=message["content"]))
        else:
            chat_history.append(AIMessage(content=message["content"]))

    # Agent response
    with st.chat_message("assistant"):
        with st.spinner("Let me think..."):
            response = get_naive_rag().invoke({"question": prompt})
            ai_message = response["messages"][-1]
            content = ai_message.content
            st.markdown(content)
    timestamp = datetime.now().strftime("%H:%M")
    st.session_state.messages.append({
        "role": "assistant",
        "content": content,
        "time": timestamp
    })