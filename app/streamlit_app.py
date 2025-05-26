import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from datetime import datetime
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from src.rag.agentic_rag import build_agentic_rag_graph, run_agentic_rag
from src.vectorestore.chroma_client import get_vectorestore

# Page config
st.set_page_config(
    page_title="Dragi Report Chat",
    layout="wide",  # use full-width for more breathing room
    initial_sidebar_state="auto",
)

# load and cache vectorestore
@st.cache_resource
def get_vectorestore_cached():
    return get_vectorestore()


if "vectorestore" not in st.session_state:
    st.session_state.vectorestore = get_vectorestore_cached()

if "rag_graph" not in st.session_state:
    st.session_state.rag_graph = build_agentic_rag_graph(st.session_state.vectorestore)

graph = st.session_state.rag_graph

# Build your RAG graph once
#graph = build_agentic_rag_graph()


# Sidebar for settings / about
with st.sidebar:
    st.header("⚙️ Settings")
    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = []
        st.rerun()
    st.markdown("---")
    st.markdown(
        """
    **About**  
    Ask me anything about the Dragi report.  
    Powered by a Retrieval-Augmented Generation agent.
    """
    )

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Main header
st.markdown(
    """
    <h1 style="text-align: center; color: #333;">🗨️ Dragi Report Assistant</h1>
    <p style="text-align: center; color: #666;">Ask any question about the report (e.g. “What does it say about inflation?”)</p>
    """,
    unsafe_allow_html=True,
)

# Custom CSS for chat bubbles
st.markdown(
    """
    <style>
    .chat-container {
        background: #f7f7f7;
        padding: 1rem;
        border-radius: 8px;
        height: 60vh;
        overflow-y: auto;
        box-shadow: inset 0 0 5px rgba(0,0,0,0.1);
    }
    .chat-bubble {
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        max-width: 80%;
        border-radius: 12px;
        line-height: 1.4;
    }
    .user {
        background: #007bff;
        color: white;
        margin-left: auto;
        border-bottom-right-radius: 0;
    }
    .assistant {
        background: #e2e3e5;
        color: #333;
        margin-right: auto;
        border-bottom-left-radius: 0;
    }
    .timestamp {
        font-size: 0.75rem;
        color: #999;
        margin-top: -0.25rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Chat display area
chat_container = st.container()
with chat_container:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"]
        ts = msg.get("time", "")
        bubble_class = "user" if role == "user" else "assistant"
        icon = "👤" if role == "user" else "🤖"
        st.markdown(
            f'''
            <div class="chat-bubble {bubble_class}">
              <strong>{icon}</strong> {content}
              <div class="timestamp">{ts}</div>
            </div>
            ''',
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)

# User input at bottom
user_input = st.chat_input("Type your question here…")
if user_input:
    # add user message
    timestamp = datetime.now().strftime("%H:%M")
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "time": timestamp
    })

    # build history for agent call
    history = []
    for m in st.session_state.messages:
        if m["role"] == "user":
            history.append(HumanMessage(content=m["content"]))
        else:
            history.append(AIMessage(content=m["content"]))

    # run agent and append response
    with st.spinner("Thinking…"):
        reply = run_agentic_rag(graph, {"messages": history})
    timestamp = datetime.now().strftime("%H:%M")
    st.session_state.messages.append({
        "role": "assistant",
        "content": reply,
        "time": timestamp
    })

    # rerun to display the new messages
    st.rerun()
