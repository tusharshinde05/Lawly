import streamlit as st
import os
import sys
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from models.llm import get_chatgroq_model
from utils.search import web_search
from models.embeddings import embed_and_retrieve


# ----------------------------
# Chat Logic with RAG + Web Fallback
# ----------------------------
def get_chat_response(chat_model, messages, system_prompt):
    try:
        # Format chat history
        formatted_messages = [SystemMessage(content=system_prompt)]
        for msg in messages:
            if msg["role"] == "user":
                formatted_messages.append(HumanMessage(content=msg["content"]))
            else:
                formatted_messages.append(AIMessage(content=msg["content"]))

        user_query = messages[-1]["content"] if messages else ""
        rag_context = ""
        search_context = ""

        # Step 1: Try RAG (local documents)
        if user_query.strip():
            rag_context = embed_and_retrieve(user_query)

        full_prompt = f"{system_prompt}\n\nContext:\n{rag_context}" if rag_context else system_prompt

        # Step 2: First response attempt (Groq + RAG)
        response = chat_model.invoke([SystemMessage(content=full_prompt)] + formatted_messages[1:])
        response_text = response.content.strip()

        # Step 3: If Groq fails / uncertain -> Web Search fallback
        if not response_text or response_text.lower() in ["i don't know", "not sure", "unknown"]:
            search_results = web_search(user_query)
            search_context = "\n".join(search_results)
            fallback_prompt = f"{system_prompt}\n\nContext:\n{rag_context}\n{search_context}"
            response = chat_model.invoke([SystemMessage(content=fallback_prompt)] + formatted_messages[1:])
            response_text = response.content.strip()

        return response_text

    except Exception as e:
        return f"Error getting response: {str(e)}"


# ----------------------------
# Streamlit Pages
# ----------------------------
def instructions_page():
    st.title("The Chatbot Blueprint")
    st.markdown("""
        1. Upload your knowledge base documents in the sidebar.  
        2. Ask questions in chat — bot will first use your docs.  
        3. If docs don’t have the answer, bot will try web search.  
    """)


def chat_page():
    st.markdown("<h1>🤖 Counselly</h1>", unsafe_allow_html=True)
    # Subtitle in smaller font
    st.markdown("<h4 style='color: gray;'>Legal++</h4>", unsafe_allow_html=True)

    st.caption(f"📝 Response Mode: {st.session_state.response_mode}")

    # Build system prompt
    if st.session_state.response_mode == "Concise":
        system_prompt = "You are a helpful assistant. Keep answers short and summarized."
    else:
        system_prompt = "You are a helpful assistant. Provide expanded, detailed explanations."

    if "messages" not in st.session_state:
        st.session_state.messages = []

    chat_model = get_chatgroq_model()

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Getting response..."):
                response = get_chat_response(chat_model, st.session_state.messages, system_prompt)
                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})


# ----------------------------
# Streamlit Main
# ----------------------------
def main():
    st.set_page_config(
        page_title="LangChain ChatBot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Sidebar navigation
    with st.sidebar:
        st.title("Navigation")
        page = st.radio("Go to:", ["Chat", "Instructions"], index=0)

        response_mode = st.radio(
            "Response Mode:",
            ["Concise", "Detailed"],
            index=0,
            help="Choose how you want the AI to respond."
        )
        st.session_state.response_mode = response_mode

        st.divider()
        st.subheader("📂 Upload Knowledge Base")
        uploaded_files = st.file_uploader("Upload PDFs, TXTs, or CSVs", type=["pdf", "txt", "csv"], accept_multiple_files=True)

        if uploaded_files:
            from models.embeddings import add_documents
            add_documents(uploaded_files)
            st.success(f"✅ {len(uploaded_files)} documents added to knowledge base!")

        if page == "Chat":
            st.divider()
            if st.button("🗑️ Clear Chat History", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

    # Page routing
    if page == "Instructions":
        instructions_page()
    else:
        chat_page()


if __name__ == "__main__":
    main()