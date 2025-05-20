import streamlit as st
import os
import numpy as np
from utils import (
    extract_text_from_pdf,
    chunk_text,
    embed_chunks,
    build_faiss_index,
    search_chunks,
    ask_groq
)

# Page setup
st.set_page_config(page_title="Chat with PDF + Summary", layout="wide")
st.title("📄 Chat with PDF - RAG powered by LlAMA via Llama")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "user_input" not in st.session_state:
    st.session_state.user_input = ""
if "text" not in st.session_state:
    st.session_state.text = None
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "model" not in st.session_state:
    st.session_state.model = None
if "index" not in st.session_state:
    st.session_state.index = None

# Upload PDF
uploaded_file = st.file_uploader("📤 Upload your PDF", type="pdf")

if uploaded_file and st.session_state.text is None:
    with st.spinner("🔄 Extracting and processing your document..."):
        pdf_path = os.path.join("data", uploaded_file.name)
        os.makedirs("data", exist_ok=True)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        text = extract_text_from_pdf(pdf_path)
        chunks = chunk_text(text)
        embeddings, model = embed_chunks(chunks)
        index = build_faiss_index(np.array(embeddings))

        st.session_state.text = text
        st.session_state.chunks = chunks
        st.session_state.model = model
        st.session_state.index = index

    st.success("✅ PDF loaded and indexed!")

# Sidebar: collapsible chat history
with st.sidebar.expander("🕒 Chat History", expanded=True):
    if st.session_state.chat_history:
        for i, (user_msg, bot_msg) in enumerate(st.session_state.chat_history):
            st.markdown(f"🧑‍💻 **Q:** {user_msg}")
            st.markdown(f"🤖 **A:** {bot_msg}")
            if i != len(st.session_state.chat_history) - 1:
                st.markdown("---")
    else:
        st.write("No chat history yet.")

# Summarize PDF
if st.session_state.text and st.button("✨ Summarize PDF"):
    with st.spinner("Generating summary..."):
        summary = ask_groq("Summarize this document.", [st.session_state.text[:4000]])
        st.subheader("📌 Summary")
        st.write(summary)

# Chat interface
if st.session_state.text:
    st.subheader("💬 Chat with your PDF")

    # Display full chat history in the main view
    for i, (user_msg, bot_msg) in enumerate(st.session_state.chat_history):
        st.markdown(f"🧑‍💻 **You:** {user_msg}")
        st.markdown(f"🤖 **Bot:** {bot_msg}")
        st.markdown("---")

    # Show input box if last message is not "exit"
    if not st.session_state.chat_history or st.session_state.chat_history[-1][0].lower().strip() != "exit":
        with st.form(key=f"chat_form_{len(st.session_state.chat_history)}"):
            user_input = st.text_input("Type your question (or 'exit' to end chat):", key=f"input_{len(st.session_state.chat_history)}")
            submitted = st.form_submit_button("Send")

            if submitted and user_input:
                with st.spinner("🤖 Thinking..."):
                    context_chunks = search_chunks(user_input, st.session_state.model, st.session_state.chunks, st.session_state.index, top_k=5)
                    answer = ask_groq(user_input, context_chunks)

                st.session_state.chat_history.append((user_input, answer))

    else:
        st.info("🔚 Chat ended by typing 'exit'. Refresh to start over.")
