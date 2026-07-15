"""
app.py
------
Streamlit front-end for the RAG Q&A system.

Run locally with:
    streamlit run app.py

This file is intentionally thin — all real logic lives in rag_pipeline.py.
"""

import os
import tempfile

import streamlit as st
from rag_pipeline import load_document, chunk_text, VectorStore, answer_question

st.set_page_config(page_title="Document Q&A (RAG)", page_icon="📄", layout="wide")

st.title("📄 Chat With Your Documents (RAG)")
st.caption(
    "Upload PDFs or text files, then ask questions. "
    "Answers are generated only from the content you upload — with sources shown."
)


# Sidebar: API key + settings


with st.sidebar:
    st.header("Settings")
    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        value=os.environ.get("GEMINI_API_KEY", ""),
        help="Get one at https://aistudio.google.com/apikey. Not stored anywhere.",
    )
    chunk_size = st.slider("Chunk size (words)", 200, 800, 500, step=50)
    overlap = st.slider("Chunk overlap (words)", 0, 150, 50, step=10)
    top_k = st.slider("Chunks to retrieve per question", 1, 8, 4)

    st.markdown("---")
    st.markdown(
        "**How this works:**\n"
        "1. Documents are split into chunks\n"
        "2. Each chunk is embedded (sentence-transformers)\n"
        "3. Embeddings stored in a FAISS vector index\n"
        "4. Your question retrieves the most relevant chunks\n"
        "5. Gemini answers using only those chunks"
    )


# Session state


if "store" not in st.session_state:
    st.session_state.store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# Step 1: Upload + index documents


st.subheader("1. Upload documents")
uploaded_files = st.file_uploader(
    "Upload one or more PDF or TXT files",
    type=["pdf", "txt"],
    accept_multiple_files=True,
)

if st.button("Build Index", type="primary", disabled=not uploaded_files):
    with st.spinner("Reading, chunking, and embedding documents..."):
        all_chunks = []
        for uploaded_file in uploaded_files:
            suffix = os.path.splitext(uploaded_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            raw_text = load_document(tmp_path)
            chunks = chunk_text(raw_text, source=uploaded_file.name,
                                 chunk_size=chunk_size, overlap=overlap)
            all_chunks.extend(chunks)
            os.remove(tmp_path)

        store = VectorStore()
        store.build(all_chunks)
        st.session_state.store = store

    st.success(f"Indexed {len(all_chunks)} chunks from {len(uploaded_files)} file(s).")


# Step 2: Ask questions


st.subheader("2. Ask a question")

if st.session_state.store is None:
    st.info("Upload documents and click 'Build Index' first.")
else:
    query = st.text_input("Your question", placeholder="e.g. What is the main conclusion of this document?")

    if st.button("Ask") and query:
        if not api_key:
            st.error("Please enter your Gemini API key in the sidebar.")
        else:
            with st.spinner("Retrieving relevant chunks and generating answer..."):
                answer, sources = answer_question(query, st.session_state.store, api_key, k=top_k)
                st.session_state.chat_history.append((query, answer, sources))

    # Display chat history, most recent first
    for q, a, sources in reversed(st.session_state.chat_history):
        st.markdown(f"**Q: {q}**")
        st.markdown(a)
        with st.expander(f"View {len(sources)} source chunk(s) used"):
            for c in sources:
                st.markdown(f"**{c.source}** (chunk {c.chunk_id})")
                st.text(c.text[:400] + ("..." if len(c.text) > 400 else ""))
        st.markdown("---")