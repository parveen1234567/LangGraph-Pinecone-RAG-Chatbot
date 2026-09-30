from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

try:
    from .agent import answer_question
    from .config import Settings
    from .ingest import load_documents, load_uploaded_documents
except ImportError:  # pragma: no cover
    from src.agent import answer_question
    from src.config import Settings
    from src.ingest import load_documents, load_uploaded_documents


st.set_page_config(page_title="RAG Agentic AI", page_icon="🤖")
st.title("RAG Agentic AI")

settings = Settings()
documents = load_documents(settings.data_dir)
uploaded_files = st.file_uploader(
    "Upload PDF documents",
    type=["pdf"],
    accept_multiple_files=True,
)
if uploaded_files:
    documents.extend(load_uploaded_documents(uploaded_files))

if documents:
    st.caption(f"{len(documents)} PDF document(s) ready")

with st.form("chat_form"):
    question = st.text_input("Ask a question about your documents")
    submitted = st.form_submit_button("Search")

if submitted and question:
    if not documents:
        st.warning("Upload a PDF before asking a question.")
    else:
        with st.spinner("Searching the document set..."):
            answer = answer_question(question, documents)
        st.write(answer)
elif not documents:
    st.info("Upload a PDF above to start asking questions.")
