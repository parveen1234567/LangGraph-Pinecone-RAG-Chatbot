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
    from .ingest import load_documents
except ImportError:  # pragma: no cover
    from src.agent import answer_question
    from src.config import Settings
    from src.ingest import load_documents


st.set_page_config(page_title="RAG Agentic AI", page_icon="🤖")
st.title("RAG Agentic AI")

settings = Settings()
documents = load_documents(settings.data_dir)

with st.form("chat_form"):
    question = st.text_input("Ask a question about your documents")
    submitted = st.form_submit_button("Search")

if submitted and question:
    with st.spinner("Searching the document set..."):
        answer = answer_question(question, documents)
    st.write(answer)
else:
    st.info("Upload PDFs into the data folder and ask a question to start.")
