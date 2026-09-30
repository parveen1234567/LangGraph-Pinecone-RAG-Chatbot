# RAG Agentic AI

This project provides a lightweight Retrieval-Augmented Generation (RAG) application for asking questions over PDF documents.

## Features

- PDF ingestion from the `data/` folder
- LangChain-compatible retrieval and answer generation flow
- FastAPI backend for REST queries
- Streamlit dashboard for local interaction
- OpenAI and Pinecone configuration via environment variables

## Setup

1. Copy `.env.example` to `.env` and fill in your keys.
2. Place PDF files in the `data/` directory.
3. Install project dependencies:

   ```bash
   pip install -r requirements.txt
   ```

When both API keys are set, the app creates the configured Pinecone index if needed, chunks and embeds the PDFs, retrieves relevant passages, and asks OpenAI to answer from those passages. The defaults create an AWS serverless index in `us-east-1` with 1536 dimensions; configure `PINECONE_CLOUD`, `PINECONE_REGION`, and `EMBEDDING_DIMENSIONS` in `.env` to match your Pinecone setup and embedding model. Live requests send document chunks and questions to those providers and may incur usage charges. Without both keys, the local keyword-search fallback is used; summary and overview prompts get a short extractive summary from the PDF text. Set `RAG_MODE=local` to force local-only mode; after an OpenAI 429 quota error, the app falls back locally and stops retrying cloud calls until restart.

The Streamlit page accepts PDF uploads, so deployed users can add documents in the browser without committing private PDFs to GitHub. Uploaded files remain available for that browser session; upload them again in a new session.

## Run the API

```powershell
$env:APP_PORT="8001"
python -m src.main
```

The terminal prints the browser address. Open either URL; the root address redirects to the docs:

- http://localhost:8001/docs
- http://localhost:8001/

Do not use `http://0.0.0.0:8001` in a browser. `0.0.0.0` is the server's bind address; use `localhost` instead.

## Run the UI

```bash
streamlit run src/streamlit_app.py --server.port 8503
```

Streamlit prints its local address in the terminal. Open http://localhost:8503.

## Example request

```bash
curl -X POST http://localhost:8001/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is covered in the documents?"}'
```
