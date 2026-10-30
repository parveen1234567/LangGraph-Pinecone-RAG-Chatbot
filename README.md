# 🤖 Agentic RAG Chatbot

<p align="center">
  <strong>An AI-powered Retrieval-Augmented Generation (RAG) application that enables users to upload PDF documents and ask questions using LangGraph, LangChain, Pinecone, OpenAI, FastAPI, and Streamlit.</strong>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Language-Python-3776AB?logo=python&logoColor=white">
  <img alt="LangGraph" src="https://img.shields.io/badge/AI-LangGraph-blueviolet">
  <img alt="LangChain" src="https://img.shields.io/badge/Framework-LangChain-00A67E">
  <img alt="Pinecone" src="https://img.shields.io/badge/Vector%20DB-Pinecone-orange">
  <img alt="OpenAI" src="https://img.shields.io/badge/LLM-OpenAI-412991?logo=openai&logoColor=white">
  <img alt="FastAPI" src="https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white">
  <img alt="Streamlit" src="https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white">
</p>

<p align="center">
  <a href="https://langgraph-pinecone-rag-chatbot-hgzefbeyqjku4dmodabv27.streamlit.app/">🌐 Live Demo</a> •
  <a href="https://github.com/parveen1234567/LangGraph-Pinecone-RAG-Chatbot">💻 Source Code</a>
</p>

---

# 🚀 Project Overview

The Agentic RAG Chatbot is an AI-powered document intelligence system that enables users to upload PDF documents and interact with them using natural language.

The application combines modern Retrieval-Augmented Generation (RAG) architecture with vector search capabilities to provide accurate, context-aware answers from uploaded documents.

Instead of manually searching large PDFs, users can simply ask questions and receive intelligent responses generated from the most relevant document sections.

---

# 💡 Problem Statement

Organizations and individuals often work with lengthy PDF documents containing large amounts of information.

Traditional document search methods require:

- Manual reading
- Keyword searching
- Navigating through multiple pages
- Time-consuming information retrieval

This project solves that challenge by enabling conversational interaction with PDF documents.

---

# 🎯 Product Value

The Agentic RAG Chatbot enables users to:

### For Students

- Ask questions from academic PDFs
- Summarize study material
- Quickly locate important concepts
- Simplify research workflows

### For Professionals

- Analyze business documents
- Extract key insights
- Generate summaries
- Improve productivity

### For Organizations

- Knowledge retrieval from internal documents
- AI-assisted document search
- Faster decision making
- Improved information accessibility

---

# ✨ Core Features

## 📄 PDF Document Processing

- Upload PDF documents
- Automatic text extraction
- Document chunking
- Intelligent indexing

## 🧠 Retrieval-Augmented Generation (RAG)

- Semantic search
- Context retrieval
- AI-generated answers
- Source-aware responses

## 🔎 Pinecone Vector Search

- Vector database storage
- Similarity search
- Fast retrieval
- Scalable knowledge indexing

## 🤖 OpenAI Integration

- Intelligent answer generation
- Context-aware responses
- Natural language understanding

## 🔗 LangGraph Workflow

- Agent-based workflow execution
- Retrieval orchestration
- Response generation pipeline

## ⚡ FastAPI Backend

- REST API architecture
- Fast request processing
- API documentation support

## 🎨 Streamlit User Interface

- Interactive web interface
- PDF upload functionality
- Conversational experience
- User-friendly design

## 🔄 Local Fallback Mode

When API keys are unavailable:

- Local keyword search
- Extractive summarization
- Offline functionality

---

# 🏗️ System Architecture

## High-Level Architecture

```text
User
 │
 ▼
Streamlit Interface
 │
 ▼
FastAPI Backend
 │
 ▼
LangGraph Workflow
 │
 ├── PDF Processing
 ├── Chunking
 ├── Embedding Generation
 ├── Pinecone Retrieval
 └── OpenAI Generation
 │
 ▼
Final Response
```

---

## RAG Workflow

```text
PDF Upload
     │
     ▼
Document Extraction
     │
     ▼
Text Chunking
     │
     ▼
Embedding Creation
     │
     ▼
Pinecone Vector Store
     │
     ▼
User Question
     │
     ▼
Similarity Search
     │
     ▼
Relevant Chunks Retrieved
     │
     ▼
OpenAI LLM
     │
     ▼
AI-Generated Answer
```

---

# 🗂️ Project Structure

```text
LangGraph-Pinecone-RAG-Chatbot/
│
├── data/
│   └── PDF Documents
│
├── src/
│   │
│   ├── main.py                 # FastAPI application entry point
│   ├── streamlit_app.py        # Streamlit UI
│   │
│   ├── ingestion/
│   │   ├── pdf_loader.py
│   │   └── chunking.py
│   │
│   ├── retrieval/
│   │   ├── vector_store.py
│   │   ├── retriever.py
│   │   └── embeddings.py
│   │
│   ├── agents/
│   │   ├── rag_agent.py
│   │   └── workflow.py
│   │
│   ├── services/
│   │   ├── openai_service.py
│   │   └── pinecone_service.py
│   │
│   └── utils/
│       ├── config.py
│       └── helpers.py
│
├── .env.example
├── requirements.txt
├── README.md
└── .gitignore
```

---

# ⚙️ Technology Stack

## Artificial Intelligence

| Technology | Purpose |
|------------|----------|
| LangGraph | Agentic Workflow |
| LangChain | RAG Framework |
| OpenAI | Answer Generation |
| Embeddings | Semantic Understanding |

---

## Backend

| Technology | Purpose |
|------------|----------|
| Python | Core Language |
| FastAPI | REST API |
| Pydantic | Validation |
| Uvicorn | ASGI Server |

---

## Data Layer

| Technology | Purpose |
|------------|----------|
| Pinecone | Vector Database |
| PDF Loader | Document Processing |
| Text Splitter | Chunking |
| Embeddings | Vector Creation |

---

## Frontend

| Technology | Purpose |
|------------|----------|
| Streamlit | Interactive Dashboard |
| HTML/CSS | Styling |
| Python UI Components | User Interface |

---

# 🔄 Document Processing Pipeline

## Step 1: PDF Ingestion

- Reads PDF files
- Extracts text
- Processes document metadata

## Step 2: Chunk Generation

- Splits large documents
- Creates manageable chunks
- Preserves context

## Step 3: Embedding Creation

- Converts text into vectors
- Generates semantic representations

## Step 4: Pinecone Storage

- Saves embeddings
- Creates searchable vector index

## Step 5: Retrieval

- Retrieves relevant document chunks

## Step 6: Generation

- Sends context to OpenAI
- Generates final answer

---

# 🔌 API Endpoints

## Ask Questions

```http
POST /ask
```

### Request

```json
{
  "question": "What is covered in the documents?"
}
```

### Response

```json
{
  "answer": "Generated answer..."
}
```

---

## API Documentation

```text
http://localhost:8001/docs
```

## OpenAPI Interface

```text
http://localhost:8001/
```

---

# 🚀 Running the Project

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Configure Environment Variables

```env
OPENAI_API_KEY=your_key
PINECONE_API_KEY=your_key

PINECONE_INDEX_NAME=index_name

PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1

EMBEDDING_DIMENSIONS=1536
```

---

## Start FastAPI Server

```powershell
$env:APP_PORT="8001"

python -m src.main
```

---

## Start Streamlit UI

```bash
streamlit run src/streamlit_app.py --server.port 8503
```

---

# 🧪 Example Workflow

```text
Upload PDF
      │
      ▼
Ask Question
      │
      ▼
Vector Search
      │
      ▼
Retrieve Context
      │
      ▼
OpenAI Response
      │
      ▼
Answer Returned
```

---

# 🧩 Challenges Faced

## Large Document Handling

Managing large PDF files while preserving context and retrieval quality.

### Learning

Proper chunking strategies are critical for effective RAG systems.

---

## Vector Search Optimization

Selecting relevant chunks quickly from the vector database.

### Learning

Embedding quality directly affects retrieval accuracy.

---

## OpenAI Rate Limits

Handling API quota and request limits gracefully.

### Solution

Fallback local retrieval mode was implemented.

---

## Pinecone Configuration

Aligning vector dimensions and index settings with embedding models.

### Learning

Vector database configuration must match embedding architecture.

---

# 📈 Future Enhancements

Future versions could include:

- 📂 Multi-document collections
- 🗨️ Chat history memory
- 🌍 Multi-language support
- 📊 Document analytics dashboard
- 📑 Citation-based answers
- 📱 Mobile-friendly interface
- 🔒 User authentication
- 🏢 Multi-user workspaces
- ☁️ Cloud document storage
- 🤖 Advanced AI agents

---

# 📚 What I Learned

During this project I gained practical experience in:

- Retrieval-Augmented Generation (RAG)
- LangGraph Agent Workflows
- LangChain Components
- Vector Databases
- Pinecone Indexing
- OpenAI API Integration
- FastAPI Development
- Streamlit Applications
- Document Processing
- Embedding Models
- AI Application Deployment

---

# 🎯 Why This Project?

This project demonstrates how modern Generative AI systems can combine Large Language Models with external knowledge sources to provide accurate and context-aware answers.

Rather than relying entirely on model memory, the application retrieves relevant document content before generating responses.

> **Upload Documents. Retrieve Knowledge. Generate Intelligent Answers.**

---

# 📌 Project Highlights

- ✅ Agentic AI Architecture
- ✅ LangGraph Workflow
- ✅ Retrieval-Augmented Generation
- ✅ OpenAI Integration
- ✅ Pinecone Vector Database
- ✅ PDF Question Answering
- ✅ FastAPI Backend
- ✅ Streamlit Frontend
- ✅ Local Fallback Mode
- ✅ Cloud Deployment Ready

---

# 🔗 Links

| Resource | Link |
|-----------|------|
| 🌐 Live Application | https://langgraph-pinecone-rag-chatbot-hgzefbeyqjku4dmodabv27.streamlit.app/ |
| 💻 GitHub Repository | https://github.com/parveen1234567/LangGraph-Pinecone-RAG-Chatbot |
| 👨‍💻 GitHub Profile | https://github.com/parveen1234567 |
| 💼 LinkedIn | https://linkedin.com/in/shaik-chinthapalli-parveen-983668257 |

---

# 🙌 Final Note

The Agentic RAG Chatbot represents my journey into Generative AI, Retrieval-Augmented Generation, vector databases, and agent-based workflows.

This project helped me understand how modern AI systems combine document retrieval, semantic search, vector embeddings, and large language models to create intelligent applications capable of answering questions from custom knowledge bases.

> **Agentic RAG Chatbot — Transform Documents into Intelligent Conversations.**
