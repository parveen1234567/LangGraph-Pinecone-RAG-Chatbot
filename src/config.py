import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass
class Settings:
    """Application configuration loaded from environment variables."""

    openai_api_key: str = ""
    pinecone_api_key: str = ""
    pinecone_index_name: str = "agentic-ai-index"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    rag_mode: str = "auto"
    data_dir: str = "data"
    chunk_size: int = 800
    chunk_overlap: int = 150

    def __init__(self):
        load_dotenv()
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY", "")
        self.pinecone_index_name = os.getenv("PINECONE_INDEX_NAME", "agentic-ai-index")
        self.pinecone_cloud = os.getenv("PINECONE_CLOUD", "aws")
        self.pinecone_region = os.getenv("PINECONE_REGION", "us-east-1")
        self.llm_model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        self.embedding_dimensions = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))
        self.rag_mode = os.getenv("RAG_MODE", "auto").strip().lower()
        self.data_dir = os.getenv("DATA_DIR", "data")
        self.chunk_size = int(os.getenv("CHUNK_SIZE", "800"))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "150"))
