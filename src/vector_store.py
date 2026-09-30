from __future__ import annotations

import hashlib
from typing import Any, Iterable, List

from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

from .config import Settings


class VectorStoreClient:
    """Thin wrapper around Pinecone access with graceful fallback for local development."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self._pinecone: Any | None = None
        self._embeddings: OpenAIEmbeddings | None = None
        self._store: PineconeVectorStore | None = None
        self._namespace: str | None = None
        self._indexed_fingerprint: str | None = None
        self._indexed_chunk_count = 0

    def document_count(self) -> int:
        if not self.settings.pinecone_api_key:
            return 0
        index = self._get_pinecone().Index(self.settings.pinecone_index_name)
        return int(index.describe_index_stats().total_vector_count)

    def upsert_documents(self, documents: Iterable[str]) -> int:
        items = [document.strip() for document in documents if document.strip()]
        if not items:
            return 0

        fingerprint = hashlib.sha256("\0".join(items).encode("utf-8")).hexdigest()
        if fingerprint == self._indexed_fingerprint:
            return self._indexed_chunk_count

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        chunks: list[str] = []
        metadata: list[dict[str, str | int]] = []
        identifiers: list[str] = []
        for document_number, document in enumerate(items):
            for chunk_number, chunk in enumerate(splitter.split_text(document)):
                if not chunk.strip():
                    continue
                chunks.append(chunk)
                metadata.append({"source": f"document-{document_number + 1}", "chunk": chunk_number})
                identifiers.append(
                    hashlib.sha256(
                        f"{document_number}:{chunk_number}:{chunk}".encode("utf-8")
                    ).hexdigest()
                )

        if not chunks:
            return 0

        namespace = f"docs-{fingerprint[:32]}"
        store = self._get_store(namespace)
        store.add_texts(
            texts=chunks,
            metadatas=metadata,
            ids=identifiers,
            async_req=False,
        )
        self._indexed_fingerprint = fingerprint
        self._indexed_chunk_count = len(chunks)
        return self._indexed_chunk_count

    def similarity_search(self, query: str, top_k: int = 4) -> List[str]:
        if not self._namespace:
            return []
        matches = self._get_store(self._namespace).similarity_search(query, k=top_k)
        return [match.page_content for match in matches]

    def ensure_index(self) -> str:
        if not self.settings.pinecone_api_key:
            return self.settings.pinecone_index_name
        pinecone = self._get_pinecone()
        if self.settings.pinecone_index_name not in pinecone.list_indexes().names():
            pinecone.create_index(
                name=self.settings.pinecone_index_name,
                dimension=self.settings.embedding_dimensions,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=self.settings.pinecone_cloud,
                    region=self.settings.pinecone_region,
                ),
            )
        return self.settings.pinecone_index_name

    def _get_pinecone(self) -> Any:
        if not self.settings.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY is required for Pinecone retrieval.")
        if self._pinecone is None:
            self._pinecone = Pinecone(api_key=self.settings.pinecone_api_key)
        return self._pinecone

    def _get_store(self, namespace: str) -> PineconeVectorStore:
        if not self.settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required to create document embeddings.")
        index_name = self.ensure_index()
        if self._embeddings is None:
            self._embeddings = OpenAIEmbeddings(
                model=self.settings.embedding_model,
                api_key=self.settings.openai_api_key,
                dimensions=self.settings.embedding_dimensions,
            )
        if self._store is None or self._namespace != namespace:
            self._store = PineconeVectorStore(
                index=self._get_pinecone().Index(index_name),
                embedding=self._embeddings,
                namespace=namespace,
            )
            self._namespace = namespace
        return self._store
