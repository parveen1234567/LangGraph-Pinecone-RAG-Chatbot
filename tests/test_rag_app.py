from io import BytesIO
from types import SimpleNamespace

import httpx
from openai import RateLimitError

from src import agent, config, ingest, vector_store
from src.config import Settings
from src.rag import build_retrieval_context


def test_settings_reads_required_env_vars(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("PINECONE_API_KEY", "test-pinecone-key")
    monkeypatch.setenv("PINECONE_INDEX_NAME", "test-index")

    settings = Settings()

    assert settings.openai_api_key == "test-openai-key"
    assert settings.pinecone_api_key == "test-pinecone-key"
    assert settings.pinecone_index_name == "test-index"


def test_settings_loads_dotenv(monkeypatch):
    loaded = False

    def load_environment_file():
        nonlocal loaded
        loaded = True

    monkeypatch.setattr(config, "load_dotenv", load_environment_file)

    Settings()

    assert loaded


def test_build_retrieval_context_returns_relevant_snippets():
    documents = [
        "The refund policy allows returns within 30 days after purchase.",
        "Customer support is available through email only.",
        "Shipping takes 3 to 5 business days."
    ]

    context = build_retrieval_context(documents, "refund policy")

    assert "refund" in context.lower()
    assert len(context) > 0


def test_load_uploaded_pdf_extracts_text(monkeypatch):
    class FakePdfReader:
        def __init__(self, uploaded_file):
            assert uploaded_file.read() == b"fake pdf bytes"
            self.pages = [SimpleNamespace(extract_text=lambda: "Uploaded PDF content")]

    monkeypatch.setattr(ingest, "PdfReader", FakePdfReader)

    documents = ingest.load_uploaded_documents([BytesIO(b"fake pdf bytes")])

    assert documents == ["Uploaded PDF content"]


def test_answer_question_uses_pinecone_and_openai_when_configured(monkeypatch):
    calls = {}
    settings = SimpleNamespace(
        openai_api_key="test-openai-key",
        pinecone_api_key="test-pinecone-key",
        pinecone_index_name="test-index",
        llm_model="test-model",
        embedding_model="test-embedding-model",
        embedding_dimensions=1536,
        pinecone_cloud="aws",
        pinecone_region="us-east-1",
        chunk_size=100,
        chunk_overlap=10,
        rag_mode="auto",
    )

    class FakeVectorStore:
        def upsert_documents(self, documents):
            calls["documents"] = documents

        def similarity_search(self, query, top_k=4):
            calls["search"] = (query, top_k)
            return ["Retrieved document context"]

    class FakeChatModel:
        def __init__(self, model, api_key, temperature):
            calls["chat_config"] = (model, api_key, temperature)

        def invoke(self, messages):
            calls["messages"] = messages
            return SimpleNamespace(content="Grounded answer")

    fake_store = FakeVectorStore()
    monkeypatch.setattr(agent, "Settings", lambda: settings)
    monkeypatch.setattr(agent, "_vector_store_client", fake_store)
    monkeypatch.setattr(agent, "_vector_store_settings", None)
    monkeypatch.setattr(agent, "VectorStoreClient", lambda received: fake_store)
    monkeypatch.setattr(agent, "ChatOpenAI", FakeChatModel)

    result = agent.answer_question("What does the PDF say?", ["PDF text"])

    assert result == "Grounded answer"
    assert calls["documents"] == ["PDF text"]
    assert calls["search"] == ("What does the PDF say?", 4)
    assert calls["chat_config"] == ("test-model", "test-openai-key", 0)
    assert "Retrieved document context" in str(calls["messages"])


def test_answer_question_falls_back_when_openai_quota_is_exhausted(monkeypatch):
    settings = SimpleNamespace(
        openai_api_key="test-openai-key",
        pinecone_api_key="test-pinecone-key",
        pinecone_index_name="test-index",
        llm_model="test-model",
        embedding_model="test-embedding-model",
        embedding_dimensions=1536,
        pinecone_cloud="aws",
        pinecone_region="us-east-1",
        chunk_size=100,
        chunk_overlap=10,
        rag_mode="auto",
    )
    calls = {"upserts": 0}

    class QuotaLimitedStore:
        def upsert_documents(self, documents):
            calls["upserts"] += 1
            raise RateLimitError(
                "No credits remaining",
                response=httpx.Response(
                    429,
                    request=httpx.Request("POST", "https://api.openai.com/v1/embeddings"),
                ),
                body={"error": {"code": "credit_balance_exhausted"}},
            )

        def similarity_search(self, query, top_k=4):
            raise AssertionError("Cloud search should not run after failed embeddings")

    quota_limited_store = QuotaLimitedStore()
    monkeypatch.setattr(agent, "Settings", lambda: settings)
    monkeypatch.setattr(agent, "_vector_store_client", quota_limited_store)
    monkeypatch.setattr(agent, "_vector_store_settings", None)
    monkeypatch.setattr(agent, "_rate_limited_credentials", None, raising=False)
    monkeypatch.setattr(agent, "VectorStoreClient", lambda received: quota_limited_store)

    first_answer = agent.answer_question(
        "refund policy", ["The refund policy allows returns within 30 days."]
    )
    second_answer = agent.answer_question(
        "refund policy", ["The refund policy allows returns within 30 days."]
    )

    assert calls["upserts"] == 1
    assert "credit" in first_answer.lower()
    assert "local keyword" in first_answer.lower()
    assert "30 days" in first_answer
    assert "30 days" in second_answer


def test_local_summary_request_returns_concise_extractive_summary(monkeypatch):
    settings = SimpleNamespace(
        openai_api_key="",
        pinecone_api_key="",
        llm_model="test-model",
        rag_mode="local",
    )
    monkeypatch.setattr(agent, "Settings", lambda: settings)
    documents = [
        "Konverge AI helps businesses build products using data, machine learning, and business insights. "
        "The book combines its experience with Emergence AI's expertise in autonomous multi-agent systems. "
        "These systems coordinate agents to adapt to complex workflows and improve business operations. "
        "This guide explains practical ways to apply agentic AI in a changing business environment."
    ]

    result = agent.answer_question("summary give", documents)

    assert result.startswith("Local extractive summary")
    assert "autonomous multi-agent systems" in result
    assert "complex workflows" in result
    assert len(result) < 1000


def test_vector_store_creates_index_and_indexes_chunks_once(monkeypatch):
    calls = {"indexes": [], "upserts": []}

    class FakePinecone:
        def __init__(self, api_key):
            calls["pinecone_key"] = api_key

        def list_indexes(self):
            return SimpleNamespace(names=lambda: [index["name"] for index in calls["indexes"]])

        def create_index(self, **kwargs):
            calls["indexes"].append(kwargs)

        def Index(self, name):
            return name

    class FakeEmbeddings:
        def __init__(self, **kwargs):
            calls["embedding_config"] = kwargs

    class FakePineconeVectorStore:
        def __init__(self, index, embedding, namespace):
            calls["store_config"] = (index, embedding, namespace)

        def add_texts(self, texts, metadatas, ids, async_req):
            calls["upserts"].append((texts, metadatas, ids))

        def similarity_search(self, query, k):
            calls["search"] = (query, k)
            return [SimpleNamespace(page_content="Matched chunk")]

    class FakeSplitter:
        def __init__(self, chunk_size, chunk_overlap):
            calls["splitter_config"] = (chunk_size, chunk_overlap)

        def split_text(self, text):
            return [text[:20], text[20:]]

    monkeypatch.setattr(vector_store, "Pinecone", FakePinecone)
    monkeypatch.setattr(vector_store, "OpenAIEmbeddings", FakeEmbeddings)
    monkeypatch.setattr(vector_store, "PineconeVectorStore", FakePineconeVectorStore)
    monkeypatch.setattr(vector_store, "RecursiveCharacterTextSplitter", FakeSplitter)

    settings = Settings()
    settings.openai_api_key = "test-openai-key"
    settings.pinecone_api_key = "test-pinecone-key"
    settings.pinecone_index_name = "test-index"
    client = vector_store.VectorStoreClient(settings)

    assert client.upsert_documents(["A sample document for indexing."]) == 2
    assert client.upsert_documents(["A sample document for indexing."]) == 2
    original_namespace = calls["store_config"][2]
    assert client.upsert_documents(["A different document set."]) == 2
    assert calls["store_config"][2] != original_namespace
    assert client.similarity_search("sample question") == ["Matched chunk"]

    assert len(calls["indexes"]) == 1
    assert calls["indexes"][0]["name"] == "test-index"
    assert calls["pinecone_key"] == "test-pinecone-key"
    assert len(calls["upserts"]) == 2
    assert calls["search"] == ("sample question", 4)
