from __future__ import annotations

import re
from collections import Counter

from langchain_openai import ChatOpenAI
from openai import RateLimitError

from .config import Settings
from .rag import build_retrieval_context
from .vector_store import VectorStoreClient

_vector_store_client: VectorStoreClient | None = None
_vector_store_settings: tuple[str, ...] | None = None
_rate_limited_credentials: tuple[str, str] | None = None
_SUMMARY_REQUEST = re.compile(r"\b(summary|summarize|summarise|overview)\b", re.IGNORECASE)
_SUMMARY_STOP_WORDS = {
    "about", "also", "and", "are", "but", "can", "each", "for", "from", "has",
    "have", "how", "into", "its", "more", "not", "our", "over", "such", "than",
    "that", "the", "their", "these", "they", "this", "through", "was", "were",
    "which", "while", "who", "will", "with", "you",
}


def generate_answer(
    question: str,
    context: str,
    model: str | None = None,
    api_key: str | None = None,
) -> str:
    """Return a concise answer using the supplied retrieval context.

    Calls OpenAI when an API key is supplied and otherwise returns the retrieved
    context as a local-development fallback.
    """
    cleaned_context = context.strip()
    if not cleaned_context:
        return "I could not find relevant information in the available documents."

    effective_model = model or Settings().llm_model
    if api_key:
        response = ChatOpenAI(
            model=effective_model,
            api_key=api_key,
            temperature=0,
        ).invoke(
            [
                (
                    "system",
                    "Answer the user's question using only the supplied document context. "
                    "If the context does not contain the answer, say so clearly.",
                ),
                ("human", f"Question:\n{question}\n\nDocument context:\n{cleaned_context}"),
            ]
        )
        return str(response.content).strip()

    return f"Local keyword retrieval result:\n\n{cleaned_context[:2000]}"


def _extractive_summary(documents: list[str], limit: int = 5) -> str:
    sentences = [
        sentence.strip()
        for document in documents
        for sentence in re.split(r"(?<=[.!?])\s+", document.strip())
        if len(sentence.split()) >= 8
    ]
    if not sentences:
        return ""

    sentence_words = [
        [
            word
            for word in re.findall(r"[a-zA-Z][a-zA-Z'-]{2,}", sentence.lower())
            if word not in _SUMMARY_STOP_WORDS
        ]
        for sentence in sentences
    ]
    word_counts = Counter(word for words in sentence_words for word in set(words))
    ranked = sorted(
        range(len(sentences)),
        key=lambda index: (
            sum(word_counts[word] for word in set(sentence_words[index]))
            / max(len(sentence_words[index]) ** 0.5, 1)
        )
        * (1.15 if index < 2 else 1),
        reverse=True,
    )

    selected: list[int] = []
    for index in ranked:
        words = set(sentence_words[index])
        if not words:
            continue
        if any(
            len(words & set(sentence_words[chosen])) / len(words | set(sentence_words[chosen])) > 0.7
            for chosen in selected
        ):
            continue
        selected.append(index)
        if len(selected) == limit:
            break

    return "\n".join(f"- {sentences[index]}" for index in sorted(selected))


def _local_answer(question: str, documents: list[str], model: str) -> str:
    if _SUMMARY_REQUEST.search(question):
        summary = _extractive_summary(documents)
        if summary:
            return f"Local extractive summary (OpenAI unavailable):\n\n{summary}"

    context = build_retrieval_context(documents, question)
    return generate_answer(question, context, model)


def answer_question(question: str, documents: list[str]) -> str:
    settings = Settings()
    if settings.openai_api_key and settings.pinecone_api_key and settings.rag_mode != "local":
        global _rate_limited_credentials, _vector_store_client, _vector_store_settings
        credentials = (settings.openai_api_key, settings.pinecone_api_key)
        if _rate_limited_credentials != credentials:
            _rate_limited_credentials = None

        if _rate_limited_credentials == credentials:
            return _local_answer(question, documents, settings.llm_model)

        current_settings = (
            settings.openai_api_key,
            settings.pinecone_api_key,
            settings.pinecone_index_name,
            settings.embedding_model,
            str(settings.embedding_dimensions),
            str(settings.chunk_size),
            str(settings.chunk_overlap),
            settings.pinecone_cloud,
            settings.pinecone_region,
        )
        if _vector_store_client is None or _vector_store_settings != current_settings:
            _vector_store_client = VectorStoreClient(settings)
            _vector_store_settings = current_settings

        try:
            _vector_store_client.upsert_documents(documents)
            context = "\n\n".join(_vector_store_client.similarity_search(question))
            return generate_answer(question, context, settings.llm_model, settings.openai_api_key)
        except RateLimitError:
            _rate_limited_credentials = credentials
            local_answer = _local_answer(question, documents, settings.llm_model)
            return (
                "OpenAI returned a rate or quota limit. Using the local document fallback instead. "
                "To restore live answers, add OpenAI credits at "
                "https://platform.openai.com/settings/organization/billing/ and restart the app.\n\n"
                f"{local_answer}"
            )

    return _local_answer(question, documents, settings.llm_model)
