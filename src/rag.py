import re
from typing import Iterable, List


def _tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-zA-Z0-9]+", text.lower()) if len(token) > 2}


def build_retrieval_context(documents: Iterable[str], query: str, limit: int = 5) -> str:
    """Rank raw document chunks by keyword overlap and return the most relevant snippets."""
    docs = [doc.strip() for doc in documents if isinstance(doc, str) and doc.strip()]
    if not docs:
        return ""

    query_terms = _tokenize(query)
    ranked: list[tuple[int, str]] = []

    for document in docs:
        if not query_terms:
            score = 1
        else:
            score = sum(1 for term in query_terms if term in document.lower())
        ranked.append((score, document))

    ranked.sort(key=lambda item: item[0], reverse=True)
    best_documents = [document for _, document in ranked[:limit] if document]
    if not best_documents:
        best_documents = docs[:limit]

    return "\n\n".join(best_documents)
