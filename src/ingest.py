from __future__ import annotations

from pathlib import Path
from typing import List

from pypdf import PdfReader


def _read_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    chunks: List[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        chunks.append(text)
    return "\n\n".join(chunks)


def load_documents(data_dir: str | Path) -> List[str]:
    """Load all PDF documents from a directory and return their text contents."""
    directory = Path(data_dir)
    if not directory.exists():
        return []

    documents: List[str] = []
    for pdf_file in sorted(directory.glob("*.pdf")):
        documents.append(_read_pdf_text(pdf_file))
    return documents
