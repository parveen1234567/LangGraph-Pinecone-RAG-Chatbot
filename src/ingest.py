from __future__ import annotations

from pathlib import Path
from typing import BinaryIO, Iterable, List

from pypdf import PdfReader


def _extract_pdf_text(reader: PdfReader) -> str:
    chunks: List[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        chunks.append(text)
    return "\n\n".join(chunks)


def _read_pdf_text(path: Path) -> str:
    return _extract_pdf_text(PdfReader(str(path)))


def load_documents(data_dir: str | Path) -> List[str]:
    """Load all PDF documents from a directory and return their text contents."""
    directory = Path(data_dir)
    if not directory.exists():
        return []

    documents: List[str] = []
    for pdf_file in sorted(directory.glob("*.pdf")):
        documents.append(_read_pdf_text(pdf_file))
    return documents


def load_uploaded_documents(uploaded_files: Iterable[BinaryIO]) -> List[str]:
    """Extract text from PDF file streams uploaded through the UI."""
    documents: List[str] = []
    for uploaded_file in uploaded_files:
        uploaded_file.seek(0)
        documents.append(_extract_pdf_text(PdfReader(uploaded_file)))
    return documents
