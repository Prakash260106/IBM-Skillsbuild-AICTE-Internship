"""
utils/document_parser.py
--------------------------
Extract plain text from uploaded Streamlit file objects (.txt, .md, .pdf).
Returns a clean string suitable for passing to the AI model.
"""

import io
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def extract_text(uploaded_file) -> tuple[str, Optional[str]]:
    """
    Accept a Streamlit UploadedFile object and return (text, error).
    On success *error* is None; on failure *text* is "".

    Supported formats: .txt  .md  .pdf
    """
    if uploaded_file is None:
        return "", "No file provided."

    filename: str = uploaded_file.name.lower()
    raw_bytes: bytes = uploaded_file.read()

    if not raw_bytes:
        return "", "The uploaded file is empty."

    # ── Plain text / Markdown ──────────────────────────────────────────────
    if filename.endswith((".txt", ".md")):
        try:
            text = raw_bytes.decode("utf-8", errors="replace")
            return text.strip(), None
        except Exception as exc:
            return "", f"Failed to decode text file: {exc}"

    # ── PDF ────────────────────────────────────────────────────────────────
    if filename.endswith(".pdf"):
        try:
            # PyPDF2 is listed in requirements; import lazily to keep startup fast
            import PyPDF2  # noqa: PLC0415

            reader = PyPDF2.PdfReader(io.BytesIO(raw_bytes))
            pages: list[str] = []
            for page in reader.pages:
                page_text = page.extract_text() or ""
                pages.append(page_text)
            text = "\n".join(pages).strip()
            if not text:
                return "", "PDF parsed but no readable text was found (scanned image PDF?)."
            return text, None
        except ImportError:
            return "", "PyPDF2 is not installed. Run: pip install PyPDF2"
        except Exception as exc:
            return "", f"PDF parsing error: {exc}"

    return "", f"Unsupported file type '{uploaded_file.name}'. Use .txt, .md, or .pdf."


def truncate_text(text: str, max_chars: int = 12_000) -> str:
    """
    Hard-truncate *text* to *max_chars* characters to keep token budgets
    within Granite's context window.  Appends a notice when truncation occurs.
    """
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    notice = f"\n\n[NOTE: Document truncated to {max_chars:,} characters for processing.]"
    return truncated + notice
