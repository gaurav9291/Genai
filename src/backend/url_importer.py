"""Utilities for importing documents from direct URLs."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import unquote, urlparse

import requests
from sqlalchemy.orm import Session

from .agents.DocumentIngestionAgent import DocumentIngestionAgent
from .config import settings


PDF_MIME_TYPES = {
    "application/pdf",
    "application/x-pdf",
    "application/acrobat",
    "applications/vnd.pdf",
}


def _filename_from_url(url: str, fallback: str = "imported-document.pdf") -> str:
    parsed = urlparse(url)
    name = Path(unquote(parsed.path)).name
    if not name:
        return fallback
    if not name.lower().endswith(".pdf"):
        name = f"{name}.pdf"
    return name


def _validate_pdf_response(url: str, response: requests.Response) -> Optional[str]:
    content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    path_is_pdf = urlparse(url).path.lower().endswith(".pdf")
    if content_type in PDF_MIME_TYPES or path_is_pdf:
        return None
    return (
        "URL does not look like a direct PDF download. Provide a direct .pdf URL "
        f"or a server response with PDF content-type. Received content-type: {content_type or 'unknown'}."
    )


async def import_pdf_from_url(
    db: Session,
    url: str,
    *,
    doc_type: str = "Reference",
    title: str = "",
    approved: bool = True,
    feedback_score: int = 3,
    timeout_seconds: int = 30,
) -> Dict[str, Any]:
    """Download a direct PDF URL and ingest it into the document database.

    The caller is responsible for only supplying URLs they are allowed to
    download and store. This helper intentionally accepts direct PDF downloads
    only; it does not scrape pages or bypass access controls.
    """
    clean_url = (url or "").strip()
    parsed = urlparse(clean_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return {"status": "error", "message": "Provide a valid http or https PDF URL."}

    temp_path: Optional[str] = None
    try:
        with requests.get(clean_url, stream=True, timeout=timeout_seconds, allow_redirects=True) as response:
            response.raise_for_status()

            validation_error = _validate_pdf_response(clean_url, response)
            if validation_error:
                return {"status": "error", "message": validation_error}

            content_length = response.headers.get("content-length")
            if content_length and int(content_length) > settings.max_file_size:
                return {
                    "status": "error",
                    "message": f"PDF is larger than the configured limit of {settings.max_file_size} bytes.",
                }

            filename = _filename_from_url(response.url or clean_url)
            fd, temp_path = tempfile.mkstemp(suffix=".pdf")
            downloaded = 0
            with os.fdopen(fd, "wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 128):
                    if not chunk:
                        continue
                    downloaded += len(chunk)
                    if downloaded > settings.max_file_size:
                        return {
                            "status": "error",
                            "message": f"PDF is larger than the configured limit of {settings.max_file_size} bytes.",
                        }
                    handle.write(chunk)

        if not temp_path or os.path.getsize(temp_path) == 0:
            return {"status": "error", "message": "Downloaded file was empty."}

        agent = DocumentIngestionAgent()
        result = await agent.execute(
            db=db,
            filename=filename,
            file_path=temp_path,
            doc_type=doc_type.strip() if doc_type.strip() else "Reference",
            approved=approved,
            feedback_score=feedback_score,
        )

        if result.get("status") == "success":
            result["source_url"] = clean_url
            result["final_url"] = response.url if "response" in locals() else clean_url
            result["downloaded_bytes"] = os.path.getsize(temp_path)
            from .models import Document

            document = db.query(Document).filter(Document.id == result["document_id"]).first()
            if document:
                if title.strip():
                    document.title = title.strip()
                    result.setdefault("metadata", {})["title"] = document.title
                metadata = dict(document.generation_metadata or {})
                metadata.update(
                    {
                        "import_method": "direct_pdf_url",
                        "source_url": clean_url,
                        "final_url": result["final_url"],
                        "downloaded_bytes": result["downloaded_bytes"],
                    }
                )
                document.generation_metadata = metadata
                db.commit()

        return result
    except requests.exceptions.RequestException as exc:
        db.rollback()
        return {"status": "error", "message": f"PDF download failed: {exc}"}
    except Exception as exc:
        db.rollback()
        return {"status": "error", "message": f"PDF import failed: {exc}"}
    finally:
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)
