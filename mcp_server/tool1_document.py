"""
Tool 1 – Document Upload, Parsing, Embeddings & Summarization
=============================================================
Flow:
  1. Receive a base64-encoded file + filename.
  2. Upload raw file to S3 (documents/ prefix).
  3. Parse to clean Markdown text using Microsoft MarkItDown.
  4. Chunk the text.
  5. Embed each chunk with Amazon Titan Embed v2 via Bedrock.
  6. Append vectors + metadata to the local FAISS index
     (creates index on first run; appends on subsequent runs).
  7. Summarize the full document with Claude 3.5 Sonnet via Bedrock.
  8. Return doc_id, s3_key, chunk_count, summary.
"""

from __future__ import annotations

import base64
from typing import Any

from utils import (
    append_to_faiss,
    chunk_text,
    embed_texts,
    make_doc_id,
    parse_with_markitdown,
    summarize_with_claude,
    upload_to_s3,
)


def process_document(
    file_b64: str,
    filename: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Upload, parse, embed, and summarize a document.

    Args:
        file_b64:  Base64-encoded file content (PDF, DOCX, PPTX, TXT, MD, etc.)
        filename:  Original filename including extension.
        metadata:  Optional extra key-value metadata stored with each chunk.

    Returns:
        {doc_id, filename, s3_key, chunk_count, total_vectors, summary}
    """
    metadata = metadata or {}

    # 1. Decode
    file_bytes = base64.b64decode(file_b64)

    # 2. Generate doc ID and upload to S3
    doc_id = make_doc_id(filename)
    s3_key = upload_to_s3(file_bytes, filename, doc_id, extra_meta=metadata)

    # 3. Parse with MarkItDown → clean Markdown text
    markdown_text = parse_with_markitdown(file_bytes, filename)

    # 4. Chunk
    chunks = chunk_text(markdown_text)
    if not chunks:
        return {
            "doc_id": doc_id,
            "filename": filename,
            "s3_key": s3_key,
            "chunk_count": 0,
            "total_vectors": 0,
            "summary": "Document appears to be empty or unreadable.",
        }

    # 5. Embed all chunks
    vectors = embed_texts(chunks)

    # 6. Build per-chunk metadata and append to FAISS
    chunk_metas = [
        {
            "doc_id": doc_id,
            "filename": filename,
            "s3_key": s3_key,
            "chunk_index": i,
            "text": chunk,
            **metadata,
        }
        for i, chunk in enumerate(chunks)
    ]
    total_vectors = append_to_faiss(vectors, chunk_metas)

    # 7. Summarize full document text
    summary = summarize_with_claude(markdown_text)

    return {
        "doc_id": doc_id,
        "filename": filename,
        "s3_key": s3_key,
        "chunk_count": len(chunks),
        "total_vectors": total_vectors,
        "summary": summary,
    }
