"""
utils.py – Shared utilities for all MCP tools
==============================================
Provides:
  - AWS Bedrock client + embedder (Titan Embed v2)
  - AWS S3 client
  - MarkItDown document parser (Microsoft open-source)
  - FAISS index helpers: load, save, append, search
  - Text chunker
  - Claude 3 Sonnet summarizer
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import tempfile
import time
from pathlib import Path
from typing import Any

import boto3
import numpy as np

# ── Config from environment ───────────────────────────────────────────────────
AWS_REGION      = os.environ.get("AWS_REGION", "us-east-1")
S3_BUCKET       = os.environ.get("S3_BUCKET", "mcp-raw-docs")
S3_DOCS_PREFIX  = os.environ.get("S3_DOCS_PREFIX", "documents/")
FAISS_INDEX_DIR = Path(os.environ.get("FAISS_INDEX_DIR", "./faiss_index"))
EMBED_MODEL_ID  = os.environ.get("EMBED_MODEL_ID", "amazon.titan-embed-text-v2:0")
LLM_MODEL_ID    = os.environ.get("LLM_MODEL_ID", "anthropic.claude-3-5-sonnet-20241022-v2:0")
CHUNK_SIZE      = int(os.environ.get("CHUNK_SIZE", "800"))
CHUNK_OVERLAP   = int(os.environ.get("CHUNK_OVERLAP", "150"))
EMBED_DIM       = int(os.environ.get("EMBED_DIM", "1024"))  # Titan v2 default

# FAISS file paths
FAISS_INDEX_FILE = FAISS_INDEX_DIR / "index.faiss"
FAISS_META_FILE  = FAISS_INDEX_DIR / "metadata.pkl"

# ── AWS clients (use inbuilt credentials / IAM role) ─────────────────────────
def get_s3_client():
    return boto3.client("s3", region_name=AWS_REGION)

def get_bedrock_client():
    return boto3.client("bedrock-runtime", region_name=AWS_REGION)

# ── Embedding ─────────────────────────────────────────────────────────────────

def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Embed a list of texts using Amazon Titan Text Embeddings v2.
    Returns a float32 numpy array of shape (N, EMBED_DIM).
    """
    import json as _json
    client = get_bedrock_client()
    vectors = []
    for text in texts:
        body = _json.dumps({"inputText": text[:8000]})  # Titan v2 max input
        resp = client.invoke_model(
            modelId=EMBED_MODEL_ID,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        result = _json.loads(resp["body"].read())
        vectors.append(result["embedding"])
    return np.array(vectors, dtype=np.float32)


def embed_query(text: str) -> np.ndarray:
    """Embed a single query string. Returns shape (1, EMBED_DIM)."""
    return embed_texts([text])


# ── MarkItDown parser ─────────────────────────────────────────────────────────

def parse_with_markitdown(file_bytes: bytes, filename: str) -> str:
    """
    Parse any supported file (PDF, DOCX, PPTX, XLSX, HTML, TXT, MD, images)
    to clean Markdown text using Microsoft MarkItDown.

    Returns the full markdown string.
    """
    from markitdown import MarkItDown

    md = MarkItDown()

    # Write to a temp file – MarkItDown works from file paths
    suffix = Path(filename).suffix or ".bin"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        result = md.convert(tmp_path)
        return result.text_content
    finally:
        os.unlink(tmp_path)


# ── Text chunker ──────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping chunks by character count.
    Simple, dependency-light implementation.
    """
    if not text.strip():
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# ── FAISS index helpers ───────────────────────────────────────────────────────

def _ensure_faiss_dir():
    FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)


def load_faiss_index():
    """
    Load the FAISS index and metadata from disk.
    Returns (index, metadata_list) or (None, []) if not found.
    metadata_list is a list of dicts, one per vector.
    """
    import faiss

    if not FAISS_INDEX_FILE.exists():
        return None, []

    index = faiss.read_index(str(FAISS_INDEX_FILE))
    with open(FAISS_META_FILE, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata


def save_faiss_index(index, metadata: list[dict]):
    """Persist the FAISS index and metadata to disk."""
    import faiss

    _ensure_faiss_dir()
    faiss.write_index(index, str(FAISS_INDEX_FILE))
    with open(FAISS_META_FILE, "wb") as f:
        pickle.dump(metadata, f)


def append_to_faiss(vectors: np.ndarray, new_metadata: list[dict]) -> int:
    """
    Append new vectors + metadata to the existing FAISS index (or create one).
    Returns the total number of vectors in the index after appending.
    """
    import faiss

    index, metadata = load_faiss_index()

    if index is None:
        # Create a new flat L2 index (cosine via normalised vectors)
        index = faiss.IndexFlatIP(vectors.shape[1])  # Inner Product = cosine if normalised

    # Normalise for cosine similarity
    faiss.normalize_L2(vectors)
    index.add(vectors)
    metadata.extend(new_metadata)

    save_faiss_index(index, metadata)
    return index.ntotal


def search_faiss(query_vector: np.ndarray, top_k: int = 5) -> list[dict]:
    """
    Search the FAISS index for the top_k nearest neighbours.
    Returns a list of metadata dicts with an added 'score' key.
    """
    import faiss

    index, metadata = load_faiss_index()
    if index is None or index.ntotal == 0:
        return []

    faiss.normalize_L2(query_vector)
    scores, indices = index.search(query_vector, min(top_k, index.ntotal))

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        entry = dict(metadata[idx])
        entry["score"] = float(score)
        results.append(entry)
    return results


def get_all_doc_ids() -> list[dict]:
    """Return unique documents stored in FAISS metadata."""
    _, metadata = load_faiss_index()
    seen = {}
    for m in metadata:
        doc_id = m.get("doc_id", "")
        if doc_id not in seen:
            seen[doc_id] = {"doc_id": doc_id, "filename": m.get("filename", ""), "chunk_count": 0}
        seen[doc_id]["chunk_count"] += 1
    return list(seen.values())


# ── S3 helpers ────────────────────────────────────────────────────────────────

def upload_to_s3(file_bytes: bytes, filename: str, doc_id: str, extra_meta: dict | None = None) -> str:
    """Upload raw file bytes to S3. Returns the S3 key."""
    s3 = get_s3_client()
    key = f"{S3_DOCS_PREFIX}{doc_id}/{filename}"
    meta = {"doc_id": doc_id}
    if extra_meta:
        meta.update({k: str(v) for k, v in extra_meta.items()})
    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=file_bytes, Metadata=meta)
    return key


def list_s3_documents() -> list[dict]:
    """List all files under S3_DOCS_PREFIX. Returns [{key, size, last_modified}]."""
    s3 = get_s3_client()
    paginator = s3.get_paginator("list_objects_v2")
    docs = []
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_DOCS_PREFIX):
        for obj in page.get("Contents", []):
            docs.append({
                "key": obj["Key"],
                "size_bytes": obj["Size"],
                "last_modified": obj["LastModified"].isoformat(),
            })
    return docs


def download_from_s3(s3_key: str) -> bytes:
    """Download a file from S3 and return its bytes."""
    s3 = get_s3_client()
    resp = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)
    return resp["Body"].read()


# ── Summarization ─────────────────────────────────────────────────────────────

def summarize_with_claude(text: str, max_chars: int = 15000) -> str:
    """
    Summarize text using Claude 3.5 Sonnet via Amazon Bedrock.
    Caps input to max_chars to stay within token limits.
    """
    import json as _json

    client = get_bedrock_client()
    truncated = text[:max_chars]

    prompt = (
        "You are a document analyst. Read the document below and provide a clear, "
        "structured summary covering:\n"
        "1. Main topic / purpose\n"
        "2. Key points or findings\n"
        "3. Any conclusions or action items\n\n"
        f"DOCUMENT:\n{truncated}"
    )

    body = _json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "temperature": 0.2,
        "messages": [{"role": "user", "content": prompt}],
    })

    resp = client.invoke_model(
        modelId=LLM_MODEL_ID,
        body=body,
        contentType="application/json",
        accept="application/json",
    )
    result = _json.loads(resp["body"].read())
    return result["content"][0]["text"]


# ── Doc ID generator ──────────────────────────────────────────────────────────

def make_doc_id(filename: str) -> str:
    """Generate a short deterministic-ish doc ID from filename + timestamp."""
    raw = f"{filename}{time.time()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]
