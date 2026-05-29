"""
MCP Server – Document Intelligence (Local)
==========================================
Exposes tools via FastMCP using stdio transport.
Start this server before launching the Chainlit app.

Tools:
  1. upload_and_process_document  – S3 upload + MarkItDown parse + FAISS embed + Claude summary
  2. rag_search                   – FAISS k-NN retrieval + Claude grounded answer
  3. build_index_from_s3          – Bootstrap FAISS index from existing S3 documents
  4. search_relevant_pages        – FAISS page/chunk search with optional filters
  5. list_stored_documents        – List all indexed documents
  6. get_document_chunks          – Retrieve all chunks for a document

Run:
    cd mcp_server
    python server.py
"""

import sys
import os

# Ensure mcp_server/ is on the path so relative imports work
sys.path.insert(0, os.path.dirname(__file__))

from mcp.server.fastmcp import FastMCP

from tool1_document import process_document
from tool2_rag_search import build_index_from_s3 as _build_index, rag_search as _rag_search
from tool3_doc_storage import (
    get_document_chunks as _get_chunks,
    list_stored_documents as _list_docs,
    search_relevant_pages as _search_pages,
)

mcp = FastMCP("DocumentIntelligenceMCP")


# ─────────────────────────────────────────────────────────────────────────────
# Tool 1 – Upload, Parse, Embed, Summarize
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool(
    name="upload_and_process_document",
    description=(
        "Upload a document (PDF, DOCX, PPTX, TXT, MD, HTML, images) to S3, "
        "parse it with MarkItDown, generate Titan embeddings for each chunk, "
        "append them to the FAISS vector index, and return a Claude 3.5 Sonnet "
        "summary. Use this when the user attaches or uploads a new document."
    ),
)
def upload_and_process_document(
    file_b64: str,
    filename: str,
    metadata: dict | None = None,
) -> dict:
    """
    Args:
        file_b64:  Base64-encoded file bytes.
        filename:  Original filename with extension.
        metadata:  Optional extra metadata dict.
    Returns:
        {doc_id, filename, s3_key, chunk_count, total_vectors, summary}
    """
    return process_document(file_b64=file_b64, filename=filename, metadata=metadata)


# ─────────────────────────────────────────────────────────────────────────────
# Tool 2 – RAG Search
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool(
    name="rag_search",
    description=(
        "Answer a question using RAG: embed the query with Titan, retrieve the "
        "top-k most relevant chunks from the FAISS index, then generate a grounded "
        "answer with Claude 3.5 Sonnet. Use this when the user asks a question "
        "about document content or requests an explanation."
    ),
)
def rag_search_tool(query: str, top_k: int = 5) -> dict:
    """
    Args:
        query:  Natural-language question.
        top_k:  Number of chunks to retrieve (default 5).
    Returns:
        {query, answer, retrieved_chunks, sources}
    """
    return _rag_search(query=query, top_k=top_k)


@mcp.tool(
    name="build_index_from_s3",
    description=(
        "Bootstrap or rebuild the FAISS vector index by processing all existing "
        "documents in the S3 documents/ folder. Run this once on first setup or "
        "when you want to re-index all S3 content."
    ),
)
def build_index_from_s3_tool() -> dict:
    """
    Returns:
        {processed_files, total_chunks, total_vectors}
    """
    return _build_index()


# ─────────────────────────────────────────────────────────────────────────────
# Tool 3 – Document Storage & Page Search
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool(
    name="search_relevant_pages",
    description=(
        "Search the FAISS vector index for the most relevant document pages or "
        "chunks matching a query. Returns ranked results with similarity scores "
        "and text excerpts. Optionally filter by doc_id or filename. "
        "Use this when the user wants to find specific passages or pages."
    ),
)
def search_relevant_pages_tool(
    query: str,
    top_k: int = 5,
    doc_id: str | None = None,
    filename: str | None = None,
) -> dict:
    """
    Args:
        query:    Search query.
        top_k:    Number of results (default 5).
        doc_id:   Optional filter by document ID.
        filename: Optional filter by filename.
    Returns:
        {query, total_found, results}
    """
    return _search_pages(query=query, top_k=top_k, doc_id=doc_id, filename=filename)


@mcp.tool(
    name="list_stored_documents",
    description=(
        "List all documents currently indexed in FAISS, with doc_id, filename, "
        "and chunk count. Use this to see what documents are available for search."
    ),
)
def list_stored_documents_tool() -> dict:
    """
    Returns:
        {total_documents, total_vectors, documents}
    """
    return _list_docs()


@mcp.tool(
    name="get_document_chunks",
    description=(
        "Retrieve all stored text chunks for a specific document by its doc_id. "
        "Useful for inspecting what was indexed for a particular file."
    ),
)
def get_document_chunks_tool(doc_id: str) -> dict:
    """
    Args:
        doc_id: Document ID to retrieve chunks for.
    Returns:
        {doc_id, filename, chunk_count, chunks}
    """
    return _get_chunks(doc_id=doc_id)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[MCP Server] Starting DocumentIntelligenceMCP on stdio...", flush=True)
    mcp.run(transport="stdio")
