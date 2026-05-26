"""
MCP Server – Document Intelligence
====================================
Exposes three MCP tools via FastMCP (stdio transport):

  • upload_and_process_document  (Tool 1)
  • rag_search                   (Tool 2)
  • search_relevant_pages        (Tool 3 – search)
  • store_document_pages         (Tool 3 – store)
  • list_stored_documents        (Tool 3 – list)

Run:
    python mcp_server/server.py
"""

from mcp.server.fastmcp import FastMCP

from tool1_document import upload_parse_embed_summarize
from tool2_rag_search import rag_search as _rag_search
from tool3_doc_storage import (
    list_stored_documents as _list_stored_documents,
    search_relevant_pages as _search_relevant_pages,
    store_document_pages as _store_document_pages,
)

mcp = FastMCP("DocumentIntelligenceMCP")


# ─────────────────────────────────────────────────────────────────────────────
# Tool 1 – Document Upload, Parsing, Embeddings & Summarization
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool(
    name="upload_and_process_document",
    description=(
        "Upload a document (PDF, DOCX, TXT, MD), parse its text, generate vector "
        "embeddings for each chunk, store them in OpenSearch, and return a summary "
        "of the document produced by Claude 3 Sonnet. "
        "Use this tool when the user uploads or mentions a new document."
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
        filename:  Original filename with extension (e.g. 'report.pdf').
        metadata:  Optional dict of extra metadata (author, department, etc.).

    Returns:
        {doc_id, filename, s3_key, page_count, chunk_count, summary}
    """
    return upload_parse_embed_summarize(
        file_b64=file_b64,
        filename=filename,
        metadata=metadata,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tool 2 – RAG Search (Bedrock Knowledge Base)
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool(
    name="rag_search",
    description=(
        "Answer a question using Retrieval-Augmented Generation (RAG) backed by "
        "Amazon Bedrock Knowledge Base. Retrieves the most relevant document chunks "
        "and generates a grounded answer with Claude 3 Sonnet. "
        "Use this tool when the user asks a question or requests a summary/explanation "
        "about content that has already been ingested."
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


# ─────────────────────────────────────────────────────────────────────────────
# Tool 3 – Document Storage & Relevant Page Search
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool(
    name="store_document_pages",
    description=(
        "Store pre-parsed document pages or text chunks directly into the vector "
        "store (OpenSearch Serverless). Use this when you have already extracted "
        "text and want to index it without re-uploading the original file."
    ),
)
def store_document_pages_tool(
    doc_id: str,
    filename: str,
    pages: list,
) -> dict:
    """
    Args:
        doc_id:    Unique document identifier.
        filename:  Original filename.
        pages:     List of {text, page_number, metadata} dicts.

    Returns:
        {doc_id, filename, stored_count, index_name}
    """
    return _store_document_pages(doc_id=doc_id, filename=filename, pages=pages)


@mcp.tool(
    name="search_relevant_pages",
    description=(
        "Search the vector store for the most relevant document pages or chunks "
        "matching a query. Returns ranked results with similarity scores and text "
        "excerpts. Optionally filter by doc_id or filename. "
        "Use this tool when the user wants to retrieve specific pages or passages "
        "from stored documents."
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
        query:    Natural-language search query.
        top_k:    Number of results (default 5).
        doc_id:   Optional filter by document ID.
        filename: Optional filter by filename.

    Returns:
        {query, total_found, results: [{rank, score, doc_id, filename, page_number, text_excerpt}]}
    """
    return _search_relevant_pages(
        query=query,
        top_k=top_k,
        doc_id=doc_id,
        filename=filename,
    )


@mcp.tool(
    name="list_stored_documents",
    description=(
        "List all documents currently stored in the vector index, with their "
        "doc_id, filename, and chunk count. Use this to discover what documents "
        "are available for search."
    ),
)
def list_stored_documents_tool() -> dict:
    """
    Returns:
        {total_documents, documents: [{doc_id, filename, chunk_count}]}
    """
    return _list_stored_documents()


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # stdio transport – consumed by the LangChain MultiServerMCPClient
    mcp.run(transport="stdio")
