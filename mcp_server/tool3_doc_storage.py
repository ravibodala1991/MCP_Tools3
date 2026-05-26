"""
Tool 3 – Document Storage & Relevant Page Search
=================================================
Provides two capabilities:
  A. store_document_pages  – persist individual pages/chunks to OpenSearch
     (useful when you already have parsed text and want to index it directly).
  B. search_relevant_pages – k-NN vector search to find the most relevant
     pages/chunks for a given query, returning ranked results with scores.

Reference: LangChain OpenSearch vector store integration +
           Amazon Titan Embeddings v2 for query embedding.
"""

from __future__ import annotations

import os
from typing import Any

import boto3
from langchain_aws import BedrockEmbeddings
from langchain_community.vectorstores import OpenSearchVectorSearch
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth

# ── env / config ──────────────────────────────────────────────────────────────
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
OPENSEARCH_ENDPOINT = os.environ.get("OPENSEARCH_ENDPOINT", "")  # host only
OPENSEARCH_INDEX = os.environ.get("OPENSEARCH_INDEX", "mcp-documents")
EMBED_MODEL_ID = os.environ.get("EMBED_MODEL_ID", "amazon.titan-embed-text-v2:0")
DEFAULT_TOP_K = int(os.environ.get("SEARCH_TOP_K", "5"))

# ── AWS clients ───────────────────────────────────────────────────────────────
bedrock_runtime = boto3.client("bedrock-runtime", region_name=AWS_REGION)
credentials = boto3.Session().get_credentials()
awsauth = AWS4Auth(
    credentials.access_key,
    credentials.secret_key,
    AWS_REGION,
    "aoss",
    session_token=credentials.token,
)


def _get_embeddings() -> BedrockEmbeddings:
    return BedrockEmbeddings(
        client=bedrock_runtime,
        model_id=EMBED_MODEL_ID,
        region_name=AWS_REGION,
    )


def _get_vector_store() -> OpenSearchVectorSearch:
    """Return a LangChain OpenSearchVectorSearch backed by OpenSearch Serverless."""
    return OpenSearchVectorSearch(
        opensearch_url=f"https://{OPENSEARCH_ENDPOINT}",
        index_name=OPENSEARCH_INDEX,
        embedding_function=_get_embeddings(),
        http_auth=awsauth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        timeout=30,
        engine="nmslib",
        space_type="cosinesimil",
        ef_search=512,
        m=16,
    )


def _get_raw_client() -> OpenSearch:
    return OpenSearch(
        hosts=[{"host": OPENSEARCH_ENDPOINT, "port": 443}],
        http_auth=awsauth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        timeout=30,
    )


# ── Tool A: Store pages ───────────────────────────────────────────────────────

def store_document_pages(
    doc_id: str,
    filename: str,
    pages: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Store pre-parsed document pages/chunks into the vector store.

    Args:
        doc_id:    Unique document identifier.
        filename:  Original filename.
        pages:     List of dicts, each with keys:
                     - text (str)       : page/chunk text content
                     - page_number (int): page number (0-indexed)
                     - metadata (dict)  : optional extra metadata

    Returns:
        dict with doc_id, filename, stored_count, index_name.
    """
    if not pages:
        return {"doc_id": doc_id, "filename": filename, "stored_count": 0, "index_name": OPENSEARCH_INDEX}

    texts: list[str] = []
    metadatas: list[dict] = []

    for page in pages:
        texts.append(page["text"])
        metadatas.append(
            {
                "doc_id": doc_id,
                "filename": filename,
                "page_number": page.get("page_number", 0),
                **(page.get("metadata") or {}),
            }
        )

    vector_store = _get_vector_store()
    ids = vector_store.add_texts(texts=texts, metadatas=metadatas)

    return {
        "doc_id": doc_id,
        "filename": filename,
        "stored_count": len(ids),
        "index_name": OPENSEARCH_INDEX,
    }


# ── Tool B: Search relevant pages ────────────────────────────────────────────

def search_relevant_pages(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    doc_id: str | None = None,
    filename: str | None = None,
) -> dict[str, Any]:
    """
    Search for the most relevant document pages/chunks for a given query.

    Args:
        query:    Natural-language search query.
        top_k:    Number of results to return (default 5).
        doc_id:   Optional – filter results to a specific document ID.
        filename: Optional – filter results to a specific filename.

    Returns:
        dict with query, results (list of ranked pages with scores), total_found.
    """
    vector_store = _get_vector_store()

    # Build optional pre-filter for OpenSearch k-NN
    pre_filter: dict | None = None
    filter_clauses: list[dict] = []
    if doc_id:
        filter_clauses.append({"term": {"metadata.doc_id": doc_id}})
    if filename:
        filter_clauses.append({"term": {"metadata.filename": filename}})
    if filter_clauses:
        pre_filter = {"bool": {"must": filter_clauses}}

    search_kwargs: dict[str, Any] = {"k": top_k}
    if pre_filter:
        search_kwargs["boolean_filter"] = pre_filter

    docs_with_scores = vector_store.similarity_search_with_score(query, **search_kwargs)

    results = [
        {
            "rank": rank + 1,
            "score": float(score),
            "doc_id": doc.metadata.get("doc_id", ""),
            "filename": doc.metadata.get("filename", ""),
            "page_number": doc.metadata.get("page_number", 0),
            "text_excerpt": doc.page_content[:500],
            "metadata": doc.metadata,
        }
        for rank, (doc, score) in enumerate(docs_with_scores)
    ]

    return {
        "query": query,
        "total_found": len(results),
        "results": results,
    }


# ── Convenience: list all documents ──────────────────────────────────────────

def list_stored_documents() -> dict[str, Any]:
    """
    Return a summary of all unique documents currently stored in the index.

    Returns:
        dict with documents (list of {doc_id, filename, chunk_count}).
    """
    client = _get_raw_client()
    agg_query = {
        "size": 0,
        "aggs": {
            "by_doc": {
                "terms": {"field": "metadata.doc_id", "size": 200},
                "aggs": {
                    "filename": {
                        "terms": {"field": "metadata.filename", "size": 1}
                    }
                },
            }
        },
    }
    response = client.search(index=OPENSEARCH_INDEX, body=agg_query)
    buckets = response.get("aggregations", {}).get("by_doc", {}).get("buckets", [])
    documents = [
        {
            "doc_id": b["key"],
            "filename": b["filename"]["buckets"][0]["key"] if b["filename"]["buckets"] else "unknown",
            "chunk_count": b["doc_count"],
        }
        for b in buckets
    ]
    return {"total_documents": len(documents), "documents": documents}
