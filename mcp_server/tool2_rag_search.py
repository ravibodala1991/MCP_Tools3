"""
Tool 2 – RAG Search via Amazon Bedrock Knowledge Base
======================================================
Uses LangChain's AmazonKnowledgeBasesRetriever to query an existing
Bedrock Knowledge Base, then passes the retrieved context to Claude 3
Sonnet to generate a grounded answer.

Reference: https://python.langchain.com/v0.1/docs/integrations/retrievers/bedrock/
"""

from __future__ import annotations

import os
from typing import Any

import boto3
from langchain_aws import AmazonKnowledgeBasesRetriever
from langchain_aws.chat_models import ChatBedrock
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

# ── env / config ──────────────────────────────────────────────────────────────
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
KNOWLEDGE_BASE_ID = os.environ.get("KNOWLEDGE_BASE_ID", "")
LLM_MODEL_ID = os.environ.get("LLM_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0")
RAG_TOP_K = int(os.environ.get("RAG_TOP_K", "5"))

# ── AWS clients ───────────────────────────────────────────────────────────────
bedrock_runtime = boto3.client("bedrock-runtime", region_name=AWS_REGION)


def _build_rag_chain(top_k: int = RAG_TOP_K):
    """
    Build a LangChain RAG chain backed by Bedrock Knowledge Base.

    Returns a runnable that accepts {"query": str} and returns a str answer.
    """
    retriever = AmazonKnowledgeBasesRetriever(
        knowledge_base_id=KNOWLEDGE_BASE_ID,
        retrieval_config={"vectorSearchConfiguration": {"numberOfResults": top_k}},
        region_name=AWS_REGION,
    )

    llm = ChatBedrock(
        client=bedrock_runtime,
        model_id=LLM_MODEL_ID,
        region_name=AWS_REGION,
        model_kwargs={"max_tokens": 2048, "temperature": 0.1},
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a helpful assistant. Answer the user's question using ONLY "
                    "the context provided below. If the answer is not in the context, "
                    "say 'I don't have enough information to answer that.'\n\n"
                    "CONTEXT:\n{context}"
                ),
            ),
            ("human", "{query}"),
        ]
    )

    def format_docs(docs):
        return "\n\n---\n\n".join(
            f"[Source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
            for d in docs
        )

    chain = (
        {"context": retriever | format_docs, "query": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever


# ── Main tool function ────────────────────────────────────────────────────────

def rag_search(query: str, top_k: int = RAG_TOP_K) -> dict[str, Any]:
    """
    Perform a RAG search against the Bedrock Knowledge Base.

    Args:
        query:  Natural-language question to answer.
        top_k:  Number of document chunks to retrieve (default 5).

    Returns:
        dict with answer, sources (list of dicts), and retrieved_chunks count.
    """
    if not KNOWLEDGE_BASE_ID:
        raise EnvironmentError(
            "KNOWLEDGE_BASE_ID environment variable is not set. "
            "Deploy the CDK stack first and set the variable."
        )

    chain, retriever = _build_rag_chain(top_k=top_k)

    # Retrieve source docs separately so we can return them
    source_docs = retriever.invoke(query)
    sources = [
        {
            "source": d.metadata.get("source", "unknown"),
            "score": d.metadata.get("score"),
            "excerpt": d.page_content[:300],
        }
        for d in source_docs
    ]

    # Generate grounded answer
    answer = chain.invoke(query)

    return {
        "query": query,
        "answer": answer,
        "retrieved_chunks": len(source_docs),
        "sources": sources,
    }
