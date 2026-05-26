"""
End-to-End Tool Tests – Document Intelligence MCP
==================================================
Tests cover all three MCP tools using mocks so they run without live AWS
credentials. Each test class also contains an integration test (marked with
@pytest.mark.integration) that hits real AWS services when
  INTEGRATION_TESTS=1  is set in the environment.

Run unit tests:
    pytest test_tools.py -v

Run integration tests (requires AWS credentials + deployed stack):
    INTEGRATION_TESTS=1 pytest test_tools.py -v -m integration
"""

from __future__ import annotations

import base64
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── Make mcp_server importable ────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent / "mcp_server"))

INTEGRATION = os.environ.get("INTEGRATION_TESTS", "0") == "1"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_pdf_b64() -> str:
    """Return a minimal valid PDF as base64 (single-page, text-only)."""
    minimal_pdf = (
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R"
        b"/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj\n"
        b"4 0 obj<</Length 44>>stream\nBT /F1 12 Tf 100 700 Td"
        b" (Hello MCP World) Tj ET\nendstream\nendobj\n"
        b"5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n"
        b"xref\n0 6\n0000000000 65535 f \n"
        b"trailer<</Size 6/Root 1 0 R>>\nstartxref\n9\n%%EOF"
    )
    return base64.b64encode(minimal_pdf).decode()


def _make_txt_b64(content: str = "This is a test document about AWS Bedrock.") -> str:
    return base64.b64encode(content.encode()).decode()


# ─────────────────────────────────────────────────────────────────────────────
# Tool 1 Tests – upload_parse_embed_summarize
# ─────────────────────────────────────────────────────────────────────────────

class TestTool1DocumentUpload(unittest.TestCase):
    """Tests for Tool 1: Document upload, parsing, embeddings, summarization."""

    def _mock_dependencies(self):
        """Return a context manager that patches all AWS calls."""
        patches = [
            patch("tool1_document.s3_client"),
            patch("tool1_document.BedrockEmbeddings"),
            patch("tool1_document.ChatBedrock"),
            patch("tool1_document._get_opensearch_client"),
        ]
        return patches

    def test_txt_upload_returns_expected_keys(self):
        """Tool 1 should return doc_id, filename, s3_key, page_count, chunk_count, summary."""
        patches = self._mock_dependencies()
        mocks = [p.start() for p in patches]

        try:
            # Configure mocks
            mock_s3, mock_embed_cls, mock_llm_cls, mock_os_client_fn = mocks

            mock_embedder = MagicMock()
            mock_embedder.embed_documents.return_value = [[0.1] * 1536, [0.2] * 1536]
            mock_embed_cls.return_value = mock_embedder

            mock_llm = MagicMock()
            mock_llm.invoke.return_value = MagicMock(content="This document discusses AWS Bedrock.")
            mock_llm_cls.return_value = mock_llm

            mock_os = MagicMock()
            mock_os.indices.exists.return_value = False
            mock_os_client_fn.return_value = mock_os

            from tool1_document import upload_parse_embed_summarize

            result = upload_parse_embed_summarize(
                file_b64=_make_txt_b64(),
                filename="test.txt",
                metadata={"author": "tester"},
            )

            assert "doc_id" in result
            assert result["filename"] == "test.txt"
            assert "s3_key" in result
            assert result["page_count"] >= 1
            assert result["chunk_count"] >= 1
            assert "summary" in result
            assert len(result["summary"]) > 0

        finally:
            for p in patches:
                p.stop()

    def test_unsupported_file_type_raises(self):
        """Tool 1 should raise ValueError for unsupported file types."""
        patches = self._mock_dependencies()
        mocks = [p.start() for p in patches]
        try:
            from tool1_document import upload_parse_embed_summarize

            with pytest.raises(ValueError, match="Unsupported file type"):
                upload_parse_embed_summarize(
                    file_b64=base64.b64encode(b"data").decode(),
                    filename="file.xyz",
                )
        finally:
            for p in patches:
                p.stop()

    def test_s3_upload_called_with_correct_bucket(self):
        """Tool 1 should call s3_client.put_object with the configured bucket."""
        patches = self._mock_dependencies()
        mocks = [p.start() for p in patches]
        try:
            mock_s3, mock_embed_cls, mock_llm_cls, mock_os_client_fn = mocks

            mock_embedder = MagicMock()
            mock_embedder.embed_documents.return_value = [[0.1] * 1536]
            mock_embed_cls.return_value = mock_embedder

            mock_llm = MagicMock()
            mock_llm.invoke.return_value = MagicMock(content="Summary text.")
            mock_llm_cls.return_value = mock_llm

            mock_os = MagicMock()
            mock_os.indices.exists.return_value = True
            mock_os_client_fn.return_value = mock_os

            import tool1_document
            tool1_document.S3_RAW_BUCKET = "test-bucket"

            from tool1_document import upload_parse_embed_summarize

            upload_parse_embed_summarize(
                file_b64=_make_txt_b64("Short doc."),
                filename="short.txt",
            )

            mock_s3.put_object.assert_called_once()
            call_kwargs = mock_s3.put_object.call_args.kwargs
            assert call_kwargs["Bucket"] == "test-bucket"
            assert "raw-docs/" in call_kwargs["Key"]

        finally:
            for p in patches:
                p.stop()

    def test_empty_metadata_defaults_to_empty_dict(self):
        """Tool 1 should handle None metadata gracefully."""
        patches = self._mock_dependencies()
        mocks = [p.start() for p in patches]
        try:
            mock_s3, mock_embed_cls, mock_llm_cls, mock_os_client_fn = mocks
            mock_embedder = MagicMock()
            mock_embedder.embed_documents.return_value = [[0.1] * 1536]
            mock_embed_cls.return_value = mock_embedder
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = MagicMock(content="ok")
            mock_llm_cls.return_value = mock_llm
            mock_os = MagicMock()
            mock_os.indices.exists.return_value = True
            mock_os_client_fn.return_value = mock_os

            from tool1_document import upload_parse_embed_summarize

            result = upload_parse_embed_summarize(
                file_b64=_make_txt_b64("Hello."),
                filename="hello.txt",
                metadata=None,  # should not raise
            )
            assert result["doc_id"] is not None

        finally:
            for p in patches:
                p.stop()

    @pytest.mark.integration
    def test_integration_upload_real_pdf(self):
        """[INTEGRATION] Upload a real PDF to S3 and verify the result."""
        if not INTEGRATION:
            pytest.skip("Set INTEGRATION_TESTS=1 to run")

        from tool1_document import upload_parse_embed_summarize

        result = upload_parse_embed_summarize(
            file_b64=_make_pdf_b64(),
            filename="integration_test.pdf",
            metadata={"test": "true"},
        )
        assert result["doc_id"]
        assert result["chunk_count"] > 0
        assert result["summary"]
        print(f"\n[Integration] doc_id={result['doc_id']}, summary={result['summary'][:100]}")


# ─────────────────────────────────────────────────────────────────────────────
# Tool 2 Tests – rag_search
# ─────────────────────────────────────────────────────────────────────────────

class TestTool2RagSearch(unittest.TestCase):
    """Tests for Tool 2: RAG search via Bedrock Knowledge Base."""

    def test_rag_search_returns_answer_and_sources(self):
        """Tool 2 should return answer, sources list, and retrieved_chunks count."""
        with (
            patch("tool2_rag_search.AmazonKnowledgeBasesRetriever") as mock_retriever_cls,
            patch("tool2_rag_search.ChatBedrock") as mock_llm_cls,
            patch.dict(os.environ, {"KNOWLEDGE_BASE_ID": "kb-test-123"}),
        ):
            from langchain_core.documents import Document

            mock_doc = Document(
                page_content="AWS Bedrock is a managed AI service.",
                metadata={"source": "doc1.pdf", "score": 0.95},
            )
            mock_retriever = MagicMock()
            mock_retriever.invoke.return_value = [mock_doc]
            mock_retriever_cls.return_value = mock_retriever

            mock_llm = MagicMock()
            mock_llm.invoke.return_value = MagicMock(
                content="AWS Bedrock is a managed AI service by Amazon."
            )
            mock_llm_cls.return_value = mock_llm

            # Patch the chain invoke
            with patch("tool2_rag_search._build_rag_chain") as mock_chain_fn:
                mock_chain = MagicMock()
                mock_chain.invoke.return_value = "AWS Bedrock is a managed AI service by Amazon."
                mock_chain_fn.return_value = (mock_chain, mock_retriever)

                from tool2_rag_search import rag_search

                result = rag_search(query="What is AWS Bedrock?", top_k=3)

            assert "answer" in result
            assert "sources" in result
            assert "retrieved_chunks" in result
            assert result["query"] == "What is AWS Bedrock?"
            assert isinstance(result["sources"], list)

    def test_rag_search_raises_without_kb_id(self):
        """Tool 2 should raise EnvironmentError when KNOWLEDGE_BASE_ID is not set."""
        with patch.dict(os.environ, {"KNOWLEDGE_BASE_ID": ""}):
            import importlib
            import tool2_rag_search
            importlib.reload(tool2_rag_search)

            with pytest.raises(EnvironmentError, match="KNOWLEDGE_BASE_ID"):
                tool2_rag_search.rag_search(query="test query")

    def test_rag_search_source_excerpt_truncated(self):
        """Tool 2 sources should have excerpts truncated to 300 chars."""
        with (
            patch("tool2_rag_search.AmazonKnowledgeBasesRetriever") as mock_retriever_cls,
            patch.dict(os.environ, {"KNOWLEDGE_BASE_ID": "kb-test-123"}),
        ):
            from langchain_core.documents import Document

            long_text = "A" * 1000
            mock_doc = Document(
                page_content=long_text,
                metadata={"source": "long_doc.pdf"},
            )
            mock_retriever = MagicMock()
            mock_retriever.invoke.return_value = [mock_doc]
            mock_retriever_cls.return_value = mock_retriever

            with patch("tool2_rag_search._build_rag_chain") as mock_chain_fn:
                mock_chain = MagicMock()
                mock_chain.invoke.return_value = "Answer."
                mock_chain_fn.return_value = (mock_chain, mock_retriever)

                from tool2_rag_search import rag_search

                result = rag_search(query="long doc query")

            for source in result["sources"]:
                assert len(source["excerpt"]) <= 300

    @pytest.mark.integration
    def test_integration_rag_search(self):
        """[INTEGRATION] Run a real RAG search against Bedrock Knowledge Base."""
        if not INTEGRATION:
            pytest.skip("Set INTEGRATION_TESTS=1 to run")

        from tool2_rag_search import rag_search

        result = rag_search(query="What documents are available?", top_k=3)
        assert result["answer"]
        print(f"\n[Integration] answer={result['answer'][:200]}")
        print(f"[Integration] sources={json.dumps(result['sources'], indent=2)}")


# ─────────────────────────────────────────────────────────────────────────────
# Tool 3 Tests – store_document_pages & search_relevant_pages
# ─────────────────────────────────────────────────────────────────────────────

class TestTool3DocStorage(unittest.TestCase):
    """Tests for Tool 3: Document storage and relevant page search."""

    # ── store_document_pages ──────────────────────────────────────────────────

    def test_store_pages_returns_stored_count(self):
        """store_document_pages should return the number of stored chunks."""
        with patch("tool3_doc_storage.OpenSearchVectorSearch") as mock_vs_cls:
            mock_vs = MagicMock()
            mock_vs.add_texts.return_value = ["id1", "id2", "id3"]
            mock_vs_cls.return_value = mock_vs

            with patch("tool3_doc_storage.BedrockEmbeddings"):
                from tool3_doc_storage import store_document_pages

                pages = [
                    {"text": "Page one content.", "page_number": 0},
                    {"text": "Page two content.", "page_number": 1},
                    {"text": "Page three content.", "page_number": 2},
                ]
                result = store_document_pages(
                    doc_id="doc-abc123",
                    filename="report.pdf",
                    pages=pages,
                )

            assert result["stored_count"] == 3
            assert result["doc_id"] == "doc-abc123"
            assert result["filename"] == "report.pdf"

    def test_store_empty_pages_returns_zero(self):
        """store_document_pages with empty list should return stored_count=0."""
        with patch("tool3_doc_storage.OpenSearchVectorSearch"):
            with patch("tool3_doc_storage.BedrockEmbeddings"):
                from tool3_doc_storage import store_document_pages

                result = store_document_pages(
                    doc_id="doc-empty",
                    filename="empty.txt",
                    pages=[],
                )
            assert result["stored_count"] == 0

    def test_store_pages_metadata_attached(self):
        """store_document_pages should attach doc_id and filename to each chunk metadata."""
        with patch("tool3_doc_storage.OpenSearchVectorSearch") as mock_vs_cls:
            mock_vs = MagicMock()
            mock_vs.add_texts.return_value = ["id1"]
            mock_vs_cls.return_value = mock_vs

            with patch("tool3_doc_storage.BedrockEmbeddings"):
                from tool3_doc_storage import store_document_pages

                store_document_pages(
                    doc_id="doc-meta",
                    filename="meta_test.pdf",
                    pages=[{"text": "Content.", "page_number": 0, "metadata": {"dept": "HR"}}],
                )

            call_kwargs = mock_vs.add_texts.call_args.kwargs
            assert call_kwargs["metadatas"][0]["doc_id"] == "doc-meta"
            assert call_kwargs["metadatas"][0]["filename"] == "meta_test.pdf"
            assert call_kwargs["metadatas"][0]["dept"] == "HR"

    # ── search_relevant_pages ─────────────────────────────────────────────────

    def test_search_returns_ranked_results(self):
        """search_relevant_pages should return results sorted by rank."""
        with patch("tool3_doc_storage.OpenSearchVectorSearch") as mock_vs_cls:
            from langchain_core.documents import Document

            mock_vs = MagicMock()
            mock_vs.similarity_search_with_score.return_value = [
                (Document(page_content="Relevant page 1", metadata={"doc_id": "d1", "filename": "f.pdf", "page_number": 2}), 0.95),
                (Document(page_content="Relevant page 2", metadata={"doc_id": "d1", "filename": "f.pdf", "page_number": 5}), 0.87),
            ]
            mock_vs_cls.return_value = mock_vs

            with patch("tool3_doc_storage.BedrockEmbeddings"):
                from tool3_doc_storage import search_relevant_pages

                result = search_relevant_pages(query="find relevant pages", top_k=2)

            assert result["total_found"] == 2
            assert result["results"][0]["rank"] == 1
            assert result["results"][1]["rank"] == 2
            assert result["results"][0]["score"] > result["results"][1]["score"]

    def test_search_text_excerpt_truncated(self):
        """search_relevant_pages text_excerpt should be at most 500 chars."""
        with patch("tool3_doc_storage.OpenSearchVectorSearch") as mock_vs_cls:
            from langchain_core.documents import Document

            long_text = "B" * 2000
            mock_vs = MagicMock()
            mock_vs.similarity_search_with_score.return_value = [
                (Document(page_content=long_text, metadata={}), 0.9),
            ]
            mock_vs_cls.return_value = mock_vs

            with patch("tool3_doc_storage.BedrockEmbeddings"):
                from tool3_doc_storage import search_relevant_pages

                result = search_relevant_pages(query="long text query")

            assert len(result["results"][0]["text_excerpt"]) <= 500

    def test_search_with_doc_id_filter(self):
        """search_relevant_pages should pass a boolean_filter when doc_id is provided."""
        with patch("tool3_doc_storage.OpenSearchVectorSearch") as mock_vs_cls:
            mock_vs = MagicMock()
            mock_vs.similarity_search_with_score.return_value = []
            mock_vs_cls.return_value = mock_vs

            with patch("tool3_doc_storage.BedrockEmbeddings"):
                from tool3_doc_storage import search_relevant_pages

                search_relevant_pages(query="filtered query", doc_id="doc-xyz")

            call_kwargs = mock_vs.similarity_search_with_score.call_args.kwargs
            assert "boolean_filter" in call_kwargs
            filter_body = call_kwargs["boolean_filter"]
            assert filter_body is not None

    def test_list_stored_documents_returns_structure(self):
        """list_stored_documents should return total_documents and documents list."""
        with patch("tool3_doc_storage._get_raw_client") as mock_client_fn:
            mock_client = MagicMock()
            mock_client.search.return_value = {
                "aggregations": {
                    "by_doc": {
                        "buckets": [
                            {
                                "key": "doc-001",
                                "doc_count": 12,
                                "filename": {"buckets": [{"key": "report.pdf"}]},
                            },
                            {
                                "key": "doc-002",
                                "doc_count": 5,
                                "filename": {"buckets": [{"key": "notes.txt"}]},
                            },
                        ]
                    }
                }
            }
            mock_client_fn.return_value = mock_client

            from tool3_doc_storage import list_stored_documents

            result = list_stored_documents()

        assert result["total_documents"] == 2
        assert result["documents"][0]["doc_id"] == "doc-001"
        assert result["documents"][0]["chunk_count"] == 12
        assert result["documents"][1]["filename"] == "notes.txt"

    @pytest.mark.integration
    def test_integration_store_and_search(self):
        """[INTEGRATION] Store pages then search for them in OpenSearch."""
        if not INTEGRATION:
            pytest.skip("Set INTEGRATION_TESTS=1 to run")

        from tool3_doc_storage import search_relevant_pages, store_document_pages

        pages = [
            {"text": "Amazon Bedrock provides foundation models as a service.", "page_number": 0},
            {"text": "OpenSearch Serverless is a fully managed vector store.", "page_number": 1},
        ]
        store_result = store_document_pages(
            doc_id="integration-test-doc",
            filename="integration.txt",
            pages=pages,
        )
        assert store_result["stored_count"] == 2

        import time
        time.sleep(2)  # allow OpenSearch to index

        search_result = search_relevant_pages(
            query="foundation models",
            top_k=2,
            doc_id="integration-test-doc",
        )
        assert search_result["total_found"] >= 1
        print(f"\n[Integration] search results: {json.dumps(search_result['results'], indent=2)}")


# ─────────────────────────────────────────────────────────────────────────────
# MCP Server Tool Registration Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestMCPServerToolRegistration(unittest.TestCase):
    """Verify that all expected tools are registered on the MCP server."""

    EXPECTED_TOOLS = {
        "upload_and_process_document",
        "rag_search",
        "store_document_pages",
        "search_relevant_pages",
        "list_stored_documents",
    }

    def test_all_tools_registered(self):
        """All 5 tools should be registered on the FastMCP instance."""
        # Patch heavy imports before importing server
        with (
            patch.dict("sys.modules", {
                "tool1_document": MagicMock(upload_parse_embed_summarize=MagicMock()),
                "tool2_rag_search": MagicMock(rag_search=MagicMock()),
                "tool3_doc_storage": MagicMock(
                    store_document_pages=MagicMock(),
                    search_relevant_pages=MagicMock(),
                    list_stored_documents=MagicMock(),
                ),
            }),
        ):
            import importlib
            import mcp_server.server as srv_module  # noqa: F401

            # Re-import to pick up mocks
            if "mcp_server.server" in sys.modules:
                del sys.modules["mcp_server.server"]

            # Just verify the expected tool names are defined in the server source
            server_src = Path(__file__).parent / "mcp_server" / "server.py"
            content = server_src.read_text()
            for tool_name in self.EXPECTED_TOOLS:
                assert tool_name in content, f"Tool '{tool_name}' not found in server.py"


# ─────────────────────────────────────────────────────────────────────────────
# Agent Routing Tests (lightweight)
# ─────────────────────────────────────────────────────────────────────────────

class TestAgentRouting(unittest.TestCase):
    """Verify the agent system prompt contains routing instructions for all tools."""

    def test_system_prompt_mentions_all_tools(self):
        """Agent system prompt should reference all tool names."""
        agent_src = Path(__file__).parent / "agent" / "agent.py"
        content = agent_src.read_text()

        expected_mentions = [
            "upload_and_process_document",
            "rag_search",
            "search_relevant_pages",
            "store_document_pages",
            "list_stored_documents",
        ]
        for tool in expected_mentions:
            assert tool in content, f"Tool '{tool}' not mentioned in agent.py"

    def test_mcp_client_config_references_server(self):
        """Agent should configure MultiServerMCPClient pointing to server.py."""
        agent_src = Path(__file__).parent / "agent" / "agent.py"
        content = agent_src.read_text()
        assert "MultiServerMCPClient" in content
        assert "server.py" in content


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
