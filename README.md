# Document Intelligence MCP – End-to-End Project

An MCP-based document intelligence solution where a LangGraph agent automatically
routes user requests to the correct tool based on intent.

## Architecture

```
User ──► LangGraph Agent (agent/agent.py)
              │
              │  MultiServerMCPClient (langchain-mcp-adapters)
              │
         MCP Server (mcp_server/server.py)
         ├── Tool 1: upload_and_process_document
         │     S3 (raw storage) → PyMuPDF/docx parse → Titan Embeddings → OpenSearch
         │     → Claude 3 Sonnet (summary)
         ├── Tool 2: rag_search
         │     Bedrock Knowledge Base Retriever → Claude 3 Sonnet (grounded answer)
         └── Tool 3: search_relevant_pages / store_document_pages / list_stored_documents
               OpenSearch Serverless (k-NN vector search)

AWS Infrastructure (infra/ – CDK)
  ├── S3 Bucket              – raw document storage
  ├── OpenSearch Serverless  – vector index (HNSW, cosine similarity)
  ├── Bedrock Knowledge Base – managed RAG retrieval
  └── IAM Roles              – least-privilege access
```

## Tool Routing

| User intent | Tool invoked |
|---|---|
| "Upload / process / ingest this document" | `upload_and_process_document` |
| "Summarize / answer a question about content" | `rag_search` |
| "Find pages / retrieve passages about X" | `search_relevant_pages` |
| "What documents are stored?" | `list_stored_documents` |
| "Index these text pages directly" | `store_document_pages` |

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Deploy AWS infrastructure

```bash
cd infra
pip install -r requirements-cdk.txt
cdk bootstrap aws://<ACCOUNT_ID>/us-east-1
cdk deploy --context account=<ACCOUNT_ID> --context region=us-east-1
```

Copy the CDK outputs into your environment:

```bash
cp .env.example .env
# Fill in S3_RAW_BUCKET, OPENSEARCH_ENDPOINT, KNOWLEDGE_BASE_ID from CDK outputs
```

### 3. Enable Bedrock model access

In the AWS Console → Amazon Bedrock → Model access, enable:
- `anthropic.claude-3-sonnet-20240229-v1:0`
- `amazon.titan-embed-text-v2:0`

### 4. Run the agent

```bash
# Load env vars
set -a && source .env && set +a   # Linux/Mac
# or on Windows: set each variable manually

python agent/agent.py
```

### 5. Run tests

```bash
# Unit tests (no AWS required)
pytest test_tools.py -v

# Integration tests (requires deployed stack + AWS credentials)
set INTEGRATION_TESTS=1
pytest test_tools.py -v -m integration
```

## Project Structure

```
mcp/
├── mcp_server/
│   ├── server.py              # FastMCP server – registers all 5 tools
│   ├── tool1_document.py      # Upload, parse, embed, summarize
│   ├── tool2_rag_search.py    # RAG search via Bedrock Knowledge Base
│   └── tool3_doc_storage.py   # Vector store: store & search pages
├── agent/
│   └── agent.py               # LangGraph agent with MultiServerMCPClient
├── infra/
│   ├── app.py                 # CDK app entry point
│   ├── cdk.json               # CDK configuration
│   ├── requirements-cdk.txt   # CDK dependencies
│   └── stacks/
│       └── mcp_stack.py       # Full AWS stack definition
├── test_tools.py              # Unit + integration tests
├── requirements.txt           # Python dependencies
└── .env.example               # Environment variable template
```

## Environment Variables

| Variable | Description |
|---|---|
| `AWS_REGION` | AWS region (default: `us-east-1`) |
| `S3_RAW_BUCKET` | S3 bucket name for raw documents |
| `OPENSEARCH_ENDPOINT` | OpenSearch Serverless collection host |
| `OPENSEARCH_INDEX` | Index name (default: `mcp-documents`) |
| `KNOWLEDGE_BASE_ID` | Bedrock Knowledge Base ID |
| `LLM_MODEL_ID` | Bedrock LLM model ID |
| `EMBED_MODEL_ID` | Bedrock embedding model ID |
| `CHUNK_SIZE` | Text chunk size in tokens (default: `1000`) |
| `CHUNK_OVERLAP` | Chunk overlap in tokens (default: `200`) |
| `RAG_TOP_K` | Chunks to retrieve for RAG (default: `5`) |
| `SEARCH_TOP_K` | Chunks to return for page search (default: `5`) |
