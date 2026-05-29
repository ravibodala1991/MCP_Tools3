# Document Intelligence MCP – End-to-End Project

An MCP-based document intelligence system where a LangGraph Agent automatically
routes user requests to the correct tool based on intent.

Built with: **Microsoft MarkItDown** + **FAISS** (local vector store) + **Amazon Bedrock** (Claude 3.5 Sonnet + Titan Embed v2) + **Chainlit** UI + **FastMCP** server.

---

## Architecture

```
User (Browser)
    |
    | http://localhost:8080
    v
Chainlit App  (chainlit_app.py)
    |
    | in-process
    v
LangGraph ReAct Agent
    | Claude 3.5 Sonnet (Bedrock)
    |
    | stdio subprocess
    | MultiServerMCPClient (langchain-mcp-adapters)
    v
MCP Server  (mcp_server/server.py)  [FastMCP]
    |
    +-- Tool 1: upload_and_process_document
    |     S3 upload -> MarkItDown parse -> Titan Embed -> FAISS append -> Claude summary
    |
    +-- Tool 2: rag_search
    |     FAISS k-NN search -> Claude grounded answer
    |
    +-- Tool 2: build_index_from_s3
    |     Bootstrap FAISS from existing S3 documents
    |
    +-- Tool 3: search_relevant_pages
    |     FAISS k-NN -> ranked page excerpts
    |
    +-- Tool 3: list_stored_documents
    |     Aggregate unique docs from FAISS metadata
    |
    +-- Tool 3: get_document_chunks
          All chunks for a specific doc_id

AWS Services used:
  - Amazon S3          (raw document storage)
  - Amazon Bedrock     (Titan Embed v2 + Claude 3.5 Sonnet)

Local storage:
  - faiss_index/index.faiss    (vector index)
  - faiss_index/metadata.pkl   (chunk metadata)
```

---

## Tool Routing

| User says... | Tool called |
|---|---|
| Uploads a file | `upload_and_process_document` |
| Asks a question about documents | `rag_search` |
| "find pages about X" | `search_relevant_pages` |
| "what documents are indexed?" | `list_stored_documents` |
| "rebuild index from S3" | `build_index_from_s3` |
| "show chunks for doc X" | `get_document_chunks` |

---

## How to Run Locally (Sequential Steps)

### Step 1 — Prerequisites (one-time)

Make sure you have Python 3.10+ and AWS CLI installed and configured.

```cmd
python --version
aws sts get-caller-identity
```

### Step 2 — Enable Bedrock Model Access (one-time, AWS Console)

Go to **AWS Console → Amazon Bedrock → Model access** and enable:
- `Claude 3.5 Sonnet`  (`anthropic.claude-3-5-sonnet-20241022-v2:0`)
- `Titan Text Embeddings V2`  (`amazon.titan-embed-text-v2:0`)

### Step 3 — Create S3 Bucket (one-time)

```cmd
aws s3 mb s3://mcp-raw-docs --region us-east-1
```

### Step 4 — Install Python dependencies

```cmd
cd c:\Users\user\Documents\mcp
pip install -r requirements.txt
```

### Step 5 — Create your .env file

```cmd
copy .env.example .env
```

Edit `.env` and set at minimum:

```
AWS_REGION=us-east-1
S3_BUCKET=mcp-raw-docs
FAISS_INDEX_DIR=./faiss_index
```

### Step 6 — Start the Chainlit app

```cmd
chainlit run chainlit_app.py --port 8080
```

This single command starts everything:
- Chainlit web server on port 8080
- MCP server (`mcp_server/server.py`) auto-spawned as a subprocess
- LangGraph agent with all 6 tools loaded

### Step 7 — Open browser

```
http://localhost:8080
```

### Step 8 — (First time) Bootstrap index from existing S3 data

If you have existing documents in S3, type in the chat:
```
rebuild index from S3
```

Skip this if starting fresh — Tool 1 builds the index automatically on first upload.

### Step 9 — Test

```
1. Drag and drop a PDF/DOCX/TXT file into the chat
   -> Tool 1: S3 upload + MarkItDown parse + FAISS embed + Claude summary

2. Ask a question about the document
   -> Tool 2: FAISS search + Claude grounded answer

3. Find specific pages
   -> Type: "find pages about <topic>"
   -> Tool 3: FAISS k-NN search + ranked excerpts

4. List indexed documents
   -> Type: "what documents are indexed?"
   -> Tool 3: list_stored_documents
```

---

## Quick Reference

| Step | Command | Frequency |
|---|---|---|
| Check Python + AWS | `python --version` / `aws sts get-caller-identity` | One-time |
| Enable Bedrock models | AWS Console | One-time |
| Create S3 bucket | `aws s3 mb s3://mcp-raw-docs` | One-time |
| Install deps | `pip install -r requirements.txt` | One-time |
| Create `.env` | `copy .env.example .env` | One-time |
| Start app | `chainlit run chainlit_app.py --port 8080` | Every run |
| Open browser | `http://localhost:8080` | Every run |
| Bootstrap index | Chat: `rebuild index from S3` | Only if existing S3 data |

---

## Project Structure

```
mcp/
├── mcp_server/
│   ├── server.py              # FastMCP server – 6 registered tools (stdio)
│   ├── tool1_document.py      # Upload + MarkItDown parse + FAISS embed + Claude summary
│   ├── tool2_rag_search.py    # FAISS RAG search + Claude answer + S3 bootstrap
│   ├── tool3_doc_storage.py   # FAISS page search, list docs, get chunks
│   └── utils.py               # Shared: AWS clients, MarkItDown, FAISS, embedder, chunker
├── agent/
│   └── agent.py               # Standalone LangGraph agent (alternative entry point)
├── infra/
│   ├── app.py                 # CDK app entry point
│   └── stacks/mcp_stack.py    # AWS CDK stack (S3, OpenSearch, Bedrock KB, IAM)
├── chainlit_app.py            # Chainlit UI – main entry point for local testing
├── local_architecture.txt     # Detailed local architecture diagram
├── HLD.txt                    # High-level design diagram
├── test_tools.py              # Unit + integration tests
├── requirements.txt           # Python dependencies
└── .env.example               # Environment variable template
```

---

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `AWS_REGION` | AWS region | `us-east-1` |
| `S3_BUCKET` | S3 bucket for raw documents | `mcp-raw-docs` |
| `S3_DOCS_PREFIX` | S3 key prefix for documents | `documents/` |
| `FAISS_INDEX_DIR` | Local directory for FAISS index | `./faiss_index` |
| `LLM_MODEL_ID` | Bedrock LLM model ID | `anthropic.claude-3-5-sonnet-20241022-v2:0` |
| `EMBED_MODEL_ID` | Bedrock embedding model ID | `amazon.titan-embed-text-v2:0` |
| `CHUNK_SIZE` | Text chunk size in characters | `800` |
| `CHUNK_OVERLAP` | Chunk overlap in characters | `150` |
| `RAG_TOP_K` | Chunks to retrieve for RAG | `5` |
