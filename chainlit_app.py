"""
Chainlit App – Document Intelligence MCP
=========================================
Local chat UI that connects to the MCP server via MultiServerMCPClient
(langchain-mcp-adapters) and routes user messages to the correct tool
using a LangGraph ReAct agent backed by Amazon Bedrock Claude 3.5 Sonnet.

Features:
  - Drag-and-drop file upload → auto-triggers Tool 1
  - Chat questions → Tool 2 (RAG search)
  - "find pages about X" → Tool 3
  - "list documents" → Tool 3
  - Streaming responses

Run:
    chainlit run chainlit_app.py --port 8080
"""

from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path

import chainlit as cl
from langchain_aws import ChatBedrockConverse
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# ── Config ────────────────────────────────────────────────────────────────────
AWS_REGION   = os.environ.get("AWS_REGION", "us-east-1")
LLM_MODEL_ID = os.environ.get("LLM_MODEL_ID", "anthropic.claude-3-5-sonnet-20241022-v2:0")
MCP_SERVER   = str(Path(__file__).parent / "mcp_server" / "server.py")

SYSTEM_PROMPT = """You are a Document Intelligence Assistant powered by AWS Bedrock.

You have access to these tools – pick the right one based on user intent:

1. upload_and_process_document
   → When the user uploads a file or asks to ingest/process a document.
   → Requires: file_b64 (base64), filename.

2. rag_search
   → When the user asks a question about document content, wants an explanation,
     or requests a summary of already-indexed material.
   → Requires: query.

3. build_index_from_s3
   → When the user asks to rebuild or bootstrap the index from existing S3 files.

4. search_relevant_pages
   → When the user wants specific pages, passages, or sections from documents.
   → Requires: query. Optional: doc_id, filename, top_k.

5. list_stored_documents
   → When the user asks what documents are available or indexed.

6. get_document_chunks
   → When the user wants to inspect all chunks of a specific document.
   → Requires: doc_id.

Always explain what you are doing. Present results clearly and concisely.
When showing sources, include the filename and a short excerpt.
"""

# ── MCP client config ─────────────────────────────────────────────────────────
MCP_CONFIG = {
    "document_intelligence": {
        "command": sys.executable,
        "args": [MCP_SERVER],
        "transport": "stdio",
        "env": {
            "AWS_REGION":      AWS_REGION,
            "S3_BUCKET":       os.environ.get("S3_BUCKET", "mcp-raw-docs"),
            "S3_DOCS_PREFIX":  os.environ.get("S3_DOCS_PREFIX", "documents/"),
            "FAISS_INDEX_DIR": os.environ.get("FAISS_INDEX_DIR", "./faiss_index"),
            "LLM_MODEL_ID":    LLM_MODEL_ID,
            "EMBED_MODEL_ID":  os.environ.get("EMBED_MODEL_ID", "amazon.titan-embed-text-v2:0"),
        },
    }
}


# ── Build LangGraph agent ─────────────────────────────────────────────────────

async def build_agent():
    """Create MCP client, load tools, compile LangGraph graph."""
    client = MultiServerMCPClient(MCP_CONFIG)
    tools = await client.get_tools()

    llm = ChatBedrockConverse(
        model=LLM_MODEL_ID,
        region_name=AWS_REGION,
        temperature=0.1,
        max_tokens=4096,
    )
    llm_with_tools = llm.bind_tools(tools)

    def call_model(state: MessagesState):
        messages = state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    builder = StateGraph(MessagesState)
    builder.add_node("call_model", call_model)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges("call_model", tools_condition)
    builder.add_edge("tools", "call_model")

    return builder.compile()


# ── Chainlit lifecycle ────────────────────────────────────────────────────────

@cl.on_chat_start
async def on_chat_start():
    """Initialise agent and conversation history on session start."""
    graph = await build_agent()
    cl.user_session.set("graph", graph)
    cl.user_session.set("history", [SystemMessage(content=SYSTEM_PROMPT)])

    await cl.Message(
        content=(
            "👋 **Document Intelligence Assistant** ready!\n\n"
            "**What I can do:**\n"
            "- 📎 **Upload a file** (drag & drop) → I'll parse, embed, and summarize it\n"
            "- ❓ **Ask questions** about your documents → RAG-powered answers\n"
            "- 🔍 **Find specific pages** → `find pages about X`\n"
            "- 📋 **List documents** → `what documents are indexed?`\n"
            "- 🔄 **Rebuild index** → `rebuild index from S3`\n\n"
            "Powered by **MarkItDown** + **FAISS** + **Amazon Bedrock Claude 3.5 Sonnet**"
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    """Handle incoming user messages and file uploads."""
    graph = cl.user_session.get("graph")
    history: list = cl.user_session.get("history")

    # ── Handle file uploads ───────────────────────────────────────────────────
    if message.elements:
        for element in message.elements:
            if hasattr(element, "path") and element.path:
                file_path = Path(element.path)
                filename = element.name or file_path.name

                status_msg = await cl.Message(
                    content=f"📎 Processing **{filename}**... (uploading to S3, parsing, embedding)"
                ).send()

                # Read and base64-encode the file
                file_bytes = file_path.read_bytes()
                file_b64 = base64.b64encode(file_bytes).decode()

                # Build a message that tells the agent to process this file
                user_text = (
                    f"Please process this uploaded document.\n"
                    f"filename: {filename}\n"
                    f"file_b64: {file_b64}\n"
                    f"Additional user note: {message.content or 'No additional notes.'}"
                )
                history.append(HumanMessage(content=user_text))

                # Run agent
                thinking = cl.Message(content="")
                await thinking.send()

                result = await graph.ainvoke({"messages": history})
                history = result["messages"]
                cl.user_session.set("history", history)

                # Find last AI response
                final_response = ""
                for msg in reversed(history):
                    if isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", None):
                        final_response = msg.content if isinstance(msg.content, str) else str(msg.content)
                        break

                await thinking.remove()
                await cl.Message(content=final_response or "Document processed successfully.").send()
                return

    # ── Handle text messages ──────────────────────────────────────────────────
    user_input = message.content.strip()
    if not user_input:
        return

    history.append(HumanMessage(content=user_input))

    # Show thinking indicator
    thinking = await cl.Message(content="🤔 Thinking...").send()

    result = await graph.ainvoke({"messages": history})
    history = result["messages"]
    cl.user_session.set("history", history)

    # Extract final AI response (skip tool-call messages)
    final_response = ""
    for msg in reversed(history):
        if isinstance(msg, AIMessage) and msg.content and not getattr(msg, "tool_calls", None):
            final_response = msg.content if isinstance(msg.content, str) else str(msg.content)
            break

    await thinking.remove()
    await cl.Message(content=final_response or "I couldn't generate a response.").send()
