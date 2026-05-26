"""
LangGraph Agent – Document Intelligence
========================================
The agent uses MultiServerMCPClient to dynamically load all MCP tools
from the document-intelligence server, then routes user requests to the
correct tool automatically based on intent:

  • "upload / process / summarize a new document"  → upload_and_process_document
  • "what does the document say about X / answer a question" → rag_search
  • "find pages about X / retrieve passages"        → search_relevant_pages
  • "what documents are stored"                     → list_stored_documents

Usage:
    python agent/agent.py
    # or import and call run_agent(user_message) from your own code
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

# ── config ────────────────────────────────────────────────────────────────────
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
LLM_MODEL_ID = os.environ.get("LLM_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0")

# Absolute path to the MCP server script
MCP_SERVER_PATH = str(
    Path(__file__).resolve().parent.parent / "mcp_server" / "server.py"
)

# ── MCP client config ─────────────────────────────────────────────────────────
MCP_CLIENT_CONFIG = {
    "document_intelligence": {
        "command": sys.executable,          # same Python interpreter
        "args": [MCP_SERVER_PATH],
        "transport": "stdio",
        "env": {
            "AWS_REGION": AWS_REGION,
            "S3_RAW_BUCKET": os.environ.get("S3_RAW_BUCKET", "mcp-raw-docs"),
            "OPENSEARCH_ENDPOINT": os.environ.get("OPENSEARCH_ENDPOINT", ""),
            "OPENSEARCH_INDEX": os.environ.get("OPENSEARCH_INDEX", "mcp-documents"),
            "KNOWLEDGE_BASE_ID": os.environ.get("KNOWLEDGE_BASE_ID", ""),
            "LLM_MODEL_ID": LLM_MODEL_ID,
            "EMBED_MODEL_ID": os.environ.get(
                "EMBED_MODEL_ID", "amazon.titan-embed-text-v2:0"
            ),
        },
    }
}

SYSTEM_PROMPT = """You are a Document Intelligence Assistant powered by AWS Bedrock.

You have access to the following tools – choose the right one based on the user's intent:

1. upload_and_process_document
   → Use when the user uploads a new file or asks to process/ingest/summarize a document.
   → Requires: file_b64 (base64 content), filename.

2. rag_search
   → Use when the user asks a question about document content, requests an explanation,
     or wants a summary of already-ingested material.
   → Requires: query (natural-language question).

3. search_relevant_pages
   → Use when the user wants to retrieve specific pages, passages, or sections from
     stored documents.
   → Requires: query. Optional: doc_id, filename, top_k.

4. store_document_pages
   → Use when the user provides pre-extracted text pages to index directly.

5. list_stored_documents
   → Use when the user asks what documents are available or stored.

Always explain what you are doing and present results clearly.
"""


# ── Build the LangGraph agent ─────────────────────────────────────────────────

async def build_agent():
    """Initialise the MCP client, load tools, and compile the LangGraph graph."""
    client = MultiServerMCPClient(MCP_CLIENT_CONFIG)
    tools = await client.get_tools()

    # Use Amazon Bedrock Claude 3 Sonnet via LangChain's init_chat_model
    model = init_chat_model(
        model=f"bedrock/{LLM_MODEL_ID}",
        region_name=AWS_REGION,
    )
    model_with_tools = model.bind_tools(tools)

    def call_model(state: MessagesState):
        messages = state["messages"]
        # Inject system prompt on first call
        if not any(m.type == "system" for m in messages):
            from langchain_core.messages import SystemMessage
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}

    builder = StateGraph(MessagesState)
    builder.add_node("call_model", call_model)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges("call_model", tools_condition)
    builder.add_edge("tools", "call_model")

    graph = builder.compile()
    return graph


async def run_agent(user_message: str) -> str:
    """
    Run the agent with a single user message and return the final text response.

    Args:
        user_message: The user's input string.

    Returns:
        The agent's final text response.
    """
    from langchain_core.messages import HumanMessage

    graph = await build_agent()
    result = await graph.ainvoke({"messages": [HumanMessage(content=user_message)]})

    # Extract the last AI message
    for msg in reversed(result["messages"]):
        if hasattr(msg, "content") and msg.type == "ai":
            return msg.content
    return str(result["messages"][-1].content)


# ── Interactive CLI ───────────────────────────────────────────────────────────

async def _interactive_loop():
    print("Document Intelligence Agent (type 'exit' to quit)\n")
    graph = await build_agent()

    from langchain_core.messages import HumanMessage, SystemMessage

    conversation: list = [SystemMessage(content=SYSTEM_PROMPT)]

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye.")
            break
        if not user_input:
            continue

        conversation.append(HumanMessage(content=user_input))
        result = await graph.ainvoke({"messages": conversation})
        conversation = result["messages"]

        # Print last AI response
        for msg in reversed(conversation):
            if hasattr(msg, "content") and msg.type == "ai" and not msg.tool_calls:
                print(f"\nAgent: {msg.content}\n")
                break


if __name__ == "__main__":
    asyncio.run(_interactive_loop())
