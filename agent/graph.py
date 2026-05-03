"""LangGraph agent: guardrail → agent ⇄ tools loop."""
from __future__ import annotations

import os
from typing import Annotated, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from agent.guardrails import is_medical_advice, safe_refusal_message
from agent.prompts import SYSTEM_PROMPT
from agent.tools import (
    fetch_github_readme,
    lookup_paper,
    query_assay_data,
    rag_search,
    wiki_search,
    set_vectorstore,
    set_wiki_vectorstore,
)


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    blocked: bool


def _get_llm(tools: list):
    model = os.getenv("MODEL_NAME", "gpt-4o-mini")
    base_url = os.getenv("OPENAI_BASE_URL", "")
    if "google" in base_url.lower():
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=0,
            max_output_tokens=800,
            google_api_key=os.getenv("OPENAI_API_KEY"),
        ).bind_tools(tools)
    return ChatOpenAI(
        model=model,
        temperature=0,
        max_tokens=800,
        base_url=base_url or None,
    ).bind_tools(tools)


# ---------------------------------------------------------------------------
# Routing (stateless helpers — no tool dependency)
# ---------------------------------------------------------------------------

def guardrail_node(state: AgentState) -> dict:
    last = state["messages"][-1]
    query = last.content if isinstance(last, HumanMessage) else ""
    if is_medical_advice(query):
        return {
            "messages": [AIMessage(content=safe_refusal_message())],
            "blocked": True,
        }
    return {"blocked": False}


def route_after_guardrail(state: AgentState) -> Literal["agent_node", END]:
    return END if state.get("blocked") else "agent_node"


def route_after_agent(state: AgentState) -> Literal["tool_node", END]:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tool_node"
    return END


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph(vectorstore, wiki_vectorstore=None, mode: str = "classic"):
    """Build and compile the LangGraph agent.

    mode="classic" uses rag_search (chunk-level retrieval from pharma_ra).
    mode="wiki"    uses wiki_search (pre-compiled wiki pages from pharma_wiki).
    """
    set_vectorstore(vectorstore)
    if wiki_vectorstore is not None:
        set_wiki_vectorstore(wiki_vectorstore)

    search_tool = wiki_search if mode == "wiki" else rag_search
    tools = [search_tool, query_assay_data, fetch_github_readme, lookup_paper]
    tool_map = {t.name: t for t in tools}
    llm = _get_llm(tools)

    def agent_node(state: AgentState) -> dict:
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(state["messages"])
        response = llm.invoke(messages)
        return {"messages": [response]}

    def tool_node(state: AgentState) -> dict:
        last = state["messages"][-1]
        tool_messages: list[ToolMessage] = []
        for call in last.tool_calls:
            fn = tool_map.get(call["name"])
            result = fn.invoke(call["args"]) if fn else f"Unknown tool: {call['name']}"
            tool_messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
        return {"messages": tool_messages}

    g = StateGraph(AgentState)
    g.add_node("guardrail_node", guardrail_node)
    g.add_node("agent_node", agent_node)
    g.add_node("tool_node", tool_node)

    g.add_edge(START, "guardrail_node")
    g.add_conditional_edges("guardrail_node", route_after_guardrail)
    g.add_conditional_edges("agent_node", route_after_agent)
    g.add_edge("tool_node", "agent_node")

    return g.compile()
