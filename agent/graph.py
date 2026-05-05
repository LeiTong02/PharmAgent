"""LangGraph agent: guardrail → intent → agent ⇄ tools loop."""
from __future__ import annotations

import json
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
    search_reagents,
    wiki_search,
    set_vectorstore,
    set_wiki_vectorstore,
    set_current_query_context,
)
from rag.query_parser import parse_query, parse_query_from_llm_output


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    blocked: bool


# ---------------------------------------------------------------------------
# Intent classification prompt — LLM-based, semantic, no string-matching rules
# ---------------------------------------------------------------------------

_INTENT_SYSTEM_PROMPT = """\
You are a query classifier for a pharmaceutical research assistant.

Classify the intent of the user query and extract key terms.

Intent options (choose exactly ONE):
- entity_lookup      : asking whether an entity/concept is known, exists, or is in the database
                       e.g. "do you know X?", "is there anything about Y?", "tell me about Z"
- concept_definition : asking what something is, how it works, or for an explanation
                       e.g. "what is X?", "explain Y", "how does Z work?"
- framework_or_architecture : asking about a model/system's design, pipeline, or overall structure
                       e.g. "what is the framework of X?", "show the architecture", "how is the model structured?"
- figure_specific    : explicitly requesting a figure, image, diagram, or visual
                       e.g. "what does Figure 3 show?", "show me the figure about Y", "display Fig 2"
- table_or_result    : asking about experimental results, performance metrics, or data in a table
                       e.g. "what are the results on X?", "what does Table 1 report?", "which method is best?"
- general_qa         : any other factual question about the research content

Also extract:
- entities    : specific proper nouns — model names, dataset names, compound names, protein names, author names
                Only include genuine named entities, not generic terms.
- figure_refs : figure numbers mentioned (e.g. "2" from "Figure 2", "3a" from "Fig. 3a")
- table_refs  : table numbers mentioned (e.g. "1" from "Table 1")

Respond ONLY with valid JSON. No markdown, no explanation.
Format: {"intent": "...", "entities": [...], "figure_refs": [...], "table_refs": [...]}
"""


def _get_classifier_llm():
    """Lightweight LLM for intent classification — same backend as agent, no tools, fewer tokens."""
    model = os.getenv("MODEL_NAME", "gpt-4o-mini")
    base_url = os.getenv("OPENAI_BASE_URL", "")
    if "google" in base_url.lower():
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=0,
            max_output_tokens=200,
            google_api_key=os.getenv("OPENAI_API_KEY"),
        )
    return ChatOpenAI(
        model=model,
        temperature=0,
        max_tokens=200,
        base_url=base_url or None,
    )


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


def route_after_guardrail(state: AgentState) -> Literal["intent_node", END]:
    return END if state.get("blocked") else "intent_node"


def intent_node(state: AgentState) -> dict:
    """Classify query intent using the LLM and store the result for downstream tool calls.

    Falls back to the regex-based parse_query() if the LLM call fails for any reason.
    The classified QueryContext is injected into tools.py module-level state so that
    rag_search can pass it directly to smart_retrieve — avoiding repeated classification.
    """
    last = state["messages"][-1]
    query = last.content if isinstance(last, HumanMessage) else ""
    if not query:
        set_current_query_context(None)
        return {}

    qctx = None
    try:
        classifier = _get_classifier_llm()
        response = classifier.invoke([
            SystemMessage(content=_INTENT_SYSTEM_PROMPT),
            HumanMessage(content=query),
        ])
        text = response.content if hasattr(response, "content") else str(response)
        # Strip markdown code fences if the model wrapped in ```json ... ```
        text = text.strip()
        if text.startswith("```"):
            text = "\n".join(
                line for line in text.splitlines()
                if not line.startswith("```")
            ).strip()
        data = json.loads(text)
        qctx = parse_query_from_llm_output(data, query)
    except Exception:
        qctx = parse_query(query)  # regex fallback

    set_current_query_context(qctx)
    return {}


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
    tools = [search_tool, query_assay_data, search_reagents, fetch_github_readme, lookup_paper]
    tool_map = {t.name: t for t in tools}
    llm = _get_llm(tools)

    def agent_node(state: AgentState) -> dict:
        # Strip __VISUAL_CHUNKS__ sentinel from ToolMessage content so the LLM
        # only sees clean text context — visual data is routed separately by the router.
        messages = [SystemMessage(content=SYSTEM_PROMPT)]
        for msg in state["messages"]:
            if isinstance(msg, ToolMessage):
                clean = "\n".join(
                    line for line in (msg.content or "").split("\n")
                    if not line.startswith("__VISUAL_CHUNKS__:")
                )
                messages.append(ToolMessage(content=clean, tool_call_id=msg.tool_call_id))
            else:
                messages.append(msg)
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
    g.add_node("intent_node", intent_node)
    g.add_node("agent_node", agent_node)
    g.add_node("tool_node", tool_node)

    g.add_edge(START, "guardrail_node")
    g.add_conditional_edges("guardrail_node", route_after_guardrail)
    g.add_edge("intent_node", "agent_node")
    g.add_conditional_edges("agent_node", route_after_agent)
    g.add_edge("tool_node", "agent_node")

    return g.compile()
