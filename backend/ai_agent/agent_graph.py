"""LangGraph agent flow definition."""

from __future__ import annotations

from typing import Dict

from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.chat_models import ChatOpenAI

from .tools import (
    KeywordGeneratorTool,
    ConstitutionSearchTool,
    IPCSearchTool,
    PredictPunishmentTool,
)
from .vector_db import MilvusVectorDB


class AgentState(Dict):
    """Simple state dict for the agent."""


def build_graph() -> StateGraph:
    llm = ChatOpenAI(temperature=0)
    vector_db = MilvusVectorDB()

    keyword_tool = KeywordGeneratorTool()
    constitution_tool = ConstitutionSearchTool(vector_db)
    ipc_tool = IPCSearchTool(vector_db)
    punish_tool = PredictPunishmentTool()

    graph = StateGraph(AgentState)

    def generate_keywords(state: AgentState) -> AgentState:
        query = state.get("query")
        keywords = keyword_tool.run(query)
        state["keywords"] = keywords
        return state

    def search_constitution(state: AgentState) -> AgentState:
        keywords = state.get("keywords") or [state.get("query")]
        results = constitution_tool.run(" ".join(keywords))
        state["constitution_results"] = results
        return state

    def search_ipc(state: AgentState) -> AgentState:
        keywords = state.get("keywords") or [state.get("query")]
        results = ipc_tool.run(" ".join(keywords))
        state["ipc_results"] = results
        return state

    def synthesize(state: AgentState) -> str:
        messages = [
            SystemMessage(
                content="You are a legal assistant answering queries about the Indian Constitution and IPC."
            ),
            HumanMessage(content=f"Query: {state.get('query')}")
        ]
        if state.get("constitution_results"):
            messages.append(
                HumanMessage(content=f"Constitution refs: {state['constitution_results']}")
            )
        if state.get("ipc_results"):
            messages.append(
                HumanMessage(content=f"IPC refs: {state['ipc_results']}")
            )
        answer = llm.invoke(messages).content
        state["answer"] = answer
        return state

    graph.add_node("keywords", generate_keywords)
    graph.add_node("constitution", search_constitution)
    graph.add_node("ipc", search_ipc)
    graph.add_node("synthesize", synthesize)

    graph.set_entry_point("keywords")
    graph.connect("keywords", "constitution")
    graph.connect("constitution", "ipc")
    graph.connect("ipc", "synthesize")

    return graph


def run_agent(query: str) -> str:
    graph = build_graph()
    state = AgentState(query=query)
    result = graph.invoke(state)
    return result["answer"]

