"""LangGraph agent flow definition."""

from __future__ import annotations

from typing import Annotated, Sequence, TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
import os
from utils.Constants import Constants


from .tools import tools

class AgentState(TypedDict):
    """State for the legal AI agent with message history."""
    messages: Annotated[Sequence[BaseMessage], add_messages]


def build_graph() -> StateGraph:
    """Build and return the compiled LangGraph agent."""
    
    # Initialize LLM with tools
    llm = ChatOpenAI(temperature=0, model=Constants.LLM_MODEL_NAME, api_key=Constants.OPENAI_API_KEY).bind_tools(tools)
    
    # Create tools dictionary for easy lookup
    tools_dict = {tool.name: tool for tool in tools}
    
    def should_continue(state: AgentState) -> str:
        """Check if the last message contains tool calls."""
        last_message = state['messages'][-1]
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"
        else:
            return "end"
    
    def call_llm(state: AgentState) -> AgentState:
        """Call the LLM with the current state and system prompt."""
        system_prompt = SystemMessage(content=Constants.LLM_PROMPT_SYSTEM)
        
        messages = [system_prompt] + list(state['messages'])
        response = llm.invoke(messages)
        return {'messages': [response]}
    
    def execute_tools(state: AgentState) -> AgentState:
        """Execute tool calls from the LLM's response."""
        tool_calls = state['messages'][-1].tool_calls
        results = []
        
        for tool_call in tool_calls:
            tool_name = tool_call['name']
            tool_args = tool_call['args']
            
            print(f"Calling tool: {tool_name} with args: {tool_args}")
            
            if tool_name not in tools_dict:
                result = f"Error: Tool '{tool_name}' not found. Available tools: {list(tools_dict.keys())}"
            else:
                try:
                    # Get the first argument value (tools expect single string input)
                    query = list(tool_args.values())[0] if tool_args else ""
                    result = tools_dict[tool_name].invoke(query)
                except Exception as e:
                    result = f"Error executing tool {tool_name}: {str(e)}"
            
            # Create tool message
            tool_message = ToolMessage(
                tool_call_id=tool_call['id'],
                name=tool_name,
                content=str(result)
            )
            results.append(tool_message)
        
        return {'messages': results}
    
    # Build the graph
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("llm", call_llm)
    graph.add_node("tools", execute_tools)
    
    # Set entry point
    graph.set_entry_point("llm")
    
    # Add conditional edges
    graph.add_conditional_edges(
        "llm",
        should_continue,
        {
            "tools": "tools",
            "end": END
        }
    )
    
    # Add edge from tools back to llm
    graph.add_edge("tools", "llm")
    
    # Compile the graph
    app = graph.compile()
    
    # Create generated directory if it doesn't exist
    os.makedirs("generated", exist_ok=True)
    
    try:
        # Save the graph visualization as a PNG file
        graph_png = app.get_graph().draw_mermaid_png()
        with open("generated/graph_visualization.png", "wb") as f:
            f.write(graph_png)
        print("Graph visualization has been saved as 'generated/graph_visualization.png'")
    except Exception as e:
        print(f"Could not save graph visualization: {e}")
    
    return app


def run_agent(query: str) -> str:
    """Run the agent with a query and return the final answer."""
    app = build_graph()
    
    # Create initial state with user message
    initial_state = {
        "messages": [HumanMessage(content=query)]
    }
    
    # Run the agent
    result = app.invoke(initial_state)
    
    # Return the last message content
    return result["messages"][-1].content


def stream_agent(query: str):
    """Stream the agent execution for real-time responses."""
    app = build_graph()
    
    initial_state = {
        "messages": [HumanMessage(content=query)]
    }
    
    for chunk in app.stream(initial_state, stream_mode="values"):
        if "messages" in chunk:
            last_message = chunk["messages"][-1]
            if hasattr(last_message, 'content') and last_message.content:
                yield last_message.content

