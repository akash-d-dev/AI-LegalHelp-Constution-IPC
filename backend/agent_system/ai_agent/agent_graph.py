"""LangGraph agent flow definition."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Annotated, Sequence, TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain_openai import ChatOpenAI
from agent_system.utils.Constants import Constants

from .tools import tools

# Configure logging with both console and file handlers
def setup_agent_logging():
    """Setup logging configuration for the agent with file output."""
    # Create generated directory if it doesn't exist
    os.makedirs("generated", exist_ok=True)
    
    # Create log file path
    log_file = os.path.join("generated", "agent_execution.log")
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Get the root logger for the agent
    agent_logger = logging.getLogger('agent_system.ai_agent')
    agent_logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    for handler in agent_logger.handlers[:]:
        agent_logger.removeHandler(handler)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    agent_logger.addHandler(file_handler)
    agent_logger.addHandler(console_handler)
    
    return agent_logger

# Setup logging
logger = setup_agent_logging()


class AgentState(TypedDict):
    """State for the legal AI agent with message history."""
    messages: Annotated[Sequence[BaseMessage], add_messages]


def build_graph() -> StateGraph:
    """Build and return the compiled LangGraph agent."""
    logger.info("🏗️ Building agent graph...")
    
    # Initialize LLM with tools
    logger.info(f"🤖 Initializing LLM with model: {Constants.LLM_MODEL_NAME}")
    llm = ChatOpenAI(temperature=0, model=Constants.LLM_MODEL_NAME, api_key=Constants.OPENAI_API_KEY).bind_tools(tools)
    logger.info(f"🛠️ LLM bound with {len(tools)} tools: {[tool.name for tool in tools]}")
    
    # Create tools dictionary for easy lookup
    tools_dict = {tool.name: tool for tool in tools}
    logger.info(f"📚 Tools dictionary created with keys: {list(tools_dict.keys())}")
    
    def should_continue(state: AgentState) -> str:
        """Check if the last message contains tool calls."""
        last_message = state['messages'][-1]
        has_tool_calls = hasattr(last_message, 'tool_calls') and last_message.tool_calls
        
        if has_tool_calls:
            tool_names = [tc['name'] for tc in last_message.tool_calls]
            logger.info(f"🔄 Agent decision: CONTINUE - Found {len(last_message.tool_calls)} tool calls: {tool_names}")
            return "tools"
        else:
            logger.info("🏁 Agent decision: END - No tool calls found, finishing execution")
            return "end"
    
    def call_llm(state: AgentState) -> AgentState:
        """Call the LLM with the current state and system prompt."""
        logger.info("🧠 Entering LLM node...")
        
        # Log current state
        message_count = len(state['messages'])
        logger.info(f"📨 Current state has {message_count} messages")
        
        # Log the last user message if it exists
        user_messages = [msg for msg in state['messages'] if isinstance(msg, HumanMessage)]
        if user_messages:
            last_user_msg = user_messages[-1].content
            logger.info(f"👤 Last user query: '{last_user_msg[:100]}...'")
        
        system_prompt = SystemMessage(content=Constants.LLM_PROMPT_SYSTEM)
        logger.info("📋 System prompt loaded from Constants")
        
        messages = [system_prompt] + list(state['messages'])
        logger.info(f"📤 Sending {len(messages)} messages to LLM...")
        
        try:
            response = llm.invoke(messages)
            logger.info("✅ LLM response received")
            
            # Log if the response contains tool calls
            if hasattr(response, 'tool_calls') and response.tool_calls:
                tool_call_info = []
                for tc in response.tool_calls:
                    tool_call_info.append(f"{tc['name']}({list(tc['args'].keys())})")
                logger.info(f"🔧 LLM wants to call tools: {', '.join(tool_call_info)}")
            else:
                content_preview = response.content[:100] if response.content else "No content"
                logger.info(f"💬 LLM final response: '{content_preview}...'")
            
            return {'messages': [response]}
            
        except Exception as e:
            logger.error(f"❌ Error in LLM call: {str(e)}")
            raise
    
    def execute_tools(state: AgentState) -> AgentState:
        """Execute tool calls from the LLM's response."""
        logger.info("🔧 Entering tools execution node...")
        
        tool_calls = state['messages'][-1].tool_calls
        logger.info(f"🎯 Executing {len(tool_calls)} tool calls...")
        
        results = []
        
        for i, tool_call in enumerate(tool_calls, 1):
            tool_name = tool_call['name']
            tool_args = tool_call['args']
            tool_id = tool_call['id']
            
            logger.info(f"🔨 Tool {i}/{len(tool_calls)}: {tool_name}")
            logger.info(f"📝 Tool args: {tool_args}")
            
            if tool_name not in tools_dict:
                error_msg = f"Error: Tool '{tool_name}' not found. Available tools: {list(tools_dict.keys())}"
                logger.error(f"❌ {error_msg}")
                result = error_msg
            else:
                try:
                    # Get the first argument value (tools expect single string input)
                    query = list(tool_args.values())[0] if tool_args else ""
                    logger.info(f"🔍 Calling {tool_name} with query: '{query[:50]}...'")
                    
                    result = tools_dict[tool_name].invoke(query)
                    result_preview = str(result)[:100] if result else "No result"
                    logger.info(f"✅ Tool {tool_name} completed. Result preview: '{result_preview}...'")
                    
                except Exception as e:
                    error_msg = f"Error executing tool {tool_name}: {str(e)}"
                    logger.error(f"❌ {error_msg}")
                    result = error_msg
            
            # Create tool message
            tool_message = ToolMessage(
                tool_call_id=tool_id,
                name=tool_name,
                content=str(result)
            )
            results.append(tool_message)
            logger.info(f"📨 Tool message created for {tool_name}")
        
        logger.info(f"🏁 All {len(tool_calls)} tools executed, returning results to LLM")
        return {'messages': results}
    
    # Build the graph
    logger.info("🔗 Building graph structure...")
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("llm", call_llm)
    graph.add_node("tools", execute_tools)
    logger.info("📍 Added nodes: llm, tools")
    
    # Set entry point
    graph.set_entry_point("llm")
    logger.info("🚪 Set entry point: llm")
    
    # Add conditional edges
    graph.add_conditional_edges(
        "llm",
        should_continue,
        {
            "tools": "tools",
            "end": END
        }
    )
    logger.info("🔀 Added conditional edges from llm")
    
    # Add edge from tools back to llm
    graph.add_edge("tools", "llm")
    logger.info("➡️ Added edge: tools -> llm")
    
    # Compile the graph
    logger.info("⚙️ Compiling graph...")
    app = graph.compile()
    logger.info("✅ Graph compiled successfully")
    
    # Create generated directory if it doesn't exist
    os.makedirs("generated", exist_ok=True)
    
    try:
        # Save the graph visualization as a PNG file
        logger.info("🎨 Generating graph visualization...")
        graph_png = app.get_graph().draw_mermaid_png()
        with open("generated/graph_visualization.png", "wb") as f:
            f.write(graph_png)
        logger.info("✅ Graph visualization saved as 'generated/graph_visualization.png'")
    except Exception as e:
        logger.warning(f"⚠️ Could not save graph visualization: {e}")
    
    logger.info("🎉 Agent graph build completed successfully")
    return app


def run_agent(query: str, chat_history: Optional[List[dict]] = None) -> str:
    """Run the agent with a query and optional chat history, return the final answer.
    
    Args:
        query: The current user question/query
        chat_history: Optional list of previous messages in format:
                     [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
    
    Returns:
        str: The agent's response to the current query
    """
    Constants.check_env_variables()
    logger.info("🚀 Starting agent execution...")
    logger.info(f"❓ User query: '{query}'")
    
    # Log chat history if provided
    if chat_history:
        logger.info(f"📚 Chat history provided with {len(chat_history)} messages")
        for i, msg in enumerate(chat_history):
            role = msg.get('role', 'unknown')
            content_preview = msg.get('content', '')[:50]
            logger.info(f"  {i+1}. {role}: '{content_preview}...'")
    else:
        logger.info("📚 No chat history provided - starting fresh conversation")
    
    app = build_graph()
    
    # Create initial state with chat history + current query
    messages = []
    
    # Add chat history if provided
    if chat_history:
        for msg in chat_history:
            role = msg.get('role')
            content = msg.get('content', '')
            
            if role == 'user':
                messages.append(HumanMessage(content=content))
            elif role == 'assistant':
                messages.append(AIMessage(content=content))
            elif role == 'system':
                messages.append(SystemMessage(content=content))
            else:
                logger.warning(f"⚠️ Unknown message role: {role}, treating as user message")
                messages.append(HumanMessage(content=content))
    
    # Add current user query
    messages.append(HumanMessage(content=query))
    
    initial_state = {"messages": messages}
    logger.info(f"📋 Initial state created with {len(messages)} total messages")
    
    # Run the agent
    logger.info("🔄 Invoking agent...")
    try:
        result = app.invoke(initial_state)
        logger.info("✅ Agent execution completed")
        
        # Log final result
        final_answer = result["messages"][-1].content
        answer_preview = final_answer[:200] if final_answer else "No content"
        logger.info(f"💡 Final answer preview: '{answer_preview}...'")
        logger.info(f"📊 Total messages in final state: {len(result['messages'])}")
        
        save_agent_conversation_log(query, final_answer, len(result['messages']), chat_history)
        
        return final_answer
        
    except Exception as e:
        logger.error(f"❌ Agent execution failed: {str(e)}")
        raise


def stream_agent(query: str, chat_history: Optional[List[dict]] = None):
    """Stream the agent execution for real-time responses with optional chat history.
    
    Args:
        query: The current user question/query
        chat_history: Optional list of previous messages in format:
                     [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
    
    Yields:
        str: Chunks of the agent's response as they become available
    """
    logger.info("🌊 Starting agent streaming...")
    logger.info(f"❓ User query: '{query}'")
    
    # Log chat history if provided
    if chat_history:
        logger.info(f"📚 Chat history provided with {len(chat_history)} messages")
    else:
        logger.info("📚 No chat history provided - starting fresh conversation")
    
    app = build_graph()
    
    # Create initial state with chat history + current query
    messages = []
    
    # Add chat history if provided
    if chat_history:
        for msg in chat_history:
            role = msg.get('role')
            content = msg.get('content', '')
            
            if role == 'user':
                messages.append(HumanMessage(content=content))
            elif role == 'assistant':
                messages.append(AIMessage(content=content))
            elif role == 'system':
                messages.append(SystemMessage(content=content))
            else:
                logger.warning(f"⚠️ Unknown message role: {role}, treating as user message")
                messages.append(HumanMessage(content=content))
    
    # Add current user query
    messages.append(HumanMessage(content=query))
    
    initial_state = {"messages": messages}
    logger.info(f"📋 Initial state created for streaming with {len(messages)} total messages")
    
    try:
        for i, chunk in enumerate(app.stream(initial_state, stream_mode="values")):
            logger.info(f"📦 Received chunk {i+1}")
            if "messages" in chunk:
                last_message = chunk["messages"][-1]
                if hasattr(last_message, 'content') and last_message.content:
                    content_preview = last_message.content[:100]
                    logger.info(f"📤 Yielding content: '{content_preview}...'")
                    yield last_message.content
        
        logger.info("✅ Agent streaming completed")
        
    except Exception as e:
        logger.error(f"❌ Agent streaming failed: {str(e)}")
        raise


def save_agent_conversation_log(query: str, final_answer: str, total_messages: int, chat_history: Optional[List[dict]] = None):
    """Save the agent conversation to a log file with chat history context."""
    try:
        # Create generated directory if it doesn't exist
        os.makedirs("generated", exist_ok=True)
        
        log_file = os.path.join("generated", "agent_conversations.log")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*100}\n")
            f.write(f"AGENT CONVERSATION LOG\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Total Messages Processed: {total_messages}\n")
            f.write(f"Chat History Messages: {len(chat_history) if chat_history else 0}\n")
            f.write(f"{'='*100}\n\n")
            
            # Log chat history if provided
            if chat_history:
                f.write(f"CHAT HISTORY:\n")
                for i, msg in enumerate(chat_history, 1):
                    role = msg.get('role', 'unknown').upper()
                    content = msg.get('content', '')
                    f.write(f"{i}. {role}: {content}\n")
                f.write(f"\n")
            
            f.write(f"CURRENT USER QUERY:\n")
            f.write(f"{query}\n\n")
            
            f.write(f"AGENT RESPONSE:\n")
            f.write(f"{final_answer}\n\n")
            
            f.write(f"{'='*100}\n\n")
            
    except Exception as e:
        logger.error(f"❌ Error saving conversation log: {e}")


def get_conversation_history_format() -> dict:
    """Return the expected format for chat history parameter.
    
    Returns:
        dict: Example format for chat_history parameter
    """
    return {
        "description": "Chat history should be a list of message dictionaries",
        "format": [
            {"role": "user", "content": "What are fundamental rights?"},
            {"role": "assistant", "content": "Fundamental rights are basic human rights..."},
            {"role": "user", "content": "Can you explain Article 21?"},
            {"role": "assistant", "content": "Article 21 of the Indian Constitution..."}
        ],
        "supported_roles": ["user", "assistant", "system"],
        "note": "Messages will be processed in order to maintain conversation context"
    }

