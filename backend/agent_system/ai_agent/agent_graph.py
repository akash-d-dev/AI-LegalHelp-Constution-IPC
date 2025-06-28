"""LangGraph agent flow definition."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Annotated, Sequence, TypedDict, List, Optional, Union
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr
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
    """State for the legal AI agent with message history and tracking variables."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    db_results: Optional[List[dict]]
    tool_calls_executed: Optional[List[str]]


def build_graph():
    """Build and return the compiled LangGraph agent."""
    logger.info("🏗️ Building agent graph...")
    
    # Initialize LLM with tools
    logger.info(f"🤖 Initializing LLM with model: {Constants.LLM_MODEL_NAME}")
    
    if Constants.LLM_MODEL_NAME == "gpt-4o-mini":
        llm = ChatOpenAI(temperature=0, model=Constants.LLM_MODEL_NAME, api_key=SecretStr(Constants.OPENAI_API_KEY) if Constants.OPENAI_API_KEY else None).bind_tools(tools)
        print("LLM initialized with OpenAI")
    elif Constants.LLM_MODEL_NAME == "gemini-2.0-flash-exp":
        llm = ChatGoogleGenerativeAI(temperature=0, model=Constants.LLM_MODEL_NAME, api_key=SecretStr(Constants.GOOGLE_API_KEY) if Constants.GOOGLE_API_KEY else None).bind_tools(tools)
        print("LLM initialized with Google")
    else:
        raise ValueError(f"Invalid LLM model: {Constants.LLM_MODEL_NAME}")
    
    logger.info(f"🛠️ LLM bound with {len(tools)} tools: {[tool.name for tool in tools]}")
    
    # Create tools dictionary for easy lookup
    tools_dict = {tool.name: tool for tool in tools}
    logger.info(f"📚 Tools dictionary created with keys: {list(tools_dict.keys())}")
    
    def should_continue(state: AgentState) -> str:
        """Check if the last message contains tool calls."""
        last_message = state['messages'][-1]
        
        if isinstance(last_message, AIMessage) and hasattr(last_message, 'tool_calls') and last_message.tool_calls:
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
        
        # Get state tracking information
        db_results = state.get('db_results', []) or []
        tool_calls_executed = state.get('tool_calls_executed', []) or []
        
        # Create state awareness message
        state_awareness_content = ""
        if db_results or tool_calls_executed:
            state_awareness_content = "\n\nCURRENT AGENT STATE:\n"
            
            if tool_calls_executed:
                state_awareness_content += f"Tools executed so far: {', '.join(tool_calls_executed)}\n"
                state_awareness_content += f"Total tool calls: {len(tool_calls_executed)}\n"
            
            if db_results:
                state_awareness_content += f"Database searches performed: {len(db_results)}\n"
                db_tools_used = [result['tool_name'] for result in db_results]
                state_awareness_content += f"Database tools used: {', '.join(set(db_tools_used))}\n"
                
                # Summarize recent database results
                if db_results:
                    state_awareness_content += "\nRecent database results summary:\n"
                    for i, result in enumerate(db_results[-3:], 1):  # Show last 3 results
                        tool_name = result['tool_name']
                        query = result['query']
                        result_preview = result['result'][:100] + "..." if len(result['result']) > 100 else result['result']
                        state_awareness_content += f"{i}. {tool_name} (query: '{query}'): {result_preview}\n"
            
            state_awareness_content += "\nUse this information to decide if you need to search more or if you have sufficient information to provide a comprehensive answer."
        
        system_prompt = SystemMessage(content=Constants.LLM_PROMPT_SYSTEM + state_awareness_content)
        logger.info("📋 System prompt loaded from Constants with state awareness")
        
        messages = [system_prompt] + list(state['messages'])
        logger.info(f"📤 Sending {len(messages)} messages to LLM...")
        
        try:
            response = llm.invoke(messages)
            logger.info("✅ LLM response received")
            
            # Log if the response contains tool calls
            if isinstance(response, AIMessage) and hasattr(response, 'tool_calls') and response.tool_calls:
                tool_call_info = []
                for tc in response.tool_calls:
                    tool_call_info.append(f"{tc['name']}({list(tc['args'].keys())})")
                logger.info(f"🔧 LLM wants to call tools: {', '.join(tool_call_info)}")
            else:
                content_preview = response.content[:100] if response.content else "No content"
                logger.info(f"💬 LLM final response: '{content_preview}...'")
            
            return {
                'messages': [response],
                'db_results': db_results,
                'tool_calls_executed': tool_calls_executed
            }
            
        except Exception as e:
            logger.error(f"❌ Error in LLM call: {str(e)}")
            raise
    
    def execute_tools(state: AgentState) -> AgentState:
        """Execute tool calls from the LLM's response and track results."""
        logger.info("🔧 Entering tools execution node...")
        
        last_message = state['messages'][-1]
        
        if not isinstance(last_message, AIMessage) or not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
            logger.warning("⚠️ No tool calls found in last message")
            return {
                'messages': [],
                'db_results': state.get('db_results', []) or [],
                'tool_calls_executed': state.get('tool_calls_executed', []) or []
            }
        
        tool_calls = last_message.tool_calls
        logger.info(f"🎯 Executing {len(tool_calls)} tool calls...")
        
        # Get current tracking state
        current_db_results = state.get('db_results', []) or []
        current_tool_calls = state.get('tool_calls_executed', []) or []
        
        results = []
        new_db_results = []
        
        for i, tool_call in enumerate(tool_calls, 1):
            tool_name = tool_call['name']
            tool_args = tool_call['args']
            tool_id = tool_call['id']
            
            logger.info(f"🔨 Tool {i}/{len(tool_calls)}: {tool_name}")
            logger.info(f"📝 Tool args: {tool_args}")
            
            # Track tool call
            current_tool_calls.append(tool_name)
            
            if tool_name not in tools_dict:
                error_msg = f"Error: Tool '{tool_name}' not found. Available tools: {list(tools_dict.keys())}"
                logger.error(f"❌ {error_msg}")
                result = error_msg
            else:
                try:
                    # Handle tool arguments more carefully
                    if not tool_args:
                        logger.warning(f"⚠️ No arguments provided for tool {tool_name}")
                        query = ""
                    else:
                        # Log all arguments for debugging
                        logger.info(f"🔍 All tool arguments: {tool_args}")
                        
                        # For generate_keywords, we expect the 'query' parameter to be the complete user query
                        if tool_name == 'generate_keywords':
                            if 'query' in tool_args:
                                query = tool_args['query']
                                logger.info(f"🔑 generate_keywords received query: '{query}'")
                            else:
                                # Fallback to first argument value
                                query = list(tool_args.values())[0] if tool_args else ""
                                logger.warning(f"⚠️ generate_keywords missing 'query' parameter, using: '{query}'")
                        else:
                            # For other tools, use the first argument value
                            query = list(tool_args.values())[0] if tool_args else ""
                    
                    logger.info(f"🔍 Calling {tool_name} with query: '{query[:50]}...'")
                    
                    result = tools_dict[tool_name].invoke(query)
                    result_preview = str(result)[:100] if result else "No result"
                    logger.info(f"✅ Tool {tool_name} completed. Result preview: '{result_preview}...'")
                    
                    # Track database results for search tools
                    if tool_name in ['search_constitution', 'search_ipc', 'enhanced_cross_domain_legal_search']:
                        db_result_entry = {
                            'tool_name': tool_name,
                            'query': query,
                            'result': str(result),
                            'timestamp': datetime.now().isoformat()
                        }
                        new_db_results.append(db_result_entry)
                        logger.info(f"📊 Tracked DB result for {tool_name}")
                    
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
        
        # Update tracking state
        updated_db_results = current_db_results + new_db_results
        
        logger.info(f"🏁 All {len(tool_calls)} tools executed")
        logger.info(f"📈 State tracking: {len(current_tool_calls)} total tool calls, {len(updated_db_results)} DB results")
        
        return {
            'messages': results,
            'db_results': updated_db_results,
            'tool_calls_executed': current_tool_calls
        }
    
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
    # Check environment variables
    try:
        Constants.check_env_variables()
    except Exception as e:
        logger.error(f"❌ Environment check failed: {e}")
        raise
    
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
    
    initial_state = {
        "messages": messages,
        "db_results": [],
        "tool_calls_executed": []
    }
    logger.info(f"📋 Initial state created with {len(messages)} total messages")
    
    # Run the agent
    logger.info("🔄 Invoking agent...")
    try:
        result = app.invoke(initial_state)  # type: ignore
        logger.info("✅ Agent execution completed")
        
        # Log final result
        final_answer = result["messages"][-1].content
        answer_preview = final_answer[:200] if final_answer else "No content"
        logger.info(f"💡 Final answer preview: '{answer_preview}...'")
        logger.info(f"📊 Total messages in final state: {len(result['messages'])}")
        
        # Save conversation and state tracking logs
        save_agent_conversation_log(query, final_answer, len(result['messages']), chat_history)
        save_agent_state_tracking(query, result.get('db_results', []), result.get('tool_calls_executed', []))
        
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
    
    initial_state = {
        "messages": messages,
        "db_results": [],
        "tool_calls_executed": []
    }
    logger.info(f"📋 Initial state created for streaming with {len(messages)} total messages")
    
    try:
        for i, chunk in enumerate(app.stream(initial_state, stream_mode="values")):  # type: ignore
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


def save_agent_state_tracking(query: str, db_results: List[dict], tool_calls_executed: List[str]):
    """Save the agent state tracking information to a JSON file."""
    try:
        # Create generated directory if it doesn't exist
        os.makedirs("generated", exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        state_file = os.path.join("generated", f"agent_state_{timestamp}.json")
        
        state_data = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "tool_calls_executed": tool_calls_executed,
            "total_tool_calls": len(tool_calls_executed),
            "db_results": db_results,
            "total_db_results": len(db_results),
            "unique_tools_used": list(set(tool_calls_executed)),
            "db_tools_used": [result['tool_name'] for result in db_results]
        }
        
        import json
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(state_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Agent state tracking saved to: {state_file}")
        logger.info(f"📊 State summary: {len(tool_calls_executed)} tool calls, {len(db_results)} DB results")
        
    except Exception as e:
        logger.error(f"❌ Error saving state tracking: {e}")


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
